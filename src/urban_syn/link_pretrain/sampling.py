"""Seed-centered directed K-hop and mixed structural negative sampling.

Graph topology is read-only. Every proposed negative is checked against the
complete original positive-edge set, including held-out supervision edges.
"""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor

from .config import NegativeSamplingSettings
from .data import LinkBatch, SamplingStats
from .graph import FullPositiveIndex, directed_edge_keys


class TrainingPositiveSelector:
    """Choose one outgoing training-supervision positive for each seed road.

    Incoming edges, held-out labels, and message-only edges are not included.
    Source-index grouping is built once and reused across selections/epochs.
    """

    def __init__(self, positive_edges: Tensor, num_nodes: int) -> None:
        """Index outgoing training-label positives by source road.

        Args:
            positive_edges: CPU ``[2, E]`` PyG indices containing only
                supervised training positives, not message or held-out edges.
            num_nodes: Number of nodes in the road graph.

        Raises:
            ValueError: If the source links are invalid or empty.
        """
        directed_edge_keys(positive_edges, num_nodes)
        if positive_edges.size(1) == 0:
            raise ValueError("Training supervision must contain at least one positive edge.")
        # Sorting by source lets each road's outgoing training links occupy
        # one contiguous slice, found quickly using the cumulative offsets.
        sorted_rows = torch.argsort(positive_edges[0], stable=True)
        sorted_src = positive_edges[0, sorted_rows]
        self._destinations = positive_edges[1, sorted_rows]
        degrees = torch.bincount(sorted_src, minlength=num_nodes)
        self._offsets = torch.cat((degrees.new_zeros(1), degrees.cumsum(0)))
        self.eligible_seeds = torch.nonzero(degrees > 0).flatten()
        self.num_nodes = num_nodes

    def select(self, seeds: Tensor, generator: torch.Generator) -> Tensor:
        """Choose one random labeled outgoing positive for each eligible seed.

        Args:
            seeds: One-dimensional CPU long tensor of source node indices.
            generator: PyTorch generator controlling positive selection.

        Returns:
            CPU long tensor of destinations aligned with the input seeds.
            The caller controls how seeds are shuffled or weighted.

        Raises:
            ValueError: If an index is invalid or lacks a supervised outgoing
                positive, or the input tensor has an unsupported shape/device.
        """
        if seeds.dtype != torch.long or seeds.ndim != 1 or seeds.device.type != "cpu":
            raise ValueError("Seeds must be a one-dimensional CPU long tensor.")
        if seeds.numel() == 0:
            return seeds.clone()
        if bool((seeds < 0).any()) or bool((seeds >= self.num_nodes).any()):
            raise ValueError("Seed index is outside the graph.")
        begin = self._offsets[seeds]
        degrees = self._offsets[seeds + 1] - begin
        if bool((degrees == 0).any()):
            raise ValueError(
                "Only seed roads with training-supervision outgoing links are eligible."
            )
        # Independent uniform choices avoid privileging roads with more links.
        # Scale one random value per seed by that seed's outgoing link count.
        # This chooses one positive uniformly without expanding all link rows.
        choice = (torch.rand(seeds.numel(), generator=generator) * degrees).long()
        return self._destinations[begin + choice]


class KHopCandidates:
    """Find directed 2..K-hop negatives using a compact outgoing-neighbor lookup.

    The lookup is built once from training-only message edges. Each query then
    visits only roads reachable from its source, rather than scanning every
    graph edge. The complete original graph is used only to reject real links.
    """

    def __init__(
        self, message_edges: Tensor, full_positives: FullPositiveIndex,
        num_nodes: int, k_hops: int
    ) -> None:
        """Build a CPU lookup of outgoing neighbors in PyG road-index order.

        Args:
            message_edges: Training-only CPU directed edge index ``[2, E]``.
            full_positives: Original-graph index for excluding direct edges.
            num_nodes: Number of road nodes in the graph.
            k_hops: Maximum outgoing shortest-path distance, at least two.

        Raises:
            ValueError: If the graph sizes disagree or ``k_hops`` is below two.
        """
        directed_edge_keys(message_edges, num_nodes)
        if num_nodes != full_positives.num_nodes or k_hops < 2:
            raise ValueError("Candidate graph size must match the full graph; k_hops >= 2.")

        self.full_positives = full_positives
        self.num_nodes = num_nodes
        self.k_hops = k_hops

        # CSR (compressed sparse row) stores all outgoing neighbors in one
        # tensor instead of keeping a Python list for every road.
        # First group directed edges by source road. Sorting once is much
        # cheaper than repeatedly scanning the entire graph during sampling.
        order = torch.argsort(message_edges[0], stable=True)
        sources = message_edges[0, order]
        self._neighbors = message_edges[1, order].contiguous()

        # offsets[i] is the first position of road i's neighbors; offsets[i+1]
        # is the end (exclusive). Example: 0 -> [2, 4], 1 -> [3] gives
        # neighbors=[2, 4, 3] and offsets starting with [0, 2, 3].
        degrees = torch.bincount(sources, minlength=num_nodes)
        self._offsets = torch.cat((degrees.new_zeros(1), degrees.cumsum(0)))

    def candidates(self, source: int) -> Tensor:
        """Return sorted, valid outgoing destinations within 2..K hops.

        Args:
            source: Zero-based PyG index of the source road.

        Returns:
            Unique CPU long road indices reachable through training-message
            edges within ``k_hops``. The source and any direct edges present
            in the full original graph are excluded, including held-out links.

        Raises:
            ValueError: If ``source`` is outside the graph.
        """
        if not 0 <= source < self.num_nodes:
            raise ValueError("Source road is outside the graph.")

        # Explore the graph one hop at a time, following edge direction.
        # 'visited' prevents cycles and repeated paths from adding a road
        # multiple times; 'frontier' holds only the roads discovered at the
        # previous hop. Reaching the same road again cannot improve its
        # shortest distance, so we do not need to explore it twice.
        visited = {source}
        frontier = [source]
        for _ in range(self.k_hops):
            next_frontier: list[int] = []
            for road in frontier:
                start = int(self._offsets[road])
                stop = int(self._offsets[road + 1])

                # CSR gives this road's outgoing edges directly. A dead-end
                # road simply has start == stop, so it adds no neighbors.
                for neighbor in self._neighbors[start:stop].tolist():
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)

            if not next_frontier:
                break
            frontier = next_frontier

        # Sorting retains the same stable candidate order as the old PyG
        # search. Sampling randomness is still controlled by the caller's
        # torch.Generator; the lookup itself uses no random numbers.
        visited.remove(source)
        if not visited:
            return torch.empty(0, dtype=torch.long)
        reachable = torch.tensor(sorted(visited), dtype=torch.long)

        # One-hop neighbors are real links, not negatives. Check against the
        # FULL graph, not only message edges: held-out direct links must also
        # be excluded to avoid accidentally treating positives as negatives.
        is_positive = self.full_positives.contains(
            torch.full_like(reachable, source), reachable
        )
        return reachable[~is_positive]


class UniformNegativeSampler:
    """Draw globally valid negatives with bounded rejection and exact fallback.

    Avoid already-chosen destinations when feasible; reuse only when the
    complete valid negative pool is smaller than the requested row amount.
    """

    def __init__(self, full_positives: FullPositiveIndex) -> None:
        """Keep the original-edge lookup used to validate sampled negatives.

        Args:
            full_positives: Membership index containing every real directed link.
        """
        self.full_positives = full_positives
        self.num_nodes = full_positives.num_nodes

    def _all_valid(self, source: int) -> Tensor:
        """Enumerate all legal global negatives as a last-resort fallback.

        Args:
            source: Zero-based source node index.

        Returns:
            CPU long tensor of destinations that are neither the source nor
            real outgoing edges in the original graph.
        """
        destinations = torch.arange(self.num_nodes, dtype=torch.long)
        valid = (destinations != source) & ~self.full_positives.contains(
            torch.full_like(destinations, source), destinations
        )
        return destinations[valid]

    def draw(
        self, source: int, count: int, excluded: Tensor, generator: torch.Generator
    ) -> Tensor | None:
        """Draw globally valid destinations with distinctness when possible.

        Args:
            source: Zero-based source road index.
            count: Number of global negatives requested for this source.
            excluded: Already-selected negative destinations to avoid when
                possible, for example the hard negatives in a mixed row.
            generator: PyTorch generator controlling random draws.

        Returns:
            CPU long tensor containing ``count`` destinations, or ``None``
            when there is no legal negative destination for the source.
            Repeated destinations are allowed only if distinct choices run out.

        Raises:
            ValueError: If the requested count or source index is invalid.
        """
        if count < 0 or not 0 <= source < self.num_nodes:
            raise ValueError("Invalid uniform sampling source or amount.")
        if count == 0:
            return torch.empty(0, dtype=torch.long)
        chosen: list[int] = []
        seen = set(excluded.tolist())

        # Draw candidate IDs in batches instead of checking one road at a time.
        # Oversampling offsets rejected real edges, self-links, and duplicates.
        # Limit retries so dense graphs do not cause an unbounded loop.
        max_attempts = 8
        for _ in range(max_attempts):
            remaining = count - len(chosen)
            if remaining == 0:
                break

            # At least 16 draws keep small requests efficient; larger requests
            # draw four times the remaining amount to reduce repeated calls.
            draw_count = max(16, 4 * remaining)
            values = torch.randint(
                self.num_nodes,
                (draw_count,),
                generator=generator
            )
            is_valid = (values != source) & ~self.full_positives.contains(
                torch.full_like(values, source),
                values
            )
            for candidate in values[is_valid].tolist():
                if candidate in seen:
                    continue
                chosen.append(candidate)
                seen.add(candidate)
                if len(chosen) == count:
                    break

        if len(chosen) < count:
            # Rejection sampling did not find enough distinct destinations.
            # Scan the full graph only in this fallback, not for every source.
            all_valid = self._all_valid(source)
            if all_valid.numel() == 0:
                return None

            already_seen = torch.tensor(list(seen), dtype=torch.long)
            available = all_valid[~torch.isin(all_valid, already_seen)]
            remaining = count - len(chosen)
            take = min(remaining, available.numel())
            if take > 0:
                # Shuffle the remaining distinct destinations before taking
                # the amount needed to finish the row.
                order = torch.randperm(available.numel(), generator=generator)
                selected = available[order[:take]]
                chosen.extend(selected.tolist())

            if len(chosen) < count:
                # Only repeat valid negatives when fewer distinct ones exist
                # than the requested row size (including excluded hard ones).
                remaining = count - len(chosen)
                repeat_indices = torch.randint(
                    all_valid.numel(),
                    (remaining,),
                    generator=generator
                )
                chosen.extend(all_valid[repeat_indices].tolist())

        return torch.tensor(chosen, dtype=torch.long)


class NoValidNegativesError(ValueError):
    """A complete requested batch has no usable structural negatives."""


class NegativeSampler(Protocol):
    """Shared batch contract for local and mixed negative strategies."""

    def sample(
        self, sources: Tensor, destinations: Tensor, generator: torch.Generator
    ) -> LinkBatch:
        """Sample negatives aligned to each supervised positive pair.

        Args:
            sources: CPU long vector of positive source node indices.
            destinations: CPU long vector of matching positive destinations.
            generator: PyTorch generator controlling negative draws.

        Returns:
            ``LinkBatch`` containing retained positives, per-positive
            negative destinations, and sampling statistics.
        """
        ...


def _check_positives(
    sources: Tensor, destinations: Tensor, membership: FullPositiveIndex
) -> None:
    """Validate that requested training pairs are aligned real directed links.

    Args:
        sources: CPU long vector of zero-based source indices.
        destinations: Matching CPU long vector of destination indices.
        membership: Full-graph index used to verify the requested links.

    Raises:
        ValueError: If shapes, dtypes, devices, or pairs are invalid.
    """
    valid_dtype = sources.dtype == torch.long and destinations.dtype == torch.long
    valid_shape = sources.ndim == 1 and sources.shape == destinations.shape
    on_cpu = sources.device.type == "cpu" and destinations.device.type == "cpu"
    if not (valid_dtype and valid_shape and on_cpu):
        raise ValueError("Positive sources/destinations must be aligned CPU long vectors.")
    if sources.numel() == 0:
        raise ValueError("A sampling request needs at least one positive link.")
    if not bool(membership.contains(sources, destinations).all()):
        raise ValueError("Input contains a pair absent from the full positive graph.")


def _pick(pool: Tensor, count: int, generator: torch.Generator) -> Tensor:
    """Select ``count`` distinct pool values without replacement.

    Args:
        pool: CPU long vector of unique candidate indices.
        count: Number of values to select, no larger than the pool size.
        generator: PyTorch generator used to shuffle the pool.

    Returns:
        CPU long vector of selected candidate indices.
    """
    order = torch.randperm(pool.numel(), generator=generator)
    return pool[order[:count]]


def _link_batch(
    sources: Tensor, destinations: Tensor, rows: list[Tensor], flags: list[Tensor],
    amount: int, requested_hard: int, empty_hard_pools: int, requested: int,
    full_positives: FullPositiveIndex
) -> LinkBatch:
    """Assemble validated negative rows and report sampling coverage.

    Args:
        sources: Retained positive sources aligned with ``rows``.
        destinations: Matching retained positive destinations.
        rows: One negative-destination tensor per retained positive.
        flags: Per-row boolean indicators for K-hop hard negatives.
        amount: Required negatives per positive.
        requested_hard: Target hard-negative count before pool shortages.
        empty_hard_pools: Count of sampled sources with no local candidates.
        requested: Number of positive pairs before any skips.
        full_positives: Full-graph positive lookup for final validation.

    Returns:
        A source-aligned ``LinkBatch`` containing negative samples and stats.

    Raises:
        NoValidNegativesError: If no requested positive can be retained.
        ValueError: If an output negative is a direct edge or self-pair.
    """
    if not rows:
        raise NoValidNegativesError("No valid negative samples were available for this batch.")
    negatives = torch.stack(rows)
    hard_mask = torch.stack(flags)
    if bool(full_positives.contains(sources[:, None], negatives).any()):
        raise ValueError("Negative sampler returned an existing directed positive edge.")
    if bool((sources[:, None] == negatives).any()):
        raise ValueError("Negative sampler returned a source-to-self pair.")
    # Repetitions count how many slots duplicated a negative within its row.
    repeat_count = sum(amount - torch.unique(row).numel() for row in rows)
    n_hard = int(hard_mask.sum().item())
    return LinkBatch(
        positive_src=sources,
        positive_dst=destinations,
        negative_dst=negatives,
        hard_mask=hard_mask,
        stats=SamplingStats(
            requested_positives=requested,
            retained_positives=len(rows),
            skipped_positives=requested - len(rows),
            requested_hard=requested_hard,
            drawn_hard=n_hard,
            drawn_uniform=len(rows) * amount - n_hard,
            repeated_negatives=int(repeat_count),
            empty_hard_pools=empty_hard_pools
        )
    )


class KHopNegativeSampler:
    """Sample pure hard negatives, skipping seeds with no 2..K candidates."""

    def __init__(self, candidates: KHopCandidates, amount: int) -> None:
        """Configure pure hard-negative sampling.

        Args:
            candidates: Directed K-hop candidate provider.
            amount: Number of negatives per retained positive pair.
        """
        if amount < 1:
            raise ValueError("Negative amount must be positive.")
        self.candidates = candidates
        self.amount = amount

    def sample(
        self, sources: Tensor, destinations: Tensor, generator: torch.Generator
    ) -> LinkBatch:
        """Sample K-hop negatives for each supervised positive pair.

        Args:
            sources: CPU long tensor of positive source indices.
            destinations: CPU long tensor of aligned positive destinations.
            generator: PyTorch generator for candidate selection.

        Returns:
            ``LinkBatch`` containing only hard negatives. Sources without
            local candidates are skipped and counted; undersized nonempty
            pools may repeat destinations to satisfy ``amount``.

        Raises:
            NoValidNegativesError: If all requested sources lack negatives.
        """
        _check_positives(sources, destinations, self.candidates.full_positives)
        kept_sources, kept_destinations, rows, masks = [], [], [], []
        empty = 0
        for source, destination in zip(sources.tolist(), destinations.tolist()):
            pool = self.candidates.candidates(source)
            if pool.numel() == 0:
                empty += 1
                continue
            # Prefer distinct candidates; repeat only when the hard-negative
            # pool contains fewer roads than the configured row amount.
            selected = _pick(pool, min(self.amount, pool.numel()), generator)
            if selected.numel() < self.amount:
                remaining = self.amount - selected.numel()
                repeat_indices = torch.randint(
                    pool.numel(),
                    (remaining,),
                    generator=generator
                )
                selected = torch.cat((selected, pool[repeat_indices]))

            kept_sources.append(source)
            kept_destinations.append(destination)
            rows.append(selected)
            masks.append(torch.ones(self.amount, dtype=torch.bool))
        return _link_batch(
            torch.tensor(kept_sources, dtype=torch.long),
            torch.tensor(kept_destinations, dtype=torch.long),
            rows,
            masks,
            self.amount,
            sources.numel() * self.amount,
            empty,
            sources.numel(),
            self.candidates.full_positives
        )


class MixedNegativeSampler:
    """Compose directed K-hop and validated global negatives per source.

    The hard fraction is targeted across the entire batch, not per positive.
    Short local pools shift requested slots to other capable seeds when
    possible, then fall back to validated global negatives.
    """

    def __init__(
        self, candidates: KHopCandidates, uniform: UniformNegativeSampler,
        amount: int, hard_fraction: float
    ) -> None:
        """Compose local hard and validated global negative sampling.

        Args:
            candidates: Directed K-hop candidate provider.
            uniform: Global sampler using the same full-positive reference.
            amount: Total negatives requested per positive pair.
            hard_fraction: Target share of hard negatives across each batch.
        """
        if amount < 1 or not 0 <= hard_fraction <= 1:
            raise ValueError("Expected positive amount and a hard fraction in [0, 1].")
        if candidates.full_positives is not uniform.full_positives:
            raise ValueError("Hard and uniform samplers must use the same full-edge reference.")
        self.candidates = candidates
        self.uniform = uniform
        self.amount = amount
        self.hard_fraction = hard_fraction

    def sample(
        self, sources: Tensor, destinations: Tensor, generator: torch.Generator
    ) -> LinkBatch:
        """Mix directed K-hop and validated global negative destinations.

        Args:
            sources: CPU long vector of supervised positive sources.
            destinations: CPU long vector of corresponding positives.
            generator: PyTorch generator controlling the entire batch draw.

        Returns:
            ``LinkBatch`` targeting the configured batch-wide hard fraction.
            Unavailable hard slots use validated global negatives, and
            achieved counts and any skipped sources are recorded.

        Raises:
            NoValidNegativesError: If every requested source is unsampleable.
        """
        _check_positives(sources, destinations, self.candidates.full_positives)
        count = sources.numel()
        requested_hard = round(count * self.amount * self.hard_fraction)
        if requested_hard > 0:
            pools = [self.candidates.candidates(source) for source in sources.tolist()]
        else:
            # Skip potentially expensive graph traversal for all-uniform batches.
            pools = [torch.empty(0, dtype=torch.long) for _ in range(count)]

        # Each source contributes at most one slot per unique hard candidate,
        # up to the configured negatives per positive. Shuffling these slots
        # allocates the batch-wide hard target without favoring earlier rows.
        capacities = [min(self.amount, pool.numel()) for pool in pools]
        available_slots = torch.repeat_interleave(
            torch.arange(count),
            torch.tensor(capacities, dtype=torch.long)
        )
        hard_quota = torch.zeros(count, dtype=torch.long)
        if available_slots.numel() > 0:
            shuffled_slots = torch.randperm(
                available_slots.numel(),
                generator=generator
            )
            selected_slots = available_slots[shuffled_slots[:requested_hard]]
            hard_quota = torch.bincount(selected_slots, minlength=count)

        kept_sources, kept_destinations, rows, flags = [], [], [], []
        for i, (source, destination) in enumerate(zip(sources.tolist(), destinations.tolist())):
            hard_count = int(hard_quota[i])
            if hard_count > 0:
                hard = _pick(pools[i], hard_count, generator)
            else:
                hard = torch.empty(0, dtype=torch.long)

            # Fill the remaining row slots globally, while avoiding the hard
            # destinations selected for this source whenever possible.
            uniform_count = self.amount - hard.numel()
            uniform = self.uniform.draw(source, uniform_count, hard, generator)
            if uniform is None:
                # No global negatives exist. This row is invalid even if no
                # hard candidates exist (the graph contains every direct link).
                continue
            row = torch.cat((hard, uniform))
            hard_flags = torch.ones(hard.numel(), dtype=torch.bool)
            uniform_flags = torch.zeros(uniform.numel(), dtype=torch.bool)
            row_flags = torch.cat((hard_flags, uniform_flags))
            # Avoid systematically placing hard negatives before global ones.
            order = torch.randperm(self.amount, generator=generator)
            kept_sources.append(source)
            kept_destinations.append(destination)
            rows.append(row[order])
            flags.append(row_flags[order])

        if requested_hard > 0:
            empty_hard_pools = sum(pool.numel() == 0 for pool in pools)
        else:
            empty_hard_pools = 0

        return _link_batch(
            torch.tensor(kept_sources, dtype=torch.long),
            torch.tensor(kept_destinations, dtype=torch.long),
            rows,
            flags,
            self.amount,
            requested_hard,
            empty_hard_pools,
            count,
            self.candidates.full_positives
        )


def build_negative_sampler(
    message_edges: Tensor, full_positives: FullPositiveIndex,
    num_nodes: int, settings: NegativeSamplingSettings
) -> NegativeSampler:
    """Construct a local or mixed negative sampler from graph references.

    Args:
        message_edges: Training-only CPU PyG edge index for K-hop discovery.
        full_positives: Full-graph membership index for false-negative checks.
        num_nodes: Number of road nodes in the indexed graph.
        settings: Configured negative strategy, K, amount, and hard fraction.

    Returns:
        ``NegativeSampler`` using the selected strategy and shared topology.

    Raises:
        ValueError: If the strategy or sampling settings are invalid.
    """
    candidates = KHopCandidates(
        message_edges,
        full_positives,
        num_nodes,
        settings.k_hops
    )
    if settings.strategy == "khop":
        return KHopNegativeSampler(candidates, settings.amount)
    if settings.strategy == "mixed":
        uniform = UniformNegativeSampler(full_positives)
        return MixedNegativeSampler(
            candidates,
            uniform,
            settings.amount,
            settings.hard_fraction
        )
    raise ValueError(f"Unsupported negative strategy: {settings.strategy}")
