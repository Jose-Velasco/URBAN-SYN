"""
Validates three complementary things:
1. Forward equivalence: candidate-only logits match full linear → gather.
2. Gradient equivalence: the optimization preserves the gradients used during training.
3. Property-based equivalence: Hypothesis exercised the identity across many tensor dimensions, road counts, candidate sizes, candidate IDs (including duplicates), weights, and hidden states.
"""

import torch

from generator.function_g_fc import FunctionGFC
from hypothesis import given, settings, strategies as st

def _build_model() -> FunctionGFC:
    """Build a small Function G model for candidate-scoring tests."""
    config = {
        "road_emb_size": 4,
        "time_emb_size": 3,
        "hidden_size": 5,
        "lstm_layer_num": 1,
        "dropout_p": 0.0,
        "dis_weight": 0.5,
    }

    data_feature = {
        "road_num": 11,
        "road_pad": 10,
        "time_size": 20,
        "time_pad": 19,
    }

    return FunctionGFC(config, data_feature)


def test_candidate_only_scores_match_full_linear_then_gather():
    """Candidate-only scoring should reproduce the original logits."""
    torch.manual_seed(7)

    model = _build_model()

    attn_hidden = torch.randn(
        2,
        model.hidden_size,
    )

    # Include repeated candidates because the existing preprocessing code
    # can pad candidate sets by repeating valid candidate IDs.
    candidate_set = torch.tensor(
        [
            [1, 3, 5, 3],
            [2, 7, 2, 9],
        ],
        dtype=torch.long,
    )

    # Original implementation:
    # score every road and then gather the candidates.
    expected = torch.gather(
        model.out_fc(attn_hidden),
        dim=1,
        index=candidate_set,
    )

    # Optimized implementation:
    # score only the candidate output rows.
    actual = model._score_candidates(
        attn_hidden,
        candidate_set,
    )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-6,
        atol=1e-7,
    )


def test_candidate_only_scores_preserve_gradients():
    """Candidate-only scoring should preserve training gradients."""
    torch.manual_seed(11)

    model = _build_model()

    candidate_set = torch.tensor(
        [
            [1, 3, 5, 3],
            [2, 7, 2, 9],
        ],
        dtype=torch.long,
    )

    # Use an arbitrary upstream gradient so that all candidate positions
    # contribute to the comparison.
    coefficient = torch.randn(2, 4)

    # ----------------------------------------------------------
    # Original implementation: full linear layer, then gather.
    # ----------------------------------------------------------
    old_hidden = torch.randn(
        2,
        model.hidden_size,
        requires_grad=True,
    )

    old_score = torch.gather(
        model.out_fc(old_hidden),
        dim=1,
        index=candidate_set,
    )

    (old_score * coefficient).sum().backward()

    expected_hidden_grad = old_hidden.grad.detach().clone()
    expected_weight_grad = model.out_fc.weight.grad.detach().clone()
    expected_bias_grad = model.out_fc.bias.grad.detach().clone()

    model.zero_grad(set_to_none=True)

    # ----------------------------------------------------------
    # Optimized implementation: candidate output rows only.
    # ----------------------------------------------------------
    new_hidden = (
        old_hidden.detach()
        .clone()
        .requires_grad_(True)
    )

    new_score = model._score_candidates(
        new_hidden,
        candidate_set,
    )

    (new_score * coefficient).sum().backward()

    torch.testing.assert_close(
        new_hidden.grad,
        expected_hidden_grad,
        rtol=1e-6,
        atol=1e-7,
    )

    torch.testing.assert_close(
        model.out_fc.weight.grad,
        expected_weight_grad,
        rtol=1e-6,
        atol=1e-7,
    )

    torch.testing.assert_close(
        model.out_fc.bias.grad,
        expected_bias_grad,
        rtol=1e-6,
        atol=1e-7,
    )

@given(
    batch_size=st.integers(min_value=1, max_value=8),
    hidden_size=st.integers(min_value=1, max_value=32),
    road_num=st.integers(min_value=3, max_value=100),
    candidate_size=st.integers(min_value=1, max_value=12),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=100, deadline=None) # PyTorch initialization can have enough timing variance that this can produce noisy deadline failures We're testing correctness, not performance here
def test_candidate_only_scoring_matches_original_for_random_inputs(
    batch_size,
    hidden_size,
    road_num,
    candidate_size,
    seed,
):
    """Candidate-only scoring should equal full scoring for random inputs.

    Hypothesis explores different tensor shapes and road/candidate counts,
    while the generated seed varies the model parameters, hidden states,
    and candidate road IDs.
    """
    torch.manual_seed(seed)

    config = {
        "road_emb_size": 4,
        "time_emb_size": 3,
        "hidden_size": hidden_size,
        "lstm_layer_num": 1,
        "dropout_p": 0.0,
        "dis_weight": 0.5,
    }

    data_feature = {
        "road_num": road_num,
        "road_pad": road_num - 1,
        "time_size": 20,
        "time_pad": 19,
    }

    model = FunctionGFC(config, data_feature)

    attn_hidden = torch.randn(
        batch_size,
        hidden_size,
    )

    # Do not use the final road ID because it represents road_pad.
    # Sampling with replacement intentionally allows duplicate candidates,
    # matching the padding behavior seen in the existing pipeline.
    candidate_set = torch.randint(
        low=0,
        high=road_num - 1,
        size=(batch_size, candidate_size),
        dtype=torch.long,
    )

    # Original implementation: compute every road logit, then gather.
    expected = torch.gather(
        model.out_fc(attn_hidden),
        dim=1,
        index=candidate_set,
    )

    # Optimized implementation: evaluate only the requested output rows.
    actual = model._score_candidates(
        attn_hidden,
        candidate_set,
    )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )
