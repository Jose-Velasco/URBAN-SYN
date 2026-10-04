from __future__ import annotations

from collections.abc import Iterable

from pipeline.config import PipelineConfig
from pipeline.stage import Stage


PIPELINE_GROUP_ORDER: tuple[str, ...] = (
    "prepare",
    "pretrain-road",
    "prepare-region",
    "pretrain-region",
    "train",
    "generate",
)

VALID_GROUPS = frozenset(PIPELINE_GROUP_ORDER)


def build_stages(config: PipelineConfig) -> tuple[Stage, ...]:
    """Build the ordered TS-TrajGen stages for the supplied configuration.

    The first wrapper iteration intentionally uses a fixed pipeline order rather
    than a dependency solver. Each command mirrors the individually tested
    commands used during the TS-TrajGen refactor.

    Args:
        config:
            Parsed wrapper configuration.

    Returns:
        All pipeline stages in execution order. Both generation implementations
        are included here; ``stages_for_group`` filters them according to
        ``config.generation_mode``.
    """
    dataset = config.dataset_name
    dataset_dir = config.dataset_dir
    repo_dir = config.repo_dir
    save_dir = config.road_save_dir

    mm_train = f"{dataset}_mm_train.csv"
    mm_test = f"{dataset}_mm_test.csv"

    road_pretrain_train = f"{dataset}_pretrain_input_train.csv"
    road_pretrain_eval = f"{dataset}_pretrain_input_eval.csv"
    road_pretrain_test = f"{dataset}_pretrain_input_test.csv"

    region_train = f"{dataset}_mm_region_train.csv"
    region_eval = f"{dataset}_mm_region_eval.csv"
    region_test = f"{dataset}_mm_region_test.csv"

    region_pretrain_train = f"{dataset}_region_pretrain_input_train.csv"
    region_pretrain_eval = f"{dataset}_region_pretrain_input_eval.csv"
    region_pretrain_test = f"{dataset}_region_pretrain_input_test.csv"

    geo_filename = f"{dataset}.geo"
    rel_filename = f"{dataset}.rel"
    graph_filename = f"{dataset}.graph"
    processed_traj_filename = f"{dataset}_traj_mm_processed.csv"

    partition_filename = f"tmppartition{config.partition_k}"

    stages: list[Stage] = []

    # ------------------------------------------------------------------
    # prepare
    # ------------------------------------------------------------------
    build_inputs_command = [
        "python",
        str(config.workspace_dir / "build_tstrajgen_inputs.py"),
        "--network_path",
        str(config.network_path),
        "--fmm_match_path",
        str(config.fmm_match_path),
        "--parquet_path",
        str(config.parquet_path),
        "--trip_id_map_csv",
        str(config.trip_id_map_path),
        "--config",
        str(config.model_config_path),
        "--out_dir",
        str(dataset_dir),
        "--log_dir",
        str(config.log_dir),
        "--dataset_name",
        dataset,
        "--min_len",
        str(config.min_len),
        "--min_delta_seconds",
        str(config.min_delta_seconds),
        "--train_ratio",
        str(config.train_ratio),
    ]

    if config.interpolate_intermediate_edges:
        build_inputs_command.append("--interpolate_intermediate_edges")
    else:
        build_inputs_command.append("--no-interpolate_intermediate_edges")

    if config.max_train_trajectories is not None:
        build_inputs_command.extend(
            [
                "--max_train_trajectories",
                str(config.max_train_trajectories),
            ]
        )

    if config.max_test_trajectories is not None:
        build_inputs_command.extend(
            [
                "--max_test_trajectories",
                str(config.max_test_trajectories),
            ]
        )

    stages.append(
        _stage(
            name="build-inputs",
            group="prepare",
            description=(
                "Build TS-TrajGen .geo/.rel files and road-level train/test "
                "map-matched trajectories."
            ),
            cwd=config.workspace_dir,
            command=build_inputs_command,
        )
    )

    stages.append(
        _stage(
            name="preprocess-road",
            group="prepare",
            description=(
                "Build road adjacency/GPS artifacts and road-level pretraining "
                "trajectory inputs."
            ),
            cwd=repo_dir,
            command=[
                "python",
                "-m",
                "script.preprocess_pretrain_input",
                "--dataset_name",
                dataset,
                "--data_root",
                str(config.datasets_dir),
                "--dataset_prefix",
                dataset,
                "--config",
                str(config.model_config_path),
            ],
        )
    )

    stages.append(
        _stage(
            name="kahip-format",
            group="prepare",
            description="Convert the road graph to KaHIP graph format.",
            cwd=repo_dir,
            command=[
                "python",
                "./script/process_kahip_graph_format.py",
                "--dataset_name",
                dataset,
                "--data_root",
                str(config.datasets_dir),
                "--geo_filename",
                geo_filename,
                "--rel_filename",
                rel_filename,
                "--graph_filename",
                graph_filename,
                "--rid2new_filename",
                "rid2new.json",
                "--new2rid_filename",
                "new2rid.json",
            ],
        )
    )

    stages.append(
        _stage(
            name="kahip-partition",
            group="prepare",
            description="Partition the road graph with KaHIP.",
            cwd=repo_dir,
            command=[
                "/opt/KaHIP/build/kaffpa",
                str(dataset_dir / graph_filename),
                "--k",
                str(config.partition_k),
                f"--preconfiguration={config.partition_preconfiguration}",
                "--output_filename",
                str(dataset_dir / partition_filename),
            ],
        )
    )

    stages.append(
        _stage(
            name="process-kahip",
            group="prepare",
            description=(
                "Convert KaHIP partitions into connected TS-TrajGen regions."
            ),
            cwd=repo_dir,
            command=[
                "python",
                "./script/process_kaffpa_res.py",
                "--dataset_name",
                dataset,
                "--data_root",
                str(config.datasets_dir),
                "--partition_filename",
                partition_filename,
                "--new2rid_filename",
                "new2rid.json",
                "--adjacent_filename",
                "adjacent_list.json",
                "--region2rid_filename",
                "region2rid.json",
                "--rid2region_filename",
                "rid2region.json",
            ],
        )
    )

    stages.append(
        _stage(
            name="construct-region-adjacency",
            group="prepare",
            description=(
                "Build region adjacency matrix and region boundary-road lookup."
            ),
            cwd=repo_dir,
            command=[
                "python",
                "./script/construct_traffic_zone_relation.py",
                "--dataset_name",
                dataset,
                "--data_root",
                str(config.datasets_dir),
                "--rel_filename",
                rel_filename,
                "--adjacent_filename",
                "adjacent_list.json",
                "--rid2region_filename",
                "rid2region.json",
                "--region2rid_filename",
                "region2rid.json",
                "--region_adj_mx_filename_output",
                "region_adj_mx.npz",
                "--region_adjacent_filename_output",
                "region_adjacent_list.json",
            ],
        )
    )

    stages.append(
        _stage(
            name="map-region-trajectories",
            group="prepare",
            description=(
                "Map road-level trajectories to region-level train/eval/test "
                "trajectories."
            ),
            cwd=repo_dir,
            command=[
                "python",
                "-m",
                "script.map_region_traj",
                "--dataset_name",
                dataset,
                "--data_root",
                str(config.datasets_dir),
                "--rid2region_filename",
                "rid2region.json",
                "--train_mm_filename",
                mm_train,
                "--test_mm_filename",
                mm_test,
                "--train_region_filename",
                region_train,
                "--eval_region_filename",
                region_eval,
                "--test_region_filename",
                region_test,
                "--config",
                str(config.model_config_path),
            ],
        )
    )

    stages.append(
        _stage(
            name="encode-region-trajectories",
            group="prepare",
            description=(
                "Encode region trajectories into region-model pretraining inputs "
                "and build region GPS lookup."
            ),
            cwd=repo_dir,
            command=[
                "python",
                "-m",
                "script.encode_region_traj",
                "--dataset_name",
                dataset,
                "--data_root",
                str(config.datasets_dir),
                "--rid2region_filename",
                "rid2region.json",
                "--region2rid_filename",
                "region2rid.json",
                "--rid_gps_filename",
                "rid_gps.json",
                "--region_adjacent_filename",
                "region_adjacent_list.json",
                "--train_region_filename",
                region_train,
                "--eval_region_filename",
                region_eval,
                "--test_region_filename",
                region_test,
                "--region_gps_output_filename",
                "region_gps.json",
                "--train_output_filename",
                region_pretrain_train,
                "--eval_output_filename",
                region_pretrain_eval,
                "--test_output_filename",
                region_pretrain_test,
                "--config",
                str(config.model_config_path),
            ],
        )
    )

    stages.append(
        _stage(
            name="construct-region-distance",
            group="prepare",
            description=(
                "Build road-length and train-derived region-distance helper "
                "artifacts."
            ),
            cwd=repo_dir,
            command=[
                "python",
                "-m",
                "script.construct_region_dist",
                "--dataset_name",
                dataset,
                "--data_root",
                str(config.datasets_dir),
                "--geo_filename",
                geo_filename,
                "--road_length_filename",
                "road_length.json",
                "--rid2region_filename",
                "rid2region.json",
                "--region_gps_filename",
                "region_gps.json",
                "--train_mm_filename",
                mm_train,
                "--test_mm_filename",
                mm_test,
                "--processed_traj_filename",
                processed_traj_filename,
                "--region_dist_filename",
                "region_count_dist.npy",
                "--config",
                str(config.model_config_path),
            ],
        )
    )

    # Train-derived helper statistics used by adversarial training. These are
    # intentionally constructed from the training split only.
    stages.append(
        _stage(
            name="road-od-routes",
            group="prepare",
            description="Build train-only road-level OD distinct-route history.",
            cwd=repo_dir,
            command=[
                "python",
                "-m",
                "script.generate_od_distinct_route",
                "--dataset_name",
                dataset,
                "--data_root",
                str(config.datasets_dir),
                "--traj_filename",
                mm_train,
                "--route_column",
                "rid_list",
                "--gps_filename",
                "rid_gps.json",
                "--output_filename",
                "od_distinct_route.json",
            ],
        )
    )

    stages.append(
        _stage(
            name="road-time-distribution",
            group="prepare",
            description="Build train-only road-level travel-time distribution.",
            cwd=repo_dir,
            command=[
                "python",
                "-m",
                "script.generate_time_distribution",
                "--dataset_name",
                dataset,
                "--data_root",
                str(config.datasets_dir),
                "--traj_filename",
                mm_train,
                "--geo_filename",
                geo_filename,
                "--output_filename",
                "road_time_distribution.npy",
            ],
        )
    )

    stages.append(
        _stage(
            name="region-transfer",
            group="prepare",
            description="Build train-only region transition probabilities.",
            cwd=repo_dir,
            command=[
                "python",
                "-m",
                "script.count_region_transfer",
                "--dataset_name",
                dataset,
                "--data_root",
                str(config.datasets_dir),
                "--rid2region_filename",
                "rid2region.json",
                "--region_adjacent_filename",
                "region_adjacent_list.json",
                "--output_filename",
                "region_transfer_prob.json",
                "--traj_filename",
                mm_train,
            ],
        )
    )

    stages.append(
        _stage(
            name="region-od-routes",
            group="prepare",
            description="Build train-only region-level OD distinct-route history.",
            cwd=repo_dir,
            command=[
                "python",
                "-m",
                "script.generate_od_distinct_route",
                "--dataset_name",
                dataset,
                "--data_root",
                str(config.datasets_dir),
                "--traj_filename",
                region_train,
                "--route_column",
                "region_list",
                "--gps_filename",
                "region_gps.json",
                "--output_filename",
                "region_od_distinct_route.json",
            ],
        )
    )

    stages.append(
        _stage(
            name="region-time-distribution",
            group="prepare",
            description="Build train-only region-level travel-time distribution.",
            cwd=repo_dir,
            command=[
                "python",
                "-m",
                "script.generate_time_distribution_region",
                "--dataset_name",
                dataset,
                "--data_root",
                str(config.datasets_dir),
                "--region2rid_filename",
                "region2rid.json",
                "--train_region_filename",
                region_train,
                "--output_filename",
                "region_time_distribution.npy",
            ],
        )
    )

    # ------------------------------------------------------------------
    # pretrain-road
    # ------------------------------------------------------------------
    stages.append(
        _stage(
            name="pretrain-road-gat",
            group="pretrain-road",
            description="Pretrain road-level Function H/GAT.",
            cwd=repo_dir,
            command=[
                "python",
                "pretrain_gat_fc.py",
                "--dataset_name",
                dataset,
                "--data_root",
                str(config.datasets_dir),
                "--device",
                config.device,
                "--debug",
                "False",
                "--train",
                "True",
                "--config",
                str(config.model_config_path),
                "--geo_path",
                str(dataset_dir / geo_filename),
                "--rel_filename",
                rel_filename,
                "--map_manager_cache_dir",
                str(dataset_dir),
                "--adjacent_np_filename",
                "adjacent_mx.npz",
                "--node_feature_filename",
                "node_feature.pt",
                "--rid_gps_filename",
                "rid_gps.json",
                "--train_filename",
                road_pretrain_train,
                "--eval_filename",
                road_pretrain_eval,
                "--test_filename",
                road_pretrain_test,
                "--save_dir",
                str(save_dir),
                "--save_file_name",
                "gat_fc.pt",
                "--temp_dir",
                str(repo_dir / "temp" / dataset / "gat"),
            ],
        )
    )

    stages.append(
        _stage(
            name="pretrain-road-function-g",
            group="pretrain-road",
            description="Pretrain road-level Function G.",
            cwd=repo_dir,
            command=[
                "python",
                "pretrain_function_g_fc.py",
                "--dataset_name",
                dataset,
                "--data_root",
                str(config.datasets_dir),
                "--device",
                config.device,
                "--train",
                "True",
                "--config",
                str(config.model_config_path),
                "--geo_path",
                str(dataset_dir / geo_filename),
                "--train_filename",
                road_pretrain_train,
                "--eval_filename",
                road_pretrain_eval,
                "--test_filename",
                road_pretrain_test,
                "--save_dir",
                str(save_dir),
                "--save_file_name",
                "function_g_fc.pt",
                "--temp_dir",
                str(repo_dir / "temp" / dataset / "function_g"),
            ],
        )
    )

    # ------------------------------------------------------------------
    # prepare-region
    # ------------------------------------------------------------------
    stages.append(
        _stage(
            name="prepare-region-features",
            group="prepare-region",
            description=(
                "Build region node features from the pretrained road GAT "
                "representation."
            ),
            cwd=repo_dir,
            command=[
                "python",
                "prepare_region_feature.py",
                "--dataset_name",
                dataset,
                "--device",
                config.device,
                "--data_root",
                str(config.datasets_dir),
                "--geo_path",
                str(dataset_dir / geo_filename),
                "--map_manager_cache_dir",
                str(dataset_dir),
                "--save_folder",
                str(save_dir),
                "--save_file_name",
                "gat_fc.pt",
                "--adjacent_np_filename",
                "adjacent_mx.npz",
                "--node_feature_filename",
                "node_feature.pt",
                "--rid2region_filename",
                "rid2region.json",
                "--region2rid_filename",
                "region2rid.json",
                "--region_feature_filename",
                "region_feature.pt",
                "--config",
                str(config.model_config_path),
            ],
        )
    )

    # ------------------------------------------------------------------
    # pretrain-region
    # ------------------------------------------------------------------
    stages.append(
        _stage(
            name="pretrain-region-function-g",
            group="pretrain-region",
            description="Pretrain region-level Function G.",
            cwd=repo_dir,
            command=[
                "python",
                "pretrain_region_function_g_fc.py",
                "--dataset_name",
                dataset,
                "--data_root",
                str(config.datasets_dir),
                "--region2rid_filename",
                "region2rid.json",
                "--train_filename",
                region_pretrain_train,
                "--eval_filename",
                region_pretrain_eval,
                "--test_filename",
                region_pretrain_test,
                "--save_dir",
                str(save_dir),
                "--save_file_name",
                "region_function_g_fc.pt",
                "--temp_dir",
                str(repo_dir / "temp" / dataset / "region_function_g"),
                "--device",
                config.device,
                "--config",
                str(config.model_config_path),
                "--train",
            ],
        )
    )

    stages.append(
        _stage(
            name="pretrain-region-gat",
            group="pretrain-region",
            description="Pretrain region-level Function H/GAT.",
            cwd=repo_dir,
            command=[
                "python",
                "pretrain_region_gat_fc.py",
                "--dataset_name",
                dataset,
                "--data_root",
                str(config.datasets_dir),
                "--region2rid_filename",
                "region2rid.json",
                "--adjacent_np_filename",
                "region_adj_mx.npz",
                "--node_feature_filename",
                "region_feature.pt",
                "--region_dist_filename",
                "region_count_dist.npy",
                "--train_filename",
                region_pretrain_train,
                "--eval_filename",
                region_pretrain_eval,
                "--test_filename",
                region_pretrain_test,
                "--save_dir",
                str(save_dir),
                "--save_file_name",
                "region_gat_fc.pt",
                "--temp_dir",
                str(repo_dir / "temp" / dataset / "region_gat"),
                "--device",
                config.device,
                "--config",
                str(config.model_config_path),
                "--train",
            ],
        )
    )

    # ------------------------------------------------------------------
    # train
    # ------------------------------------------------------------------
    stages.append(
        _stage(
            name="train-road-gan",
            group="train",
            description="Run road-level TS-TrajGen adversarial training.",
            cwd=repo_dir,
            command=[
                "python",
                "train_gan.py",
                "--dataset_name",
                dataset,
                "--data_root",
                str(config.datasets_dir),
                "--exp_id",
                str(config.gan_exp_id),
                "--save_dir",
                str(config.road_gan_save_dir),
                "--pretrain_g_file",
                str(save_dir / "function_g_fc.pt"),
                "--pretrain_gat_file",
                str(save_dir / "gat_fc.pt"),
                "--trajectory_filename",
                mm_train,
                "--node_feature_filename",
                "node_feature.pt",
                "--adjacent_np_filename",
                "adjacent_mx.npz",
                "--adjacent_list_filename",
                "adjacent_list.json",
                "--rid_gps_filename",
                "rid_gps.json",
                "--road_length_filename",
                "road_length.json",
                "--od_distinct_route_filename",
                "od_distinct_route.json",
                "--road_time_dist_filename",
                "road_time_distribution.npy",
                "--geo_filename",
                geo_filename,
                "--map_manager_cache_dir",
                str(dataset_dir),
                "--device",
                config.device,
                "--config",
                str(config.model_config_path),
                "--pretrain_discriminator",
                _bool_arg(config.pretrain_discriminator),
                "--debug",
                _bool_arg(config.gan_debug),
            ],
        )
    )

    stages.append(
        _stage(
            name="train-region-gan",
            group="train",
            description="Run region-level TS-TrajGen adversarial training.",
            cwd=repo_dir,
            command=[
                "python",
                "train_region_gan.py",
                "--dataset_name",
                dataset,
                "--data_root",
                str(config.datasets_dir),
                "--trajectory_file",
                region_train,
                "--pretrain_region_function_g_file",
                str(save_dir / "region_function_g_fc.pt"),
                "--pretrain_region_gat_file",
                str(save_dir / "region_gat_fc.pt"),
                "--save_folder",
                str(config.region_gan_save_dir),
                "--adjacent_list_file",
                "adjacent_list.json",
                "--rid_gps_file",
                "rid_gps.json",
                "--road_length_file",
                "road_length.json",
                "--region_adjacent_list_file",
                "region_adjacent_list.json",
                "--region_adj_mx_file",
                "region_adj_mx.npz",
                "--region_feature_file",
                "region_feature.pt",
                "--region_dist_file",
                "region_count_dist.npy",
                "--region_transfer_file",
                "region_transfer_prob.json",
                "--rid2region_file",
                "rid2region.json",
                "--region2rid_file",
                "region2rid.json",
                "--region_gps_file",
                "region_gps.json",
                "--region_od_file",
                "region_od_distinct_route.json",
                "--road_time_dist_file",
                "road_time_distribution.npy",
                "--region_time_dist_file",
                "region_time_distribution.npy",
                "--device",
                config.device,
                "--config",
                str(config.model_config_path),
                "--debug",
                _bool_arg(config.gan_debug),
                "--pretrain_discriminator",
                _bool_arg(config.pretrain_discriminator),
            ],
        )
    )

    # ------------------------------------------------------------------
    # generate
    # ------------------------------------------------------------------
    stages.append(
        _stage(
            name="generate-pretrained",
            group="generate",
            description=(
                "Generate trajectories using the original-style pretrained "
                "Function G/H checkpoints."
            ),
            cwd=repo_dir,
            command=[
                "python",
                "our_model_generate.py",
                "--dataset_name",
                dataset,
                "--data_root",
                str(config.datasets_dir),
                "--true_traj_file",
                mm_test,
                "--generated_trace_output_file",
                "TS_TrajGen_non_gan_generated_output.csv",
                "--pretrain_gen_file",
                str(save_dir / "function_g_fc.pt"),
                "--pretrain_gat_file",
                str(save_dir / "gat_fc.pt"),
                "--pretrain_region_gen_file",
                str(save_dir / "region_function_g_fc.pt"),
                "--pretrain_region_gat_file",
                str(save_dir / "region_gat_fc.pt"),
                "--geo_path",
                str(dataset_dir / geo_filename),
                "--map_manager_cache_dir",
                str(dataset_dir),
                "--node_feature_file",
                "node_feature.pt",
                "--adjacent_np_file",
                "adjacent_mx.npz",
                "--region_adjacent_np_file",
                "region_adj_mx.npz",
                "--region_feature_file",
                "region_feature.pt",
                "--region2rid_file",
                "region2rid.json",
                "--adjacent_list_file",
                "adjacent_list.json",
                "--rid_gps_file",
                "rid_gps.json",
                "--road_length_file",
                "road_length.json",
                "--region_adjacent_list_file",
                "region_adjacent_list.json",
                "--region_dist_file",
                "region_count_dist.npy",
                "--region_transfer_file",
                "region_transfer_prob.json",
                "--rid2region_file",
                "rid2region.json",
                "--road_time_distribution_file",
                "road_time_distribution.npy",
                "--region_time_distribution_file",
                "region_time_distribution.npy",
                "--config",
                str(config.model_config_path),
                "--device",
                config.device,
            ],
        )
    )

    stages.append(
        _stage(
            name="generate-gan",
            group="generate",
            description=(
                "Generate trajectories using the full GAN-trained road and "
                "region GeneratorV4 checkpoints."
            ),
            cwd=repo_dir,
            command=[
                "python",
                "our_model_generate_using_gan.py",
                "--dataset_name",
                dataset,
                "--data_root",
                str(config.datasets_dir),
                "--true_traj_file",
                mm_test,
                "--generated_trace_output_file",
                "TS_TrajGen_GAN_generated_output.csv",
                "--road_gan_generator_file",
                str(
                    config.road_gan_save_dir
                    / f"adversarial_3_generator_{config.gan_exp_id}.pt"
                ),
                "--region_gan_generator_file",
                str(config.region_gan_save_dir / "adversarial_region_generator.pt"),
                "--geo_path",
                str(dataset_dir / geo_filename),
                "--map_manager_cache_dir",
                str(dataset_dir),
                "--node_feature_file",
                "node_feature.pt",
                "--adjacent_np_file",
                "adjacent_mx.npz",
                "--region_adjacent_np_file",
                "region_adj_mx.npz",
                "--region_feature_file",
                "region_feature.pt",
                "--region2rid_file",
                "region2rid.json",
                "--rid2region_file",
                "rid2region.json",
                "--adjacent_list_file",
                "adjacent_list.json",
                "--rid_gps_file",
                "rid_gps.json",
                "--road_length_file",
                "road_length.json",
                "--region_adjacent_list_file",
                "region_adjacent_list.json",
                "--region_dist_file",
                "region_count_dist.npy",
                "--region_transfer_file",
                "region_transfer_prob.json",
                "--road_time_distribution_file",
                "road_time_distribution.npy",
                "--region_time_distribution_file",
                "region_time_distribution.npy",
                "--config",
                str(config.model_config_path),
                "--device",
                config.device,
            ],
        )
    )

    return tuple(stages)


def stages_for_group(
    config: PipelineConfig,
    group: str,
) -> tuple[Stage, ...]:
    """Return stages belonging to one wrapper group.

    The ``generate`` group is filtered using ``generation.mode`` from the
    wrapper configuration.

    Args:
        config:
            Parsed pipeline configuration.
        group:
            One of the values in ``PIPELINE_GROUP_ORDER``.

    Returns:
        Ordered stages for the requested group.

    Raises:
        ValueError:
            If ``group`` is not a supported pipeline group.
    """
    if group not in VALID_GROUPS:
        allowed = ", ".join(PIPELINE_GROUP_ORDER)
        raise ValueError(f"Unknown pipeline group {group!r}. Expected one of: {allowed}")

    stages = tuple(stage for stage in build_stages(config) if stage.group == group)

    if group != "generate":
        return stages

    if config.generation_mode == "pretrained":
        return tuple(stage for stage in stages if stage.name == "generate-pretrained")

    if config.generation_mode == "gan":
        return tuple(stage for stage in stages if stage.name == "generate-gan")

    # PipelineConfig validates that the only remaining value is "both".
    return stages


def stages_for_all(config: PipelineConfig) -> tuple[Stage, ...]:
    """Return every stage in wrapper execution order."""
    ordered: list[Stage] = []

    for group in PIPELINE_GROUP_ORDER:
        ordered.extend(stages_for_group(config, group))

    return tuple(ordered)


def _stage(
    *,
    name: str,
    group: str,
    description: str,
    cwd,
    command: Iterable[str],
) -> Stage:
    """Create a Stage while normalizing its command to an immutable tuple."""
    return Stage(
        name=name,
        group=group,
        description=description,
        cwd=cwd,
        command=tuple(str(part) for part in command),
    )


def _bool_arg(value: bool) -> str:
    """Format a boolean for TS-TrajGen arguments parsed with ``str2bool``."""
    return "True" if value else "False"
