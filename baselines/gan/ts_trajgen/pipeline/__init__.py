"""TS-TrajGen wrapper pipeline package.

This package contains the lightweight orchestration layer used to run the
refactored TS-TrajGen baseline inside the container.

The first wrapper iteration intentionally keeps responsibilities small:

- ``config`` loads wrapper/runtime configuration.
- ``stage`` defines the Stage data model.
- ``stages`` defines the ordered TS-TrajGen commands.
- ``runner`` executes stages sequentially and stops on failure.
"""

from pipeline.config import PipelineConfig, load_pipeline_config
from pipeline.runner import PipelineRunner, PipelineStageError
from pipeline.stage import Stage

__all__ = [
    "PipelineConfig",
    "PipelineRunner",
    "PipelineStageError",
    "Stage",
    "load_pipeline_config",
]
