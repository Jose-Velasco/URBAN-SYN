"""One converter per baseline. Each has convert(files, options) -> Converted."""
from evaluation.converters import difftraj, diffrntraj, lstm_trajgan, mobilitygpt, tstrajgen

# Model name (as used in eval_config.yaml and on the command line) -> converter function.
CONVERTERS = {
    "difftraj": difftraj.convert,
    "diffrntraj": diffrntraj.convert,
    "tstrajgen": tstrajgen.convert,
    "mobilitygpt": mobilitygpt.convert,
    "lstm_trajgan": lstm_trajgan.convert,
}
