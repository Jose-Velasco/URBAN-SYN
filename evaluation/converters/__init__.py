"""One converter per model. Each has convert(files, options) -> Converted.

files holds the model's files from eval_config.yaml, plus network_<key> for the files of the road network its road
tables use (the NETWORK constant of its module; paths under networks: in eval_config.yaml).
"""
from evaluation.converters import difftraj, diffrntraj, lstm_trajgan, mobilitygpt, tstrajgen, urban_syn

# Model name (as used in eval_config.yaml and on the command line) -> converter function.
CONVERTERS = {
    "difftraj": difftraj.convert,
    "diffrntraj": diffrntraj.convert,
    "tstrajgen": tstrajgen.convert,
    "mobilitygpt": mobilitygpt.convert,
    "lstm_trajgan": lstm_trajgan.convert,
    "urban_syn": urban_syn.convert,
}
