"""Focused NYC workflow and shared production-fix regressions."""
import argparse
from pathlib import Path
import unittest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    directory = Path(__file__).resolve().parent
    suite = unittest.defaultTestLoader.discover(str(directory), top_level_dir=str(directory.parents[1]))
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
