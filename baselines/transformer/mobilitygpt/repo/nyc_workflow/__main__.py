"""One command interface for the NYC workflow."""
import argparse
import importlib
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'metrics', 'evaluate', 'summarize', 'check'))
    if not argv or argv[0] in ('-h', '--help'):
        parser.print_help()
        return 0
    command = argv.pop(0)
    modules = {'run': 'pipeline', 'metrics': 'metrics', 'evaluate': 'evaluation',
               'summarize': 'html_report', 'check': 'tests'}
    if command not in modules:
        parser.error(f'Unknown command: {command}')
    module = importlib.import_module(f'nyc_workflow.{modules[command]}')
    try:
        return module.main(argv)
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        print(f'{command} failed: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
