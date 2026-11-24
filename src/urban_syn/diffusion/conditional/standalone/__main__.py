from .runner import main

if __name__ == '__main__':
    result = main()
    if isinstance(result, dict) and result.get('status') == 'missing_inputs':
        raise SystemExit(2)
