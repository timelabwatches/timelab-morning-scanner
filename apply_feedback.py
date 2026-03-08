#!/usr/bin/env python3
import json
from pathlib import Path

from timelab_core.model_engine import register_feedback


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True, help='feedback json')
    args = ap.parse_args()
    entry = json.loads(args.json)
    register_feedback(entry)
    print('stored')


if __name__ == '__main__':
    main()