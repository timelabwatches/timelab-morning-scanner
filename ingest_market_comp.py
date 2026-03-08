#!/usr/bin/env python3
import json
from pathlib import Path

DB = Path('data/market_comp_db.json')


def ingest(row: dict) -> None:
    doc = {"comparables": []}
    if DB.exists():
        doc = json.loads(DB.read_text(encoding='utf-8'))
    comps = doc.setdefault('comparables', [])
    comps.append(row)
    DB.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--json', required=True, help='Comparable row JSON string')
    args = p.parse_args()
    ingest(json.loads(args.json))
    print('ok')