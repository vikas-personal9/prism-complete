import json
import argparse
from src.pipeline import process_event
from src.ingest import create_event
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
"""Locally reproduce Event Hub -> Function -> scoring -> reasoning -> result store."""
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id', default='WF001-R10')
    ap.add_argument('--stage', default='checkpoint',
                    choices=['start', 'checkpoint'])
    a = ap.parse_args()
    print(json.dumps(process_event(create_event(
        a.run_id, a.stage), backend='local'), indent=2))
