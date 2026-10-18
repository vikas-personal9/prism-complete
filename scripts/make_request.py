import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
"""Produce a JSON file for `az ml online-endpoint invoke --request-file ...`."""
import argparse,json
from pathlib import Path
from src.ingest import create_event
p=argparse.ArgumentParser();p.add_argument('--run-id',default='WF001-R10');p.add_argument('--stage',default='checkpoint');p.add_argument('--out',default='examples/request.json');a=p.parse_args()
e=create_event(a.run_id,a.stage)
Path(a.out).parent.mkdir(parents=True,exist_ok=True)
Path(a.out).write_text(json.dumps({'stage':e['stage'],'features':e['features']},indent=2))
print('Wrote',a.out)
