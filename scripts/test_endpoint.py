import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
"""Invoke local predictor (default) or deployed AML endpoint (--azure)."""
import argparse,json
from src.ingest import create_event
from src.predict import predict
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',default='WF001-R10')
    p.add_argument('--stage',default='checkpoint',choices=['start','checkpoint'])
    p.add_argument('--azure',action='store_true');args=p.parse_args()
    event=create_event(args.run_id,args.stage)
    print(json.dumps({'request':event,'response':predict(event['features'],args.stage,
                      backend='azure' if args.azure else 'local')},indent=2))
