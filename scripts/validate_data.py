import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
from src.features import load_sources,build_features,build_task_summary,build_telemetry_summary

def main():
    s=load_sources('data')
    f=build_features('data')
    t=build_task_summary('data')
    c=build_telemetry_summary('data')
    report={'datasets':{k:{'rows':len(v),'columns':list(v.columns)} for k,v in s.items()},
            'joined_feature_rows':len(f),'workflows':int(f.workflow_id.nunique()),
            'task_summaries':len(t),'telemetry_summaries':len(c),
            'all_runs_successful':bool(f.result_state.eq('SUCCESS').all()),
            'runtime_range_minutes':[round(float(f.runtime_min.min()),3),round(float(f.runtime_min.max()),3)],
            'checkpoint_features_use_end_time':False}
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
