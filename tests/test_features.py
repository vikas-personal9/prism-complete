import json
from pathlib import Path
import pytest
from src.features import (build_features,build_task_summary,build_telemetry_summary,
                          sample_event,START_FEATURES,CHECKPOINT_FEATURES,load_sources)
from src.pipeline import flags_for,process_event
from ml.score import run

DATA='data'

def test_exact_source_schemas():
    d=load_sources(DATA)
    assert {x:len(y) for x,y in d.items()} == {'runs':50,'tasks':150,'cluster':250,'spark':250}
    assert set(d['runs'])=={'workflow_id','workflow_name','run_id','start_time','end_time','result_state','cluster_id'}
    assert d['runs']['result_state'].eq('SUCCESS').all()

def test_feature_join_and_checkpoint():
    df=build_features(DATA)
    assert len(df)==50 and df.run_id.nunique()==50
    assert df.remaining_min.gt(0).all() and df.elapsed_min.gt(0).all()
    assert all(col in df for col in START_FEATURES+CHECKPOINT_FEATURES)
    assert len(build_task_summary(DATA))==len(build_telemetry_summary(DATA))==50

def test_no_target_leakage():
    assert not {'runtime_min','remaining_min','end_time','result_state'}.intersection(START_FEATURES+CHECKPOINT_FEATURES)
    df=build_features(DATA)
    first=df[df.prior_count==0]
    assert first.prior_runtime_min.isna().all()
    event=sample_event(df.iloc[0])
    assert 'runtime_min' not in event['features'] and 'end_time' not in event['features']

def test_scoring_contract_and_failures():
    df=build_features(DATA)
    e=sample_event(df.iloc[0], 'checkpoint')
    pred=run({'stage':e['stage'],'features':e['features']})
    assert pred['prediction_minutes']>=0 and pred['target']=='remaining_min'
    assert 'error' in run({'stage':'start','features':{}})

def test_end_to_end_persists_only_recommendation(monkeypatch,tmp_path):
    monkeypatch.setenv('PRISM_LOCAL_RESULTS',str(tmp_path))
    monkeypatch.setenv('PRISM_STORAGE_BACKEND','local')
    monkeypatch.delenv('AZURE_OPENAI_ENDPOINT',raising=False)
    e=sample_event(build_features(DATA).iloc[-1])
    r=process_event(e,backend='local')
    assert r['execution_status']=='NOT_EXECUTED'
    assert r['analysis']['approval_required']
    assert list((tmp_path/'predictions').glob('*.json'))

def test_risk_flags():
    f={'cluster_cpu_percent':95,'cluster_pending_tasks':6,'spark_spill_disk_bytes':123}
    flags=flags_for(f,{'stage':'checkpoint','prediction_minutes':50})
    assert 'High CPU at checkpoint' in flags and 'Spark disk spill at checkpoint' in flags
