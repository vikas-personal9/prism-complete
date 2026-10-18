"""Leakage-controlled features from the four supplied CSV schemas.

Start prediction uses only information available when a run starts.
Checkpoint prediction uses only the first paired timestamp of cluster and Spark metrics.
Historical aggregates are exclusively for visualization and retrospective explanations.
"""
from pathlib import Path
import numpy as np
import pandas as pd

FILES = {
    'runs': 'workflow_runs_history(1)(1).csv',
    'tasks': 'workflow_tasks_history(1)(1).csv',
    'cluster': 'cluster_metrics_history(1)(1).csv',
    'spark': 'spark_execution_metrics_history(1)(1).csv',
}
KEY = ['workflow_id', 'run_id']
START_FEATURES = ['workflow_id', 'cluster_id', 'start_hour', 'start_dayofweek',
                  'prior_runtime_min', 'prior_3_mean_min', 'prior_3_std_min', 'prior_count']
CHECKPOINT_FEATURES = START_FEATURES + [
    'elapsed_min', 'node_type', 'compute_type', 'cluster_worker_count',
    'cluster_cpu_percent', 'cluster_memory_percent', 'cluster_active_tasks',
    'cluster_pending_tasks', 'cluster_autoscaling_enabled',
    'spark_executor_count', 'spark_active_tasks', 'spark_completed_tasks',
    'spark_failed_tasks', 'spark_input_rows', 'spark_output_rows',
    'spark_input_bytes', 'spark_shuffle_read_bytes', 'spark_shuffle_write_bytes',
    'spark_spill_disk_bytes', 'spark_spill_memory_bytes',
]

def load_sources(root='data'):
    root = Path(root)
    sources = {k: pd.read_csv(root / f) for k, f in FILES.items()}
    for name, df in sources.items():
        if df[KEY].isna().any().any():
            raise ValueError(f'{name}: missing workflow_id or run_id')
    if sources['runs'].duplicated(KEY).any():
        raise ValueError('workflow_runs_history: duplicate workflow/run ID')
    for name in ('tasks', 'cluster', 'spark'):
        joined = sources[name][KEY].merge(sources['runs'][KEY], on=KEY, how='left', indicator=True)
        if joined['_merge'].eq('left_only').any():
            raise ValueError(f'{name}: orphan run IDs')
    return sources

def _first_samples(df):
    # Stable sort and one record per run; tie timestamps preserve input order.
    return df.sort_values(KEY + ['metric_timestamp'], kind='stable').drop_duplicates(KEY, keep='first')

def build_features(root='data'):
    d = load_sources(root)
    runs = d['runs'].copy()
    for col in ('start_time', 'end_time'):
        runs[col] = pd.to_datetime(runs[col], errors='raise')
    runs['runtime_min'] = (runs.end_time - runs.start_time).dt.total_seconds() / 60
    if not runs.runtime_min.gt(0).all():
        raise ValueError('Workflow runtime must be strictly positive')
    runs['start_hour'] = runs.start_time.dt.hour
    runs['start_dayofweek'] = runs.start_time.dt.dayofweek
    runs = runs.sort_values(['workflow_id', 'start_time', 'run_id']).reset_index(drop=True)
    grouped = runs.groupby('workflow_id', sort=False)['runtime_min']
    runs['prior_runtime_min'] = grouped.shift(1)
    runs['prior_3_mean_min'] = grouped.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    runs['prior_3_std_min'] = grouped.transform(lambda x: x.shift(1).rolling(3, min_periods=2).std())
    runs['prior_count'] = runs.groupby('workflow_id').cumcount()

    cluster = d['cluster'].copy()
    spark = d['spark'].copy()
    for df in (cluster, spark):
        df['metric_timestamp'] = pd.to_datetime(df['metric_timestamp'], errors='raise')
    # Paired first timestamps must match. Otherwise first common timestamp is required.
    c = cluster.sort_values(KEY + ['metric_timestamp'], kind='stable')
    s = spark.sort_values(KEY + ['metric_timestamp'], kind='stable')
    paired = c.merge(s, on=KEY + ['metric_timestamp'], validate='one_to_one', suffixes=('_cluster', '_spark'))
    if paired.groupby(KEY).ngroups != len(runs):
        raise ValueError('Every run must have at least one matching cluster and Spark metric timestamp')
    paired = _first_samples(paired)
    cfields = ['worker_count', 'cpu_percent', 'memory_percent', 'active_tasks', 'pending_tasks', 'autoscaling_enabled']
    sfields = ['executor_count', 'active_tasks', 'completed_tasks', 'failed_tasks', 'input_rows',
               'output_rows', 'input_bytes', 'shuffle_read_bytes', 'shuffle_write_bytes',
               'spill_disk_bytes', 'spill_memory_bytes']
    selected = paired[KEY + ['metric_timestamp', 'node_type', 'compute_type'] +
                      [x + '_cluster' if x == 'active_tasks' else x for x in cfields] +
                      [x + '_spark' if x == 'active_tasks' else x for x in sfields]].copy()
    rename = {'metric_timestamp': 'checkpoint_time'}
    rename.update({x + '_cluster' if x == 'active_tasks' else x: 'cluster_' + x for x in cfields})
    rename.update({x + '_spark' if x == 'active_tasks' else x: 'spark_' + x for x in sfields})
    selected = selected.rename(columns=rename)
    result = runs.merge(selected, on=KEY, how='left', validate='one_to_one')
    result['elapsed_min'] = (result.checkpoint_time - result.start_time).dt.total_seconds() / 60
    result['remaining_min'] = result.runtime_min - result.elapsed_min
    if not result.elapsed_min.ge(0).all() or not result.remaining_min.gt(0).all():
        raise ValueError('Checkpoint must occur after start and before end of workflow')
    return result

def build_task_summary(root='data'):
    d = load_sources(root)['tasks'].copy()
    d['start_time'] = pd.to_datetime(d['start_time'])
    d['end_time'] = pd.to_datetime(d['end_time'])
    d['task_runtime_min'] = (d.end_time - d.start_time).dt.total_seconds() / 60
    return d.groupby(KEY, as_index=False).agg(task_count=('task_key', 'size'),
               task_runtime_sum_min=('task_runtime_min', 'sum'),
               task_failed_count=('task_status', lambda x: int((x != 'SUCCESS').sum())))

def build_telemetry_summary(root='data'):
    d = load_sources(root)
    c = d['cluster'].groupby(KEY, as_index=False).agg(
        cpu_avg=('cpu_percent', 'mean'), cpu_max=('cpu_percent', 'max'),
        memory_avg=('memory_percent', 'mean'), memory_max=('memory_percent', 'max'),
        workers_avg=('worker_count', 'mean'), pending_max=('pending_tasks', 'max'))
    s = d['spark'].groupby(KEY, as_index=False).agg(
        spill_disk_max=('spill_disk_bytes', 'max'),
        shuffle_read_max=('shuffle_read_bytes', 'max'),
        spark_failed_max=('failed_tasks', 'max'))
    return c.merge(s, on=KEY, validate='one_to_one')

def sample_event(row, stage='checkpoint'):
    """Only model inputs go over the Event Hub, never ground-truth runtime/end_time."""
    if stage not in ('start', 'checkpoint'):
        raise ValueError(stage)
    feature_names = START_FEATURES if stage == 'start' else CHECKPOINT_FEATURES
    clean = {}
    for key in feature_names:
        value = row[key]
        if pd.isna(value):
            clean[key] = None
        elif isinstance(value, (np.integer, np.floating)):
            clean[key] = value.item()
        elif isinstance(value, (np.bool_, bool)):
            clean[key] = bool(value)
        else:
            clean[key] = value
    return {'schema_version': 1, 'stage': stage, 'workflow_id': str(row.workflow_id),
            'run_id': str(row.run_id), 'features': clean}
