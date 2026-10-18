"""PRISM orchestration: telemetry -> ML -> diagnostics -> records."""
from datetime import datetime, timezone
from src.agents import reason
from src.predict import predict
from src.storage import write_json

def flags_for(features, prediction):
    flags = []
    rules = [
        ('cluster_cpu_percent', 85, 'High CPU at checkpoint'),
        ('cluster_memory_percent', 90, 'High memory at checkpoint'),
        ('cluster_pending_tasks', 5, 'Pending task backlog at checkpoint'),
        ('spark_failed_tasks', 0, 'Spark task failures at checkpoint'),
        ('spark_spill_disk_bytes', 0, 'Spark disk spill at checkpoint'),
    ]
    for col, threshold, label in rules:
        val = features.get(col)
        if val is not None and float(val) > threshold:
            flags.append(label)
    previous = features.get('prior_3_mean_min')
    if prediction.get('stage') == 'start' and previous not in (None, 0):
        if float(prediction['prediction_minutes']) > 1.25 * float(previous):
            flags.append('Predicted runtime exceeds previous 3-run mean by more than 25%')
    return flags

def process_event(event, backend=None, persist=True):
    required = {'schema_version', 'run_id', 'workflow_id', 'stage', 'features'}
    if not required.issubset(event):
        raise ValueError(f'Event missing: {sorted(required - set(event))}')
    if event['schema_version'] != 1:
        raise ValueError('Unsupported schema_version')
    stage = event['stage']
    features = event['features']
    result = predict(features, stage, backend=backend)
    if 'error' in result:
        raise ValueError(result['error'])
    flags = flags_for(features, result)
    analysis = reason(features, result, flags)
    record = {'workflow_id': event['workflow_id'], 'run_id': event['run_id'],
              'stage': stage, 'processed_at_utc': datetime.now(timezone.utc).isoformat(),
              'prediction': result, 'flags': flags, 'analysis': analysis,
              'review_status': 'PENDING', 'execution_status': 'NOT_EXECUTED'}
    if persist:
        write_json('predictions', f"{event['run_id']}-{stage}", record)
    return record
