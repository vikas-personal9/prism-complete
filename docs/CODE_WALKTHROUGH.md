# PRISM source code walkthrough, contracts, and validation

Read [Azure Portal deployment walkthrough](AZURE_PORTAL_STEP_BY_STEP.md) for resource provisioning. This file explains the Python modules and why each field exists.

## 1. Four exact uploaded schemas

**Workflow runs (50 rows)**: `workflow_id`, `workflow_name`, `run_id`, `start_time`, `end_time`, `result_state`, `cluster_id`.

**Workflow tasks (150 rows)**: `workflow_id`, `run_id`, `task_key`, `task_name`, `task_sequence`, `start_time`, `end_time`, `task_status`.

**Cluster metrics (250 rows)**: `workflow_id`, `run_id`, `cluster_id`, `metric_timestamp`, `compute_type`, `node_type`, `driver_node_type`, `worker_count`, `cpu_percent`, `memory_percent`, `active_tasks`, `pending_tasks`, `autoscaling_enabled`.

**Spark metrics (250 rows)**: `workflow_id`, `run_id`, `metric_timestamp`, `job_id`, `stage_id`, `executor_count`, `active_tasks`, `completed_tasks`, `failed_tasks`, `input_rows`, `output_rows`, `input_bytes`, `shuffle_read_bytes`, `shuffle_write_bytes`, `spill_disk_bytes`, `spill_memory_bytes`.

All 50 historical runs have result_state `SUCCESS`. Distinct workflows `WF001`–`WF005`; 10 runs each. Each run has 3 tasks and 5 cluster + 5 Spark metric samples. No source rows have missing keys.

### Data relationships

```
workflow_runs (one row per workflow_id + run_id)
  |-- workflow_tasks  (3 task rows, used for dashboard analysis)
  |-- cluster_metrics (5 checkpoint candidates, used for trained model first paired row)
  `-- spark_metrics (5 checkpoint candidates, used for trained model first paired row)
```

Use `workflow_id` **and** `run_id` to join, never only `cluster_id`, because a cluster can serve multiple workflows. Cluster and Spark metrics join on `workflow_id, run_id, metric_timestamp`, preserving the first *paired* metric measurement. This prevents Cartesian multiplication of 5 x 5 metrics per run. Task aggregates are not joined into the start or checkpoint models, because task completion measurements may arrive after prediction time.

## 2. Feature engineering: `src/features.py`

### Runtime target

```python
runtime_min = (end_time - start_time).total_seconds() / 60
```

Only training and retrospective charts use this ground truth; it is **never** exported into an incoming event. The payload intentionally omits `end_time`, `runtime_min`, `remaining_min`, `result_state`.

### Start of run predictor — exactly 8 inputs

| Actual input | Calculation | Available when? |
|---|---|---|
| `workflow_id` | from workflow history | start |
| `cluster_id` | from assigned job cluster | start |
| `start_hour` | hour of `start_time` | start |
| `start_dayofweek` | weekday (0=Monday) | start |
| `prior_runtime_min` | previous **completed** run's runtime for this workflow | start if history exists |
| `prior_3_mean_min` | rolling mean of up to three **previous** runs | start if history exists |
| `prior_3_std_min` | std of two or three previous runs | start if sufficient history |
| `prior_count` | number of prior runs for the workflow | start |

Lagged values rely on `groupby(workflow_id).shift(1)`. The very first run per workflow has `None` lags and is processed by an imputer. This is intentional; fake zeros would distort the history.

### First-checkpoint remaining runtime — exactly 28 inputs

Includes all 8 start features **plus** the following 20 observed first-checkpoint fields:

- Timing/type: `elapsed_min`, `node_type`, `compute_type`
- Cluster: `cluster_worker_count`, `cluster_cpu_percent`, `cluster_memory_percent`, `cluster_active_tasks`, `cluster_pending_tasks`, `cluster_autoscaling_enabled`
- Spark: `spark_executor_count`, `spark_active_tasks`, `spark_completed_tasks`, `spark_failed_tasks`, `spark_input_rows`, `spark_output_rows`, `spark_input_bytes`, `spark_shuffle_read_bytes`, `spark_shuffle_write_bytes`, `spark_spill_disk_bytes`, `spark_spill_memory_bytes`

The total is **28 input features** (8 + 20). `elapsed_min` comes from `checkpoint_time - start_time`. `remaining_min` target comes from `runtime_min - elapsed_min`.

### Dashboard-only retrospective aggregates

`build_task_summary()` computes task count, completed durations and count of non-success task states. `build_telemetry_summary()` computes CPU/memory averages and maximums, mean worker count, maximum pending tasks, Spark disk spill and shuffle metrics. These describe historical run outcomes; they are not predictive features for the start or first-checkpoint models.

## 3. Model training: `ml/train.py`

`make_pipeline()` performs numeric median imputation with missing indicators, categorical most-frequent imputation, and one-hot encoding. It then fits a `RandomForestRegressor` with `n_estimators=200`, `min_samples_leaf=2`, `random_state=42`. Two pipelines use the *same* feature table but different columns and targets.

Global chronological holdout (not random split): first 8 calendar days, September 1–8, train (40 rows). Latest 2 days, September 9–10, test (10 rows). Both models report MAE, RMSE and a historical rolling-average baseline MAE. Saved outputs: `artifacts/start.joblib`, `artifacts/checkpoint.joblib`, `artifacts/metrics.json`.

The small holdout and highly regular workflows make this a **proof of architecture**, not an operational guarantee. Retrain on months of real runs and test on future periods, variable cluster configurations, more workflows and failure cases before adopting alerts or sizing decisions.

## 4. Endpoint scoring: `ml/score.py`

Azure ML invokes `init()` once to discover both model files under `AZUREML_MODEL_DIR` and `run(raw_data)` for each request. `run` checks that the request stage is supported and every expected feature key is present. It returns a numeric number of minutes, or an `error` object for an invalid payload.

Input (actual, captured from `WF001-R10`) is generated by:

```bash
python -m scripts.make_request --stage checkpoint --run-id WF001-R10 --out examples/request.json
```

The output is copied into the endpoint invocation `--request-file examples/request.json`. Start predictions return `target: runtime_min`; checkpoint predictions return `target: remaining_min`. Azure ML's endpoint secret is supplied as a Bearer token in the REST Authorization header (or transparently by Azure CLI `az ml online-endpoint invoke`).

## 5. Operational messaging: `src/ingest.py` and `function_app.py`

- `src.ingest.create_event()` loads historical CSV records and publishes a **synthetic** operational event with run ID, stage, and feature values only.
- `src.ingest.publish()` sends the JSON payload to Event Hubs (`prism-metrics`).
- `IngestOperationalMetrics` is a Python v2 Azure Function Event Hubs trigger bound to consumer group `prism-functions`. It validates required event keys and writes it to Storage Queue `prism-inference`.
- `RunPrismInference` is a separate queue-trigger Azure Function. It requests a live ML endpoint prediction, calls the agent orchestrator, and writes a Blob result.

Queue-based decoupling tolerates inference spikes and transient endpoint failures better than an in-memory handoff. The consumer group is independent of `$Default`. Failed queue messages are retried; persistent failures are routed to the poison queue after the configured maximum count. Long-running agent chains could exceed a 5-minute Function timeout: lower model latency, parallelize read-only agents or adopt Durable Functions/Container Apps jobs as necessary.

## 6. Predictor: `src/predict.py`

`backend='local'`: import `ml.score` and evaluate the exact local model artifacts. `backend='azure'`: invoke `AML_SCORING_URI` using the endpoint key (`AML_ENDPOINT_KEY`) or a DefaultAzureCredential token for supported Entra endpoint authentication. Tests and UI share the same JSON contract, avoiding a separate mock predictor.

## 7. Agent orchestration: `src/agents.py`

`process_event()` in `src/pipeline.py` obtains the **actual prediction** and passes `{features, ml_prediction, risk_flags}` into `src.agents.reason()`.

**Preferred managed Foundry mode** (`PRISM_AGENT_BACKEND=foundry`): `AIProjectClient` uses `FOUNDRY_PROJECT_ENDPOINT`; `get_openai_client(agent_name=...)` executes six previously created managed prompt agents sequentially. The responses are passed as structured context to the next agent. Resources are manually created in Foundry or provisioned by `scripts/create_foundry_agents.py`, using current `azure-ai-projects>=2.3` SDK.

**Alternative Azure OpenAI mode** (`PRISM_AGENT_BACKEND=openai`): six role-specific sequential chat completions executed against a single Azure OpenAI deployment, without managed Foundry agent resources. **Offline demo mode** (`PRISM_AGENT_BACKEND=rules`): deterministic evidence flags and safe recommendations.

Roles: Monitoring -> Diagnostics -> Sizing -> Cost -> Remediation -> Communication. All variants: *recommendations only*, no actuator, no Databricks access, no implicit approval. If a Foundry call fails in explicit Foundry mode, a rules-only fallback is saved with a visible `foundry_error` field; a reviewer should treat it as an integration failure.

Risk rules in `src/pipeline.py`: CPU >85%, memory >90%, pending tasks >5, failed Spark tasks >0, disk spill >0. These are **illustrative operational heuristics**, not learned classifier probabilities or official SLAs. A start-of-run prediction >25% above recent baseline generates a separate flag. Actual alert thresholds require application-owner approval.

## 8. Persist results and reviews: `src/storage.py`

`PRISM_STORAGE_BACKEND=local` writes JSON records to `runtime_results/predictions/` and `runtime_results/approvals/`. `PRISM_STORAGE_BACKEND=azure` reads and writes a private Blob container named `prism-results`.

Each processed event produces a stable key: `predictions/<run_id>-<stage>.json`. Queue retries overwrite the same prediction key, avoiding uncontrolled duplicate run records. Each review is a new event with a UUID; it records a reviewer, decision and timestamp. **Approval is not a permission check**, and it never triggers a Databricks change. Secure the UI and use Entra identity + RBAC and an immutable audit store before production use.

## 9. Streamlit: `ui/app.py`

Seven tabs:

1. **Executive overview**: historical workflow count, success status, runtime trend, processed event results.
2. **Runtime predictions**: select historical run and stage, inspect payload, call local or live Azure ML endpoint, then optionally orchestrate and save the complete result.
3. **Cluster sizing**: correlations of CPU, worker count, memory and runtime; no unsupported optimal-size conclusions.
4. **Spark + tasks**: shuffle, spill, task activity and time statistics.
5. **AI agent reasoning**: managed agent JSON or explicit fallback recommendations.
6. **Review / audit**: record human approval/rejection, with `execution_status=NOT_EXECUTED`.
7. **Dataset + model quality**: raw source row counts, exact feature names, holdout MAE/RMSE and baseline comparisons.

The retrospective chart intentionally includes training runs and is labeled *not holdout validation*; only `artifacts/metrics.json` reports the chronological test results. Dashboard is a simulator until live event telemetry is connected.

## 10. Tests and what they prove

```bash
python -m pytest -q
python scripts/validate_data.py
python scripts/test_endpoint.py --run-id WF001-R10 --stage checkpoint
python scripts/run_local_event.py --run-id WF001-R10
```

- `test_exact_source_schemas`: joins are consistent and statuses are all success.
- `test_feature_join_and_checkpoint`: one model row per run, valid early metric checkpoint, exact required fields.
- `test_no_target_leakage`: no end time, result status or current runtime in model input.
- `test_scoring_contract_and_failures`: local model returns numeric prediction; incomplete request returns explicit error.
- `test_end_to_end_persists_only_recommendation`: reconstructed event reaches prediction, reasoning and local JSON output, but never runs infrastructure changes.
- `test_risk_flags`: illustrative diagnostic threshold rules produce expected labels.

Cloud service integration, portal-created resources, Foundry model calls and Streamlit's actual browser rendering were **not** executed in the authoring container; those require your Azure subscription/credentials and a browser. Use the Azure guide's cloud test matrix after deployment.
