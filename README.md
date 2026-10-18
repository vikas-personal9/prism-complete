# PRISM — Predictive Runtime Intelligence & Sizing Monitor

**Regenerated Azure Python reference project grounded in all four uploaded historical CSV datasets.**

## Read this first

**[Azure Portal click-by-click guide](docs/AZURE_PORTAL_STEP_BY_STEP.md)** describes 14 phases: manual Azure resource provisioning, AML training, endpoint deployment, Event Hub, Functions, managed Foundry agents, Container Apps, Monitor, expected test results, troubleshooting, and cleanup. **Nothing is already deployed in your Azure subscription.**

This implementation uses the exact columns of:

- `data/workflow_runs_history(1)(1).csv` — 50 rows
- `data/workflow_tasks_history(1)(1).csv` — 150 rows
- `data/cluster_metrics_history(1)(1).csv` — 250 rows
- `data/spark_execution_metrics_history(1)(1).csv` — 250 rows

## Application flow

```
Four uploaded historical CSVs -> data/ / Blob prism-history
                               -> Azure ML job -> registered model folder
                               -> Azure ML managed online endpoint
CSV replay (synthetic event) -> Event Hubs: prism-metrics
                               -> Azure Function IngestOperationalMetrics
                               -> Azure Storage Queue prism-inference
                               -> Azure Function RunPrismInference
                               -> Azure ML predict(features)
                               -> Managed Foundry monitoring -> diagnostics -> sizing
                                  -> cost -> remediation -> communication agents
                               -> Blob prism-results/predictions/*.json
                               -> Streamlit dashboard in Container Apps
                               -> review records in Blob (NO actual infrastructure changes)
Application Insights + Log Analytics + Azure Monitor Action Groups
```

## Local 5-minute reproduction

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt pytest
python scripts/validate_data.py
python -m pytest -q
python -m ml.train --data-dir data --out-dir artifacts
python -m scripts.test_endpoint --run-id WF001-R10 --stage checkpoint
python -m scripts.run_local_event --run-id WF001-R10
streamlit run ui/app.py
```

Open <http://localhost:8501>. In **Runtime predictions**, pick `WF001-R10` and `checkpoint`, then click **Process event, run agents, and store result**. Review AI agent output in **AI agent reasoning**, and record a review in **Review / audit**. In local mode the rules-only reasoner is used; use Foundry settings to enable six managed agents.

## Training details

- **Target 1**: `runtime_min = (workflow end_time - start_time) / 60` (total duration).
- **Start inputs**: `workflow_id`, `cluster_id`, `start_hour`, `start_dayofweek`, `prior_runtime_min`, `prior_3_mean_min`, `prior_3_std_min`, `prior_count`.
- **Target 2**: `remaining_min = runtime_min - elapsed_min`, calculated at the first common Spark/cluster metric timestamp.
- **Checkpoint inputs**: all start inputs plus elapsed time, node and compute type, cluster worker/CPU/memory/active/pending/autoscale, first Spark executor/task/row/byte/shuffle/spill metrics. See `src/features.py` for exact columns.
- **Holdout strategy**: September 1–8 train (40 runs), September 9–10 test (10 runs); all workflows appear in train/test.
- **Model**: random forest regression, numeric imputation, one-hot categorical features, fixed `random_state=42`.
- **Measured local holdout**: approximately 0.709 min MAE at start; approximately 0.396 min MAE at checkpoint. These figures are from *10 test runs* and may not hold in production. No failure classifier: all 50 labels are SUCCESS.
- Task and **full-run** telemetry aggregates are included **only for dashboard retrospective analysis**; they are not smuggled into predictive model inputs.

## Key files

| File | Function |
|---|---|
| `src/features.py` | Validate, join and engineer exact columns; event payload |
| `ml/train.py` | Chronological model training and holdout metrics |
| `ml/score.py` | Azure ML `init()` and `run()` scoring functions |
| `src/predict.py` | Local/AML online endpoint inference adapter |
| `src/ingest.py` | Publish a synthetic operational event to Event Hubs |
| `function_app.py`, `host.json` | Event Hubs consumer and queue-trigger inference worker |
| `src/agents.py` | Six Foundry agents or Azure OpenAI role prompting + rules fallback |
| `scripts/create_foundry_agents.py` | Provision six *managed* Foundry prompt agents in existing project |
| `src/pipeline.py` | Model -> risk flags -> agent output -> persisted record |
| `src/storage.py` | Local/Azure Blob results and approval store |
| `ui/app.py` | Seven-tab Streamlit UI |
| `deploy/train-job.yml` | Azure ML remote training on `cpu-prism` |
| `deploy/endpoint.yml`, `deploy/deployment.yml`, `deploy/conda.yml` | AML managed endpoint |
| `Dockerfile` | Streamlit container for Azure Container Apps |
| `tests/test_features.py` | Dataset, leakage, endpoint and local orchestration tests |

## Important boundaries

- **Architecture parity:** Event Hub -> Functions -> queue -> ML -> AI Foundry -> stored outputs -> Streamlit is wired in Python; cloud service calls require actual Azure resource creation, credentials and live deployment.
- **Managed agents:** Foundry 2.3+ managed prompt agents are supported. Set `PRISM_AGENT_BACKEND=foundry`, `FOUNDRY_PROJECT_ENDPOINT` after creating those agents. `openai` and `rules` are alternatives.
- **Real-time limitations:** Data source is four historical CSVs replayed as simulated telemetry, not live Databricks instrumentation. Databricks APIs and Azure Monitor scheduled ingestion require real tenant permissions and input systems not present in the uploads.
- **Approvals:** Review button records a decision only. No Databricks resizing, retry or notifications are executed, regardless of approved status.
- **SLA/cost:** None is invented: no user-defined SLA target and no price list are provided.
- **Security:** The starter favors rapid testing. Protect the dashboard with Microsoft Entra and move secrets to Key Vault/managed identity before exposing it to other users.
