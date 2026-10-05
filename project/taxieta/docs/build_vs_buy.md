# Build vs buy (Milestone 9)

Decide using the book's three factors: **company stage**, **focus / competitive advantage**, **tool maturity**.

| Component | What we used | Managed alternative | 20-person startup | Bank |
| --- | --- | --- | --- | --- |
| Compute | Laptop / Docker | EC2, GCE | | |
| Streaming | Redpanda | Confluent Cloud, Kinesis | | |
| Orchestration | Airflow 3 (standalone) | MWAA, Cloud Composer, Astronomer | | |
| Feature store | Feast (local) | Tecton, Databricks, SageMaker FS | | |
| Model registry | File store / MLflow | SageMaker, Vertex AI, Databricks | | |
| Serving | FastAPI + Docker | SageMaker, Vertex AI, Seldon | | |
| Monitoring | Custom KS/PSI | Evidently Cloud, Arize, … | | |

Reasoning:
