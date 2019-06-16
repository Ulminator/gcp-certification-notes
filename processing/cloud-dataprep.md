# Cloud Dataprep

- Intelligent data preparation
- Partnered with Trifecta for data cleaning/processing service
- Fully managed, serverless, and web based
- User friendly interface
    - Clean data by clicking on it
- Supported file types
    - Inputs
        - CSV, JSON, Plain Text, Excel, LOG, TSV, and Avro
    - Outputs
        - CSV, JSON, Avro, BQ Table
            - CSV/JSON can be compressed or uncompressed

## How it works
- Backed by Cloud Dataflow
    - After preparing Dataflow processes via Apache Beam pipeline
    - “User-friendly Dataflow pipeline”
- Dataprep Process
    - Import data
    - Transform sampled data with recipes
    - Run Dataflow job on transformed dataset
        - Batch Job
        - Every recipe step is its own transform
    - Export results (GCS, BigQuery)
- Intelligent Suggestions
    - Selecting data will often automatically give the best suggestion
    - Can manually create recipes, however simple tasks (remove outliers, de-duplicate) should just use auto-suggestions.

## IAM
- Dataprep.projects.user
    - Run Dataprep in a project
- Dataprep.serviceAgent
    - Gives Trifecta necessary access to project resources.
        - Access GCS buckets, Dataflow Developer, BQ user/data editor
        - Necessary for cross-project access + GCE service account

## Cost
- 1.16 * cost of a Dataflow job

## Flows
- Add or import datasets to process with recipes
- Public Bucket for testing: gs://dataprep-samples
- For large datasets:
    - UI only shows a sample to work with
    - Recipe created is then applied to entirety of dataset

## Jobs
- Create a dataset in BQ first
- Click on Run Job
    - Default option is CSV in GCS bucket
    - Choose BQ dataset instead
    - Name table
    - Run Job: Create Apache Beam pipeline with Dataflow
