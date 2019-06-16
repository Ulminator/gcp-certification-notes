# Cloud Dataprep

-	Intelligent data preparation
-	Partnered with Trifecta for data cleaning/processing service
-	Fully managed, serverless, and web based
-	User friendly interface
o	Clean data by clicking on it
-	Supported file types
o	Inputs
	CSV, JSON, Plain Text, Excel, LOG, TSV, and Avro
o	Outputs
	CSV, JSON, Avro, BQ Table
•	CSV/JSON can be compressed or uncompressed
-	How it works
o	Backed by Cloud Dataflow
	After preparing Dataflow processes via Apache Beam pipeline
	“User-friendly Dataflow pipeline”
o	Dataprep Process
	Import data
	Transform sampled data with recipes
	Run Dataflow job on transformed dataset
•	Batch Job
•	Every recipe step is its own transform
	Export results (GCS, BigQuery)
-	Intelligent Suggestions
o	Selecting data will often automatically give the best suggestion
o	Can manually create recipes, however simple tasks (remove outliers, de-duplicate) should just use auto-suggestions.
-	IAM
o	Dataprep.projects.user
	Run Dataprep in a project
o	Dataprep.serviceAgent
	Gives Trifecta necessary access to project resources.
•	Access GCS buckets, Dataflow Developer, BQ user/data editor
•	Necessary for cross-project access + GCE service account
-	Pricing
o	1.16 * cost of a Dataflow job
-	Flows
o	Add or import datasets to process with recipes
o	Public Bucket for testing: gs://dataprep-samples
o	For large datasets:
	UI only shows a sample to work with
	Recipe created is then applied to entirety of dataset
-	Jobs
o	Create a dataset in BQ first
o	Click on Run Job
	Default option is CSV in GCS bucket
	Choose BQ dataset instead
	Name table
	Run Job: Create Apache Beam pipeline with Dataflow
