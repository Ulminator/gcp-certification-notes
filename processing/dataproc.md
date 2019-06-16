# Dataproc

Dataproc

-	Managed Hadoop (Spark, SparkML, Hive, Pig, etc…)
-	Automated cluster management, resizing
-	Code/Query only
-	Job management screen in the console
-	Think in terms of a ‘job specific resource’ – for each job, create a cluster and then delete it.
-	Used if migrating existing on-premise Hadoop or Spark infrastructure to GCP without redevelopment effort.
-	Can sale even when jobs are running.
-	Use Dataflow for streaming instead. This is better for batch.
-	Connecting to Web Interface of Dataproc Cluster
o	Allow necessary web ports access via firewall rules, and limit access to your network.
	Tcp:8088 (Cluster Manager)
•	<Master Node IP>:8088
	Tcp:50070 (Connect to HDFS name node)
•	<Master Node IP>:50070
o	OR SOCKS proxy (routes through an SSH tunnel for secure access)
	gcloud compute ssh [master_node_name]
-	Pricing
o	Standard Compute Engine machine type pricing + managed Dataproc premium.
o	Premium = $0.01 per vCPU core/hour
o	Billed by the second, with a minimum of 1 minute.
-	IAM
o	Project level only (primitive and predefined roles)
o	Cloud Dataproc Editor, Viewer, and Worker
	Editor – Full access to create/edit/delete clusters/jobs/workflows
	Viewer – View access only
	Worker – Assigned to service accounts
•	Read/write GCS, write to Cloud Logging
-	Storage
o	Can use on disk (HDFS) or GCS
o	HDFS
	Split up on the cluster, but requires cluster to be up.
o	GCS
	Allows for the use of preemptible machines that can reduce costs significantly.
•	DO not need to configure startup and shutdown scripts to gracefully handle shutdown, Dataproc already handles this.
•	Cluster MUST have at least 2 standard worker nodes however.
	Separate cluster and storage.
	Cloud Storage Connector
•	Allows you to run Hadoop or Spark jobs directly on GCS.
•	Quick startup
o	In HDFS, a MapReduce job can’t start until the NameNode is out of safe mode.
o	With GCS, can start job as soon as the task nodes start, leading to significant cost savings over time.
-	Cluster Machine Types
o	Build using Compute Engine VM instances
o	Cluster – need at least 1 master and 2 workers
-	High Availability Mode
o	3 masters rather than 1
o	3 masters run in an Apache Zookeeper cluster for automatic failover.
-	Restartable Jobs
o	Jobs do NOT restart on failure (default)
o	Can change this – useful for long running and streaming jobs (ex. Spark Streaming)
o	Mitigates out-of-memory errors, unscheduled reboots
-	Updating Clusters
o	Can only change # workers/preemptible VM’s/labels/toggle graceful decommission.
	Graceful Decommissioning
•	Finish work in progress on a worker before it is removed from Dataproc cluster.
•	Incorporates graceful YARN decommissioning.
•	May fail for preemptible workers.
o	Can forcefully decommission preemptible workers at any time.
•	Will always work with primary workers.
•	gcloud dataproc clusters update –gracefult-decommission-timeout
o	Default to “0s” – forceful decommissioning.
o	Need to provide a time.
o	Max value 1 day.
o	Automatically reshards data for you.
-	Migrating and Optimizing for GCP Best Practices
o	Move data first
	Generally to GCS buckets.
	Possible exceptions
•	Apache HBase data to BigTable
•	Apache Impala to BigQuery
•	Can still choose to move to GCS if BigTable/BQ feature not needed.
o	Small scale experimentation
	Use a subset of data to test.
o	Think of it in terms of ephemeral clusters.
o	Use GCP tools to optimize and save costs.
-	Converting from HDFS to GCS
o	Copy data to GCS
	Install connector or copy manually
o	Update file prefix in scripts
	Hdfs:// to gs://
o	Use Dataproc, and run against/output to GCS
-	Connectors
o	BQ/BigTable (copies data to GCS) /CloudStorage
-	Optional Components
o	Anaconda, Druid, Hive WebHCat, Jupyter, Kerberos, Presto, Zeppelin, Zookeeper
-	How to configure Hadoop to use all cores?
o	Think spark executor cores
-	How to handle out of memory errors?
o	Hint - Executor memory
-	How to install other components?
o	Hint – Initialization actions
