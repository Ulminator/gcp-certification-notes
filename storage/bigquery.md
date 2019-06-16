# BigQuery

-	Hive equivalent
-	No ACID properties
-	Great for analytics/business intelligence/data warehouse (OLAP)
-	Fully managed data warehouse
-	Has connectors to BigTable, GCS, Google Drive, and can import from Datastore backups, CSV, JSON, and AVRO.
-	Performance
o	Petabyte scale
o	High latency
	Worse than BigTable and DataStore
-	Architecture
o	Jobs (queries) can scale up to thousands of CPU’s across many nodes, but the process is completely invisible to end user.
o	Storage and compute are separated, connected by petabit network.
o	Columnar data store
	Separates records into column values, stores each value on different storage volume.
	Poor writes (BQ does not update existing records)
-	IAM
o	Security can be applied at project and dataset level, but not at table or view level.
o	Predefined roles BQ
	Admin – Full access
	Data owner – Full dataset access
	Data editor – edit dataset tables
	Data viewer – view datasets and tables
	Job User – run jobs
	User – run queries and create datasets (but not tables)
	metaDataViewer
	readSessionUser – Create and use read sessions within project.
o	Authorized views allow you to share query results with particular users/groups without giving them access to underlying data.
	Restrict access to particular columns or rows
	Create a separate dataset to store the view.
	How:
•	Grant IAM role for data analysts (bigquery.user)
o	They won’t have access to query data, view table data, or view table schema details for datasets they did not create.
•	(In source dataset) Share the dataset, In permissions go to Authorized views tab.
o	View gets access to source data, not analyst group.
-	Pricing
o	Based on:
	storage (amount of data stored)
	querying (amount of data/number of bytes processed by query)
	streaming inserts.
o	Storage options are active and long term
	Modified or not past 90 days
o	Query options are on-demand and flat-rate
-	Data Model
o	Dataset = set of tables and views
o	Table must belong to dataset
o	Dataset must belong to a project
o	Tables contain records with rows and columns (fields)
	Nested and repeatable fields are OK
-	Table Schema
o	Can be specified at creation time
o	Can also specify schema during initial load
o	Can update schema later too
-	Query
o	Standard SQL (preferred) or Legacy SQL (old)
	Standard
•	Table names can be referenced with backticks
o	Needed for wildcards
o	Cannot use both Legacy and SQL2011 in same query.
o	Table partitioning
o	Distributed writing to file for output (i.e. file-0001-of-0002)
o	User defined functions in JS (UDFJS)
	Temporary – Can only use for current query or command line session.
o	Query jobs are actions executed asynchronously to load, export, query, or copy data.
o	If you use the LIMIT clause, BigQuery will still process the entire table.
o	Avoid SELECT * (full scan), select only columns needed (SELECT * EXCEPT)
o	Denormalized Data Benefits
	Increases query speed
	Makes queries simpler
	BUT: Normalization makes dataset better organized, but less performance optimized.
o	Types
	Interactive (default)
•	Query executed immediately
•	Counts towards 
o	Daily usage
o	Concurrent usage
	Batch
•	Scheduled to run whenever possible (idle resources)
•	Don’t count towards limit on concurrent usage.
•	If not started within 24hr, BQ makes them interactive.
-	Data Import
o	Data is converted into columnar format for Capacitor.
o	Batch (free)
	web console (local files), GCS, GDS, Datastore backups (particularly logs)
	Other Google services (i.e. Google Ad Manager, Google Ads)
o	Streaming (costly)
	Data with CDF, Cloud Logging, or POST calls
	High volume event tracking logs
	Realtime dashboards
	Can stream data to datasets in both the US and EU
	Streaming into ingestion-time partitioned tables:
•	Use tabledata.insertAll requests
•	Destination partition is inferred from current date based on UTC time.
o	Can override destination partition using a decorator like so: `mydataset.table$20170301`
•	Newly arriving data will be associated with the UNPARTITIONED partition while in the streaming buffer.
•	A query can therefore exclude data in the streaming buffer from a query by filtering out the NULL values from the UNPARTITIONED partition by using one of the pseudo-columns ([_PARTITIONTIME]) or [_PARTITIONDATE] depending on preferred data type.
	Streaming to a partitioned table:
•	Can stream data into a table partitioned on a DATE or TIMESTAMP column that is between 1 year in the past and 6 months in the future.
•	Data between 7 days prior and 3 days in the future is placed in the streaming buffer, and then extracted to corresponding partitions.
•	Data outside that window (but within 1 year, 6 month range) is extracted to the UNPARTITIONED partition and loaded to corresponding partitions when there’s enough data.
	Creating tables automatically using template tables
•	Common usage pattern for streaming is to split a logical table into many smaller tables to create smaller sets of data.
•	To create smaller tables by date -> partitioned tables
•	To create smaller tables that are not date based -> template tables
o	BQ will create the tables for you.
o	Add templateSuffix parameter to your insertAll request.
o	<target_table_name> + <templateSuffix>
o	Only need to update template table schema then all subsequently generated tables will use the updated schema.
	Quotas
•	Max row size: 1MB
•	HTTP request size limit: 10MB
•	Max rows per second: 100,000 rows/s for all tables combined.
•	Max bytes per second: 100MB/s per table
o	Raw Files
	Federated data source, CSV/JSON/Avro on GCS, Google sheets
o	Google Drive
	Loading is not currently supported.
	Can query data in Drive using an external table.
o	Expects all source data to be UTF-8 encoded.
o	To support (occasionally) schema changing you can use automatically detect (not default setting).
	Available while:
•	Loading data
•	Querying external data
o	Web UI
	Upload a file greater than 10MB in size
	Upload multiple files at the same time
	Upload a file in SQL format
	Cannot load multiple files at once.
•	Can with CLI though.
-	Loading Compressed and Uncompressed Data
o	Avro preferred for loading compressed data.
	Faster to load since it can be read in parallel, even when data blocks are compressed.
o	Parquet Binary format also a good choice
	Efficient per-column encoding typically results in better compression ratio and smaller files.
o	ORC Binary format offers benefits similar to Parquet
	Fast to load because data stripes can be read in parallel.
	Rows in each stripe are loaded sequentially.
	To optimize load time: data stripe size of 256MB or less.
o	CSV and JSON
	BQ load uncompressed files significantly faster than compressed.
	Uncompressed can be read in parallel.
	Uncompressed are larger => bandwidth limitations and higher GCS costs for data staged prior to being loaded into BQ.
	Line ordering not guaranteed for compressed or uncompressed.
o	If bandwidth limited, compress with GCIP before uploading to GCS.
o	If speed is important and you have a lot of bandwidth, leave uncompressed.
-	Loading Denormalized, Nested, and Repeated Data
o	BQ performs best with denormalized data.
o	Increases in storage costs worth the performance gains of denormalized data.
o	Joins require data coordination (communication bandwidth)
	Denormalization localizes the data to individual slots so execution can be done in parallel.
o	If need to maintain data while denormalizing data
	Use nested and repeated fields instead of completely flattening data.
	When completely flattened, network communication (shuffling) can negatively impact query performance.
o	Avoid denormalization when:
	Have a star schema with frequently changing dimensions.
	BQ complements an OLTP system with row-level mutation, but can’t replace it.
-	BigQuery Transfer Service
o	Automates loading data into BQ from Google Services:
	Campaign Manager
	Cloud Storage
	Amazon S3
	Google Ad Manager
	Google Ads
	Google Play
	YouTube – Channel Reports
	YouTube – Content Owner Reports
-	Partitions
o	Improves query performance => reduces costs
o	Cannot change an existing table into a partitioned table.
o	Types
	Ingestion Time
•	Partition based on data’s ingestion date or arrived date.
•	Pseudo column `_PARTITIONTIME`
o	Reserved by BQ and can’t be used.
•	Need to update schema of table before loading data if loading into a partition with a different schema.
	Partitioned Tables
•	Tables that are partitioned based on a `TIMESTAMP` or `DATE` column.
•	2 special partitions are created
o	__NULL__ paritition
	Represents rows with NULL values in the partitioning column
o	__UNPARTITIONED__ partition
	Represents data that exists outside the allowed range of dates
•	All data in partitioning column matches the date of the partition identifier with the exception of those 2 special partitions.
o	Allows query to determine which partitions contain no data that satisfies the filter conditions.
o	Queries that filter data on the partitioning column can restrict values and completely prune unnecessary partitions.
o	Wildcard tables
	Used if you want to union all similar tables with similar names. (i.e. project.dataset.Table*)
	Filter in WHERE clause
•	AND _TABLE_SUFFIX BETWEEN ‘table003’ and ‘table050’
-	Windowing
o	Window functions increase the efficiency and reduce the complexity of queries that analyze partitions (windows) of a dataset by providing complex operations without the need for many intermediate calculations.
o	Reduce the need for intermediate tables to store temporary data.
-	Bucketing
o	Like partitioning, but each split/partition should be the same size and is based on the hash function of a column.
o	Each bucket is a separate file, which makes for more efficient sampling and joining data.
-	Legacy vs. Standard SQL
o	Standard: ‘project.dataset.tablename*’
o	Legacy: [project.dataset.tablename]
o	It is set each time you run a query
o	Default query language is
	Legacy SQL for classic UI
	Standard SQL for Beta UI
-	Anti-Patterns
o	Avoid self joins
o	Partition/Skew
	Avoid unequally sized partitions
	Values occurring more often than other values..
o	Cross-Join
	Joins that generate more outputs than inputs
o	Update/Insert Single Row/Column
	Avoid a specific DML, instead batch updates/inserts
o	Anti-Patterns: https://cloud.google.com/bigtable/docs/schema-design
-	Table Types
o	Native Tables
	Backed by native BQ storage
o	External Tables
	Backed by storage external to BQ (federated data source)
	BigTable, Cloud Storage, Google Drive
o	Views
	Virtual tables defined by SQL query.
	Logical – not materialized
	Underlying query will execute each time the view is accessed.
	Benefits:
•	Reduce query complexity
•	Restrict access to data
•	Construct different logical tables from same physical table
	Cons:
•	Can’t export data from a view
•	Can’t use JSON API to retrieve data
•	Can’t mix standard and legacy SQL
o	E.g. standard sql cannot access legacy sql view
•	No user-defined functions allowed
•	No wildcard table references
o	Due to partitioning
•	Limit of 1000 authorized views per dataset
-	Caching
o	No charge for a query that retrieves results from cache.
o	Results are cached for 24 hours.
o	Caching is per user only.
o	bq query –nouse_cache ‘<QUERY>’
o	Cached by Default unless
	A destination table is specified.
	If any referenced tables or logical units have changed since results previously cached.
	If any referenced tables have recently received streaming inserts even if no new rows have arrived.
	If the query uses non-deterministic functions such as CURRENT_TIMESTAMP(), NOW(), CURRENT_USER()
	Querying multiple tables using a wildcard
	If the query runs against an external data source.
-	Export
o	Destination has to be GCS.
	Can copy table to another BigQuery dataset though.
o	Can be exported as JSON/CSV/Avro
	Default is CSV
o	Only compression option: GZIP
	Not supported for Avro
o	To export > 1 GB
	Need to put a wildcard in destination filename
	Up to 1 GB of table data in a single file
o	bq extract ‘project:dataset.table’ gs://bucket
-	Query Plan Explanation
o	In web UI, click on “Explanation”
o	Good for debugging complex queries not running as fast as needed/expected.
o	Monitoring Query Performance (UI)
	`Details` button after running query.
	Colors
•	Yellow – Wait
•	Purple – Read
•	Orange – Compute
•	Blue – Write
	Less parallel inputs => better performance => best cost 
-	Slots
o	Unit of computational capacity needed to run queries.
o	BQ calculates on basis of query size, complexity
o	Usually default slots are sufficient
o	Might need to be expanded over time, complex queries
o	Subject to quota policies ($$)
o	Can use StackDriver Monitoring to track slot usage.
-	Clustered Tables
o	Order of columns determines sort order of data.
o	Think of Clustering Columns in Cassandra
o	When to use:
	Data is already partitioned on date or timestamp column.
	You commonly use filters or aggregation against particular columns in your queries.
o	Does not work if the clustered column is used in a complex filter (used in a function in the filter expression)
-	BigQuery ML
o	Create and execute machine learning models in BQ using standard SQL
o	Supported models
	Linear regression
	Binary Logistic regression
	Multiclass logistic regression for classification
	K-means clustering for data segmentation (beta)
	Import TensorFlow Models (alpha)
o	TensorFlow DNN
	Classifier
	Regressor
o	Benefits from not having to export and re-format data
	Brings machine learning to the data.
-	Best Practices
o	Costs
	Avoid SELECT *
•	Query only columns you need.
	Sample data using preview options
•	Don’t run queries to explore or preview table data.
	Price your queries before running them.
•	Before running queries, preview them to estimate costs.
	Limit query costs by restricting the number of bytes billed.
•	Use the maximum bytes billed setting to limit query costs.
	LIMIT doesn’t affect cost
•	Do not use LIMIT clause as a method of cost control as it does not affect the amount of data that is read.
	View costs using a dashboard and query your audit logs
•	Create a dashboard to view your billing data so you can make adjustments to your BigQuery usage. Also consider streaming audit logs to BigQuery to analyze usage patterns.
	Partition data by date
	Materialize query results in stages
•	Break large query into stages where each stage materializes the results by writing to a destination table.
•	Querying smaller destination table reduces amount of data that is read and lowers costs.
	Consider cost of large result sets
•	Use default table expiration time to remove data when not needed.
•	Good for when writing large query results to a destination table.
	Use streaming inserts with caution
•	Only use if data is needed immediately available.
o	Query Performance
	Input data and data sources (I/O)
•	Control projection – Avoid SELECT *
•	Prune partitioned queries
o	Use partition columns to filter
•	Denormalize data when possible
o	JSON, Parquet, or Avro
o	When creating, specify Type in the Schema as RECORD
•	Use external data sources appropriately
o	If performance is a top priority, do not use external source
•	Avoid excessive wildcard tables
o	Use most granular prefix possible
	Communication between nodes (shuffling)
•	Reduce data before using a JOIN
•	Do not treat WITH clauses as prepared statements
•	Avoid tables sharded by date
o	Use time-based partitioned tables instead
	Copy of schema and metadata is maintained for each sharded table.
	BQ might have to verify permissions for each queries table. (overhead)
•	Avoid oversharding tables
	Computation
•	Avoid repeatedly transforming data via SQL queries
•	Avoid JavaScript user-defined functions.
o	Use native UDFs instead.
•	Use approximate aggregation functions
o	COUNT(DISTINCT) vs. APPROX_COUNT_DISTINCT()
•	Order query operations to maximize performance
o	Only use in the outermost query or within window clauses.
o	Push complex operations to the end of the query.
•	Optimize join patterns
o	Start with the largest table
•	Prune partitioned queries
	Outputs (materialization)
•	Avoid repeated joins and subqueries
•	Carefully consider materializing large result sets
•	Use LIMIT clause with large sorts
	Anti-patterns
•	Self-joins
o	Potentially doubles number of output rows
o	Use window function instead
•	Data skew
o	If query processes keys that are heavily skewed to a few values, filter your data as early as possible.
•	Cross joins (Cartesian product)
o	Avoid joins that generate more outputs than inputs.
o	Pre-aggregate data first if it is required.
•	DML statements that update or insert single rows
o	Use batch.
o	Storage Optimization
	Use expiration settings to remove unneeded tables and partitions
•	Configure default table expiration for datasets
•	Configure expiration time for tables
•	Configure partition expiration for partitioned tables
	Take advantage of long term storage
•	Untouched tables (90 days) are as cheap as GCS Nearline
•	Each partition is considered separately.
	Use pricing calculator to estimate storage costs

BigQuery Components

Dremel - Execution Engine
	- Turns SQL query into an execution tree.
	- Leaves on the tree are 'slots'
- Does the heavy lifting of reading data from Colossus and doing any computation necessary.
	- Branches of tree are 'mixers'
		- Perform aggregation
	- In between is 'shuffle'
- Uses Google's Jupiter network to move data extremely rapidly from one place to another.
	- Mixers and Slots all run by Borg
		- Doles out hardware resources.

- Dynamically appropriates slots to queries on an as needed basis, maintaining fairness amongst multiple users who are all querying at once.
	- Widely used at Google
		- Search
		- Ads
		- YouTube
		- BigQuery

Colossus - Distributed Storage
	- Google's latest generation distributed file system.
	- Colossus cluster in each Google datacenter.
- Each cluster has enough disks to give every BigQuery user thousands of dedicated disks at a time.
- Handles replication, recovery (when disk crash), and distributed management (so no single point of failure).
	- Leverages columnar storage

Borg - Compute
	- Gives thousands of CPU cores dedicated to processing your task.
	- Large-scale cluster management system.
	- Run on dozens of thousands of machines and hundreds of thousands of cores.
	- Assigns resources to jobs - Dremel cluster in this scenario.

Jupiter - The Network
	- Can deliver 1 Petabit/sec of total bisection bandwidth.
- Enough to allow 100,000 machines to communicate with any other machines at 10Gbs.
	- Full duplex bandwidth means that locality within cluster is not important.
		- Each macine can talk at 10Gbs regardless of rack.
