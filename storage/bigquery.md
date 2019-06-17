# BigQuery

- Hive equivalent
- No ACID properties
- Great for analytics/business intelligence/data warehouse (OLAP)
- Fully managed data warehouse
- Has connectors to BigTable, GCS, Google Drive, and can import from Datastore backups, CSV, JSON, and AVRO.
- Performance
    - Petabyte scale
    - High latency
        - Worse than BigTable and DataStore

## Data Model
- Dataset = set of tables and views
- Table must belong to dataset
- Dataset must belong to a project
- Tables contain records with rows and columns (fields)
    - Nested and repeatable fields are OK

## Table Schema
- Can be specified at creation time
- Can also specify schema during initial load
- Can update schema later too

## Query
- Standard SQL (preferred) or Legacy SQL (old)
    - Standard
        - Table names can be referenced with backticks
            - Needed for wildcards
    - Cannot use both Legacy and SQL2011 in same query.
    - Table partitioning
    - Distributed writing to file for output (i.e. file-0001-of-0002)
    - User defined functions in JS (UDFJS)
        - Temporary – Can only use for current query or command line session.
    - Query jobs are actions executed asynchronously to load, export, query, or copy data.
    - If you use the LIMIT clause, BigQuery will still process the entire table.
    - Avoid SELECT * (full scan), select only columns needed (SELECT * EXCEPT)
    - Denormalized Data Benefits
        - Increases query speed
        - Makes queries simpler
        - BUT: Normalization makes dataset better organized, but less performance optimized.
    - Types
        - Interactive (default)
            - Query executed immediately
            - Counts towards 
                - Daily usage
                - Concurrent usage
        - Batch
            - Scheduled to run whenever possible (idle resources)
            - Don’t count towards limit on concurrent usage.
            - If not started within 24hr, BQ makes them interactive.

## Data Import
- Data is converted into columnar format for Capacitor.
- Batch (free)
    - web console (local files), GCS, GDS, Datastore backups (particularly logs)
    - Other Google services (i.e. Google Ad Manager, Google Ads)
- Streaming (costly)
    - Data with CDF, Cloud Logging, or POST calls
    - High volume event tracking logs
    - Realtime dashboards
    - Can stream data to datasets in both the US and EU
    - Streaming into ingestion-time partitioned tables:
        - Use tabledata.insertAll requests
        - Destination partition is inferred from current date based on UTC time.
            - Can override destination partition using a decorator like so: `mydataset.table$20170301`
        - Newly arriving data will be associated with the UNPARTITIONED partition while in the streaming buffer.
        - A query can therefore exclude data in the streaming buffer from a query by filtering out the NULL values from the UNPARTITIONED partition by using one of the pseudo-columns ([_PARTITIONTIME]) or [_PARTITIONDATE] depending on preferred data type.
    - Streaming to a partitioned table:
        - Can stream data into a table partitioned on a DATE or TIMESTAMP column that is between 1 year in the past and 6 months in the future.
        - Data between 7 days prior and 3 days in the future is placed in the streaming buffer, and then extracted to corresponding partitions.
        - Data outside that window (but within 1 year, 6 month range) is extracted to the UNPARTITIONED partition and loaded to corresponding partitions when there’s enough data.
    - Creating tables automatically using template tables
        - Common usage pattern for streaming is to split a logical table into many smaller tables to create smaller sets of data.
        - To create smaller tables by date -> partitioned tables
        - To create smaller tables that are not date based -> template tables
            - BQ will create the tables for you.
            - Add templateSuffix parameter to your insertAll request.
            - <target_table_name> + <templateSuffix>
            - Only need to update template table schema then all subsequently generated tables will use the updated schema.
    - Quotas
        - Max row size: 1MB
        - HTTP request size limit: 10MB
        - Max rows per second: 100,000 rows/s for all tables combined.
        - Max bytes per second: 100MB/s per table
- Raw Files
    - Federated data source, CSV/JSON/Avro on GCS, Google sheets
- Google Drive
    - Loading is not currently supported.
    - Can query data in Drive using an external table.
- Expects all source data to be UTF-8 encoded.
- To support (occasionally) schema changing you can use automatically detect (not default setting).
    - Available while:
        - Loading data
        - Querying external data
- Web UI
    - Upload a file greater than 10MB in size
    - Upload multiple files at the same time
    - Upload a file in SQL format
    - Cannot load multiple files at once.
        - Can with CLI though.
## Loading Compressed and Uncompressed Data
- Avro preferred for loading compressed data.
    - Faster to load since it can be read in parallel, even when data blocks are compressed.
- Parquet Binary format also a good choice
    - Efficient per-column encoding typically results in better compression ratio and smaller files.
- ORC Binary format offers benefits similar to Parquet
    - Fast to load because data stripes can be read in parallel.
    - Rows in each stripe are loaded sequentially.
    - To optimize load time: data stripe size of 256MB or less.
- CSV and JSON
    - BQ load uncompressed files significantly faster than compressed.
    - Uncompressed can be read in parallel.
    - Uncompressed are larger => bandwidth limitations and higher GCS costs for data staged prior to being loaded into BQ.
    - Line ordering not guaranteed for compressed or uncompressed.
- If bandwidth limited, compress with GCIP before uploading to GCS.
- If speed is important and you have a lot of bandwidth, leave uncompressed.

## Loading Denormalized, Nested, and Repeated Data
- BQ performs best with denormalized data.
- Increases in storage costs worth the performance gains of denormalized data.
- Joins require data coordination (communication bandwidth)
    - Denormalization localizes the data to individual slots so execution can be done in parallel.
- If need to maintain data while denormalizing data
    - Use nested and repeated fields instead of completely flattening data.
    - When completely flattened, network communication (shuffling) can negatively impact query performance.
- Avoid denormalization when:
    - Have a star schema with frequently changing dimensions.
    - BQ complements an OLTP system with row-level mutation, but can’t replace it.

## BigQuery Transfer Service
- Automates loading data into BQ from Google Services:
    - Campaign Manager
    - Cloud Storage
    - Amazon S3
    - Google Ad Manager
    - Google Ads
    - Google Play
    - YouTube – Channel Reports
    - YouTube – Content Owner Reports

## Partitions
- Improves query performance => reduces costs
- **Cannot change an existing table into a partitioned table.**
- **Types**
    - **Ingestion Time**
        - Partition based on data’s ingestion date or arrived date.
        - Pseudo column `_PARTITIONTIME`
        - Reserved by BQ and can’t be used.
        - Need to update schema of table before loading data if loading into a partition with a different schema.
    - **Partitioned Tables**
        - Tables that are partitioned based on a `TIMESTAMP` or `DATE` column.
        - 2 special partitions are created
            - __NULL__ paritition
                - Represents rows with NULL values in the partitioning column
            - __UNPARTITIONED__ partition
                - Represents data that exists outside the allowed range of dates
        - All data in partitioning column matches the date of the partition identifier with the exception of those 2 special partitions.
            - Allows query to determine which partitions contain no data that satisfies the filter conditions.
            - Queries that filter data on the partitioning column can restrict values and completely prune unnecessary partitions.
- **Wildcard tables**
    - Used if you want to union all similar tables with similar names. (i.e. project.dataset.Table*)
    - Filter in WHERE clause
        - AND _TABLE_SUFFIX BETWEEN ‘table003’ and ‘table050’

## Windowing
- Window functions increase the efficiency and reduce the complexity of queries that analyze partitions (windows) of a dataset by providing complex operations without the need for many intermediate calculations.
- Reduce the need for intermediate tables to store temporary data.

## Bucketing
- Like partitioning, but each split/partition should be the same size and is based on the hash function of a column.
- Each bucket is a separate file, which makes for more efficient sampling and joining data.

## Legacy vs. Standard SQL
- Standard: ‘project.dataset.tablename*’
- Legacy: [project.dataset.tablename]
- It is **set each time you run a query**
- Default query language is
- Legacy SQL for classic UI
- Standard SQL for Beta UI

## Anti-Patterns
- Avoid self joins
- Partition/Skew
    - Avoid unequally sized partitions
    - Values occurring more often than other values..
- Cross-Join
    - Joins that generate more outputs than inputs
- Update/Insert Single Row/Column
    - Avoid a specific DML, instead batch updates/inserts
- Anti-Patterns: https://cloud.google.com/bigtable/docs/schema-design

## Table Types
- **Native Tables**
    - Backed by native BQ storage
- **External Tables**
    - Backed by storage external to BQ (federated data source)
    - BigTable, Cloud Storage, Google Drive
- **Views**
    - Virtual tables defined by SQL query.
    - Logical – not materialized
    - Underlying query will execute each time the view is accessed.
    - Benefits:
        - Reduce query complexity
        - Restrict access to data
        - Construct different logical tables from same physical table
    - Cons:
        - Can’t export data from a view
        - Can’t use JSON API to retrieve data
        - Can’t mix standard and legacy SQL
            - e.g. standard sql cannot access legacy sql view
        - No user-defined functions allowed
        - No wildcard table references
            - Due to partitioning
        - Limit of 1000 authorized views per dataset

## Caching
- No charge for a query that retrieves results from cache.
- Results are cached for 24 hours.
- Caching is per user only.
- bq query –nouse_cache ‘<QUERY>’
- Cached by Default unless
    - A destination table is specified.
    - If any referenced tables or logical units have changed since results previously cached.
    - If any referenced tables have recently received streaming inserts even if no new rows have arrived.
    - If the query uses non-deterministic functions such as CURRENT_TIMESTAMP(), NOW(), CURRENT_USER()
    - Querying multiple tables using a wildcard
    - If the query runs against an external data source.

## Export
- Destination has to be GCS.
    - Can copy table to another BigQuery dataset though.
- Can be exported as JSON/CSV/Avro
    - Default is CSV
- Only compression option: GZIP
    - Not supported for Avro
- To export > 1 GB
    - Need to put a wildcard in destination filename
    - Up to 1 GB of table data in a single file
- bq extract ‘project:dataset.table’ gs://bucket

## Query Plan Explanation
- In web UI, click on “Explanation”
- Good for debugging complex queries not running as fast as needed/expected.
- Monitoring Query Performance (UI)
    - `Details` button after running query.
    - Colors
        - Yellow – Wait
        - Purple – Read
        - Orange – Compute
        - Blue – Write
    - Less parallel inputs => better performance => best cost 

## Slots
- Unit of computational capacity needed to run queries.
- BQ calculates on basis of query size, complexity
- Usually default slots are sufficient
- Might need to be expanded over time, complex queries
- Subject to quota policies ($$)
- Can use StackDriver Monitoring to track slot usage.

## Clustered Tables
- Order of columns determines sort order of data.
- Think of Clustering Columns in Cassandra
- When to use:
    - Data is already partitioned on date or timestamp column.
    - You commonly use filters or aggregation against particular columns in your queries.
- Does not work if the clustered column is used in a complex filter (used in a function in the filter expression)

## BigQuery ML
- Create and execute machine learning models in BQ using standard SQL
- Supported models
    - Linear regression
    - Binary Logistic regression
    - Multiclass logistic regression for classification
    - K-means clustering for data segmentation (beta)
    - Import TensorFlow Models (alpha)
- TensorFlow DNN
    - Classifier
    - Regressor
- Benefits from not having to export and re-format data
    - Brings machine learning to the data.

## Best Practices
### Costs
- Avoid SELECT *
    - Query only columns you need.
- Sample data using preview options
    - Don’t run queries to explore or preview table data.
- Price your queries before running them.
    - Before running queries, preview them to estimate costs.
- Limit query costs by restricting the number of bytes billed.
    - Use the maximum bytes billed setting to limit query costs.
- LIMIT doesn’t affect cost
    - Do not use LIMIT clause as a method of cost control as it does not affect the amount of data that is read.
- View costs using a dashboard and query your audit logs
    - Create a dashboard to view your billing data so you can make adjustments to your BigQuery usage. Also consider streaming audit logs to BigQuery to analyze usage patterns.
- Partition data by date
- Materialize query results in stages
    - Break large query into stages where each stage materializes the results by writing to a destination table.
    - Querying smaller destination table reduces amount of data that is read and lowers costs.
- Consider cost of large result sets
    - Use default table expiration time to remove data when not needed.
    - Good for when writing large query results to a destination table.
- Use streaming inserts with caution
    - Only use if data is needed immediately available.
### Query Performance
- Input data and data sources (I/O)
    - Control projection – Avoid SELECT *
    - Prune partitioned queries
        - Use partition columns to filter
    - Denormalize data when possible
        - JSON, Parquet, or Avro
        - When creating, specify Type in the Schema as RECORD
    - Use external data sources appropriately
        - If performance is a top priority, do not use external source
    - Avoid excessive wildcard tables
        - Use most granular prefix possible
- Communication between nodes (shuffling)
    - Reduce data before using a JOIN
    - Do not treat WITH clauses as prepared statements
    - Avoid tables sharded by date
        - Use time-based partitioned tables instead
            - Copy of schema and metadata is maintained for each sharded table.
            - BQ might have to verify permissions for each queries table. (overhead)
    - Avoid oversharding tables
- Computation
    - Avoid repeatedly transforming data via SQL queries
    - Avoid JavaScript user-defined functions.
        - Use native UDFs instead.
    - Use approximate aggregation functions
        - COUNT(DISTINCT) vs. APPROX_COUNT_DISTINCT()
    - Order query operations to maximize performance
        - Only use in the outermost query or within window clauses.
        - Push complex operations to the end of the query.
    - Optimize join patterns
        - Start with the largest table
    - Prune partitioned queries
- Outputs (materialization)
    - Avoid repeated joins and subqueries
    - Carefully consider materializing large result sets
    - Use LIMIT clause with large sorts
- Anti-patterns
    - Self-joins
        - Potentially doubles number of output rows
        - Use window function instead
    - Data skew
        - If query processes keys that are heavily skewed to a few values, filter your data as early as possible.
    - Cross joins (Cartesian product)
        - Avoid joins that generate more outputs than inputs.
        - Pre-aggregate data first if it is required.
    - DML statements that update or insert single rows
        - Use batch.
### Storage Optimization
- Use expiration settings to remove unneeded tables and partitions
    - Configure default table expiration for datasets
    - Configure expiration time for tables
    - Configure partition expiration for partitioned tables
- Take advantage of long term storage
    - Untouched tables (90 days) are as cheap as GCS Nearline
    - Each partition is considered separately.
- Use pricing calculator to estimate storage costs

## Architecture
- Jobs (queries) can scale up to thousands of CPU’s across many nodes, but the process is completely invisible to end user.
- Storage and compute are separated, connected by petabit network.
- Columnar data store
    - Separates records into column values, stores each value on different storage volume.
    - Poor writes (BQ does not update existing records)
    
### Components

- Dremel - Execution Engine
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

- Colossus - Distributed Storage
	- Google's latest generation distributed file system.
	- Colossus cluster in each Google datacenter.
        - Each cluster has enough disks to give every BigQuery user thousands of dedicated disks at a time.
    - Handles replication, recovery (when disk crash), and distributed management (so no single point of failure).
	- Leverages columnar storage

- Borg - Compute
	- Gives thousands of CPU cores dedicated to processing your task.
	- Large-scale cluster management system.
	- Run on dozens of thousands of machines and hundreds of thousands of cores.
	- Assigns resources to jobs - Dremel cluster in this scenario.

- Jupiter - The Network
	- Can deliver 1 Petabit/sec of total bisection bandwidth.
        - Enough to allow 100,000 machines to communicate with any other machines at 10Gbs.
	- Full duplex bandwidth means that locality within cluster is not important.
		- Each machine can talk at 10Gbs regardless of rack.

    
## Cost
- Based on:
    - storage (amount of data stored)
    - querying (amount of data/number of bytes processed by query)
    - streaming inserts.
- Storage options are active and long term
    - Modified or not past 90 days
- Query options are on-demand and flat-rate

## IAM
- Security can be applied at project and dataset level, but not at table or view level.
- Predefined roles BQ
    - Admin – Full access
    - Data owner – Full dataset access
    - Data editor – edit dataset tables
    - Data viewer – view datasets and tables
    - Job User – run jobs
    - User – run queries and create datasets (but not tables)
    - metaDataViewer
    - readSessionUser – Create and use read sessions within project.
- **Authorized views** allow you to share query results with particular users/groups without giving them access to underlying data.
    - Restrict access to **particular columns or rows**
    - Create a **separate dataset** to store the view.
    - How:
        - Grant IAM role for data analysts (bigquery.user)
            - They won’t have access to query data, view table data, or view table schema details for datasets they did not create.
        - (In source dataset) Share the dataset, In permissions go to Authorized views tab.
            - View gets access to source data, not analyst group.

