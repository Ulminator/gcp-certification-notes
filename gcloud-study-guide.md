# Big Data

## Big Query

## DataFlow
	- Transform data
	- Loosely semantically equivalent to Apache Spark
	- Based on Apache Beam
		- Older version (1.x) was not based on Beam

	Flink Programming Model
		- Data Source -> Transformations -> Data Sink

	DAG
		- Flink
		- Beam
		- TensorFlow
		- Can do topological sort
		- Parallelization

	Apache Beam Architecture
		Pipeline
			- Entire set of computations
			- Not linear, it is a DAG

			- A single, potentially repeatable job, from start to finish, in Dataflow
			- Defined by driver program
				- Actual computations run on a `backend`, abstracted in the driver by a `runner`

			Driver
				- Defines a computation DAG (pipeline)

			Runner
				- Executes Dag on a backend

			Beam supports multiple backends
				- Spark
				- Flink
				- Dataflow
				- Beam Model

		PCollection
			- Data set in pipeline (immutable)

			- Specialized container classes that can represent data sets of virtually unlimited size

			- Fixed size: text file or BQ table
			- Unbounded: Pub/Sub subscription

			Side Inputs
				- Inject additional data into some PCollection

		Transforms
			- Data processing step in pipeline

			- Takes one or more PCollections as input, performs a processing function that you provide on the elements of that PCollection, and produces an output PCollection

			- "What (operations)/Where (windows)/When (triggers)/How"

		Sources & Sink
			- Different data storage formats, ex. GCS, BigQuery tables.

	Require a Staging Location where intermediate files may be stored

	Apache Beam Runner --runner=DataflowRunner

	Need Dataflow API enabled
	Service Account needed as well.

	PipelineOptionsFactory
		- used to create an instance of a pipeline using the command line arguments passed in

	PipelineOptions
		- interface used to configure beam pipelines

	IAM + Dataflow
		- Permission exists that allows devs to work with pipelines without having access to the data.

	When to use Side Inputs?
		View
			- Singleton transform an immutable collection from PCollection objects
			- Used as side input to another pipeline

		PCollectionView
			- An immutable view of a PCollection used as a side input in Dataflow pipelines

		JavaProjectsThatNeedHelp.java is a good example

	System Lag
		- The max time an element has been waiting for processing in this stage of pipeline.

	Data Watermark(?)
		- Indicates all windows ending before or at this timestamp are closed.
		- No longer accept any streaming entities that are before this timestamp.

	Wall Time
		- How long the processing takes.

	Stopping A Dataflow Job
		- Cancelling
			- Immediately stop and abort all data ingestion and processing.
			- Buffered data may be lost.
		- Draining
			- Cease ingestion, but will attempt to finish processing any remaining buffered data.
			- Pipeline resources will be maintained until buffered data has finished processing and any pending output has finished writing.

## Pub/Sub

	- Messaging "middleware"
	- Many-to-many asynchronous messaging
	- Decouple sender and receiver
	- Attributes can be set by sender (KV pairs)

	Publisher apps create and send messages to a Topic
		- Messages persisted in a message store until delivered/acknowledged
		- One queue per subscription
	
	Subscriber apps subscribe to a topic to receive messages
		- Push - WebHook endpoint (must accept POST HTTPS request)
		- Pull - HTTPS request to endpoint

	Once acknowledged by subscriber, message deleted from queue

	Use Cases
		- Balancing workloads in network clusters
		- Asynchronous order processing
		- Distributing event notifications
		- Refreshing of distributed caches
		- Logging to multiple systems simultaneously
		- Data streaming

	Architecture
		- Data plane
			- Handles moving messages between publishers and subscribers
			- Forwarders
		- Control plane
			- Handles assignment of publishers and subscribers to servers on the data plane
			- Routers

	Publishers
		- Any application that can make HTTPS requests to googleapis.com
			- Pretty much anything

	Acknowledgement Deadline
		- Per subscriber
		- Once deadline has passed, an outstanding message becomes unacknowledged

	Encoding Data as a Bytestring (utf-8) is required for publishing

	Topic.Batch (the call publish)

	Streaming (Dataflow) Data from PubSub and Writing to BQ
		```
		// Java Classes needed
		PubsubIO
		BigqueryIO
		TableFieldSchema
		TableRow
		TableSchema
		```

# Storage Technologies
	Block storage for compute VMs - persistent disks or SSDs
	Immutable blobs like video/images - Cloud Storage
	OLTP - Cloud SQL or Cloud Spanner
	NoSQL Documents like HTML/XML - Datastore
	NoSQL Key-values - BigTable (~HBase)
	Getting data into Cloud Storage - Transfer Service

## Use Cases
	Storage for Compute, Block Storage
		- Persistent (hard disks), SSD
	Storing media, Blob Storage
		- File system - maybe HDFS
		- Cloud Storage
			- Does not require a master controlling entity unlike HDFS
	SQL Interface atop file data
		- Hive (SQL-like, but it's actually MapReduce on HDFS)
		- BigQuery
			- Has a columnar datastore under the hood unlike Hive
	Document database (hierarchical/tree like data), NoSQL
		- CouchDB, MongoDB (key-value/indexed database)
		- DataStore
			- Fast queries that scale reasonably with the data.
			- Indexes built on every column.
	Fast scanning, NoSQL
		- HBase (columnar database)
		- BigTable
			- Essentially HBase
	Transaction Processing (OLTP)
		- RDBMS
		- Cloud SQL, Cloud Spanner
	Analytics/Data Warehouse (OLAP)
		- Hive
		- BigQuery

	Mobile Specific
		Storage for Compute, Block Storage 
			- With mobile SDKs
				- Cloud Storage for Firebase
			- Fast random access with mobile SDKs
				- Firebase Realtime DB

## Block Storage (HDFS equivalent)
	- Data stored in cylinders or some physical form.
	- Data is not structured
	- Lowest level of storage - no abstraction at all
	- Meant for use from VMs
	- Location tied to VM location

	- Data stored in volumes (called blocks)

## Cloud Storage
	- Create buckets to store data
	- Buckets are globally unique
		- Name (globally unique)
		- Location
		- Storage Class
			- Multi-regional
				- Frequent access from anywhere in the world

				- Serving website content, interactive workloads, or mobile game and gaming applications.
				- Highest availability of storage classes (99.5%)
				- Geo-redundant
					- Stores data redundantly in at least 2 regions separated by at least 100 miles within the multi-regional location of the bucket.
				- Most expensive.

			- Regional
				- Frequent access from specific region

				- 99.9% availability
				- Appropriate for storing data used by Compute Engine
				- Better performance for data-intensive computations.


			- Nearline
				- Accessed once a month max

				- 99.0% availability
				- 30 day minimum storage duration.
				- Costs assocaited with data retrieval.

				- Data backup, disaster recovery, and archival storage.

			- Coldline
				- Accessed once a year at max

				- 99.0% availability
				- Same throughput and latency as non cold storage services like Regional or Multi-Regional
				- 90 day minimum storage duration, costs for data access, and higher per-operation costs
				- Costs assocaited with data retrieval.

				- Infrequently accessed data, such as data stored for legal or regulatory reasons

			- Storage classes can change, but the objects (files) within them retain their storage class.

	gsutil mb -c regional -l asia-east1 gs://bucket-name
		- Creates a new bucket

	gcloud compute regions list

	gsutil ls
		- List all buckets

	gsutil cp -r -p gs://bucket-1/* gs://bucket-2/
		- Copy recursively and copy over permissions as well.

	gsutil list -L -b gs://bucket
		- Show metadata (-L) for the bucket

	LifeCycle Management for Buckets
		- Specify how long file sticks around in bucket before a specific activity is triggered.

		gsutil lifecycle get gs://bucket
			- Gets lifecycles setup for a bucket

		gsutil lifecycle set setting.json gs://bucket
			- Applies a lifecycle setting
			- Can be used to automatically delete log files.
			- settings.json contains rules

		Fix 403 errors
			- gsutil config -b

	Transfer Service
		- Helps get data into Cloud Storage
		- From where?
			- AWS S3 Bucket
			- HTTP/HTTPS location
			- Local files
				- Just use gsutil instead for on prem
			- One gcloud bucket to another
		- Simple to do through the UI
		
		- One time vs. recurring transfers
		- Delete from destination if they don't exist in source
		- Delete from source after copying over
		- Periodic synchronization of source and destination based on file filters.

	ACLs and API access with Service Account
		gcloud auth activate-service-account --key-file creds.json

		gcloud init
			- Reinitialize configuration

		Access List Controls
			gsutil act get gs://bucket/setup.html > acl.txt
				- Gets the default access list that's been assigned to setup.html

			gsutil acl set private gs://bucket/setup.html
				- The setup.html is now private

			gsutil acl ch -u AllUsers:R gs://bucket/setup.html
				- All users have read permissions

	Customer-Supplied Encryption Keys and Life-Cycle Management
		- GCS always encrypts the file server side before saving to disl

		- Implement Customer Supplied Key
			1. Generate encryption key
				python -c 'import base64; import os; print(base64.encodebytes(os.urandom(32)))'
			2. Edit config file `.boto`
			3. Edit field in file `encryption_key=${new encryption key}`

		- Can still read files that are downloaded even if the customer supplies an encryption key

		- Rotate CSEK keys
			Edit .boto
			Comment out encryption-key
			Make it the decryption-key
			Generate a new key and make it the encryption-key

			`gsutil rewrite -k gs://bucket/setup.html`
				First decrypt the file using the old key
				Then encrypts the file using the new key

		gsutil versioning get gs://bucket
		gsutil versioning set on gs://bucket
			- When saving a file with the same name as an already existing file, it will create a new file with a timestamp

	Versioning, Directory Sync

		Synchronization
			gsutil rsync -r . gs://bucket
				- Copies all files from current directory into bucket

		Sharing of Buckets Across Projects 
			- Use service account private key from project-2 in a VM in project-1
			- Auth a VM with that keyfile. Re-intialize gcloud and set project to project-2

## BigQuery (Hive equivalent)
	- Latency bit higher than BigTable, DataStore
		- Prefer those for low latency
	- No ACID properties
		- Can't use for transaction processing (OLTP)
	- Great for analytics/business intelligence/data warehouse (OLAP)
	- OLTP needs strict write consistency, OLAP does not
		- Atomicity, Consistency, Isolation, Durability

	Enterprise Data Warehouse

	SQL queries - with Google storage underneath

	Fully managed - no server, no resources deployed

	Access through - Web UI, REST API, clients

	Data Model
		- Dataset = set of tables and views
		- Table must belong to dataset
		- Dataset must belong to a project
		- Tables contain records with rows and columns (fields)
		- Nested and repeatable fields are OK

	Table Schema
		- Can be specified at creation time
		- Can also specify schema during intitial load
		- Can update schema later too

	Table Types
		- Native tables: BigQuery storage
		- External tables
			- BigTable
			- Cloud Storage
			- Google Drive
		- Views

	Schema Auto-Detection
		- Available while
			- Loading data
			- Querying external data
		- BigQuery selects a random file in the data source and scans up to 100 rows of data to use a representative sample
		- Then examines each field and attempts to assign a data type to that field based on the values in the sample
		- Important due to dealing with data not owned by BigQuery (external sources)

	Loading Data
		- Batch
			- CSV
			- JSON (newline delimited)
			- Avro
				- Open source data format that bundles serialized data with the data's schema in the same file.
			- GCP DataStore backups (particularly logs)
				- BigQuery converts data from each entity in Cloud Datastore backup files to BigQuery's data types
		- Streaming
			- High volume event tracking logs
			- Realtime dashboards

	Alternatives to Loading
		- Public datasets
		- Shared datasets
		- Stackdriver log files (need export - but direct)

	Querying and Viewing
		- Interactive Queries
			- Default mode (executed as soon as possible)
			- Count towards limits on
				- Daily usage
				- Concurrent usage

		- Batch Queries
			- Will schedule these to run whenever possible (idle resources)
			- Don't count towards limit on concurrent usage
			- If not started within 24 hours, BQ makes them interactive

		- Views
			- Logical - not materialized
			- Underlying query will execute each time view is accessed
			- Billing will happen accordingly
			- Reduces query complexity, restict access to data, and construct different logical tables from the same physical table.

			Access Control
				- Can't assign access control - based on user running view
				- Can create authorized view
					- Share query results with groups without giving read access to underlying data
				- Can give row-level persmissions to different users within same view
				- Cannot restrict access at a table level, only at dataset level.
				- What are Authorized Views?

			- Can't export data from a view
			- Can't use JSON API to retrieve data
			- Can't mix standard and legacy SQL
				- ex. standard SQL query can't access legacy-SQL view
			- No user-defined functions allowed
			- No wildcard table references
				- Due to partitioning
			- Limit of 1000 authorized views per dataset

		- Partitioned Tables
			Not present initially in BQ

			- Special table where data is partitioned for you.
			- No need to create partitions manually or programmatically
			- Manual partitions - performance degrades
			- Limit of 1000 tables per query does NOT apply

			- Date-partitioned tables offered by BQ
			- Need to declare table as partitioned at creation time
			- No need to specify schema (can do while loading data)
			- BQ automatically creates date partitions

			Types of BQ Partitioning
			- Tables partitioned by ingestion time
				- Include a pseudo column named _PARTITIONTIME
				- Need to update schema of table before loading data if you need to load data into a partition with a different schema.
			- Tables partitioned based on a TIMESTAMP or DATE column
				- Do not need a _PARTITIONTIME pseudo column
				- 2 Special Partitions are created
					- __NULL__ parititon
						- represents rows with NULL values in the parititoning column
					- __UNPARTITIONED__ partition
						- represents data that exists outside the allowed range of dates
				- With the exception of those two partitions, all data in the partitioning column matches the date of the partition identifier.
					- This allows a query to determine which partitions contain no data that satisfies the filter conditions. Queries that filter data on the partitioning column can restrict values and completely prune unnecessary partitions.

	Query Plan Explanation
		- In the web UI, click on "Explanation"
		- Good for debugging complex queries not running as fast as needed.

	Slots
		- Unit of Computational capacity needed to run queries
		- BQ calculates on basis of query size, complexity
		- Usually default slots sufficient
		- Might need to be expanded for very large, complex queries
		- Slots are subject to quota policies ($$)
		- Can use StackDriver Monitoring to track slot usage

	No OPS and Serverless

	Querying
	```
		select * from [project.dataset.name] limit 1000
		# square brackets required for legacy sql

	```
	Referencing BQ Tables
		- datasetName.tableName

	Table ID
		- projectId.datasetId.tableName

	Billed for all data in table (even if query contains a LIMIT clause)

	Public Datasets Available for Free
		- https://cloud.google.com/bigquery/public-data
		- Only need pay to query
		- Access by `bigquery-public-data.samples.table` in query

	What effects the amount of data processed per query?
		- Reducing the number of columns reduces the amount of data processed
			- Implies the entire table is not scanned when queries are run (columnar)

	CLI
		```
			bq help # see general commands
			bq help mk # more help on creating

			bq show <dataset>
				# last_modified, ACLs (owners, writers, readers), Labels

			bq show <project id>:<dataset>.<table>

			bq head -n 10 <project id>:<dataset>.<table>

			# Storage Bucket -> BQ

			bq load --source_format=CSV \
				babynames.names_2011 \ # table name
				gs://bucket/babynames/yob2011.txt \
				name:string,gender:string,count:integer

			bq load --source_format=CSV \
				babynames.allNames \ # table name
				gs://bucket/babynames/yob20*.txt \
				name:string,gender:string,count:integer

			# BQ -> Storage Bucket

			bq extract babynames.allNames gs://bucket/export/all_names.csv
		```
	
	~/.bigqueryrc
		- specifies project

	Aggregations and Conditionals in Aggregations

		Turn off legacy sql to use backticks (needed for wildcards)	
			- Standard SQL needs table names enclosed in backticks
			- Wildcard expressions allow to query multiple tables in one go

		SUM(IF(f.departure_delay > 0, 1, 0)) AS num_delayed
			-> if f.departure_delay ? 1 : 0

	Subqueries and Joins

		# YYYY-mm-dd
		CONCAT(CAST(year AS STRING), '-', LPAD(CAST(month AS STRING), 2, '0'), '-', LPAD(CAST(day AS STRING), 2, '0'))


		```
			SELECT
			  airline,
			  num_delayed,
			  total_flights,
			  num_delayed/total_flights AS frac_delayed
			FROM (
			SELECT
			  f.airline,
			  SUM(IF(f.arrival_delay > 0, 1, 0)) AS num_delayed,
			  COUNT(f.arrival_delay) AS total_flights
			FROM
			  `bigquery-samples.airline_ontime_data.flights` AS f
			JOIN
			(SELECT CONCAT(CAST(year AS STRING), '-', LPAD(CAST(month AS STRING), 2, '0'), '-', LPAD(CAST(day AS STRING), 2, '0')) AS rainyday
			FROM
			  `bigquery-samples.weather_geo.gsod`
			WHERE
			  station_number = 722190
			  AND total_precipitation > 0) AS w
			ON
			  w.rainyday = f.date
			WHERE f.arrival_airport = 'ATL'
			GROUP BY f.airline
			)
			ORDER BY frac_delayed ASC
		```

	Advanced Queries using REGEX
		Multiple Tables with Standard SQL
		```
			SELECT title, SUM(views) views
			FROM 
				[bigquery-samples:wikimedia_pageviews.201112],
				[bigquery-samples:wikimedia_pageviews.201111]
			WHERE wikimedia_project = 'wp'
			AND REGEXP_MATCH(title, 'Red.*t') -- any article with red in title
			GROUP BY title
			ORDER BY views DESC
		```

		Multiple tables with Legacy SQL
			- a lot more data processed since looking at all of 2011
		```
			SELECT title, SUM(views) views
			FROM 
				TABLE_QUERY([bigquery-samples:wikimedia_pageviews],
					'REGEXP_MATCH(table_id, r"^2011[\d]{2}"}')
			WHERE wikimedia_project = 'wp'
			AND REGEXP_MATCH(title, 'Red.*t') -- any article with red in title
			GROUP BY title
			ORDER BY views DESC
		```

	Using With Statement for Subqueries

		UNNEST
			- Flattening Command
			- Apply to column with repeating records
			- Will create new rows for each entry in array
				1 ROW - email, [path1, path2, path3]
				3 ROWS - email,path1 email,path2 email,path3

		```
			SELECT
				author.email,
				diff.new_path AS path,
				author.date
			FROM
				`bigquery_public_data.github_repos.commits`,
				UNNEST(difference) diff
			WHERE
				EXTRACT(YEAR
				FROM
					author.date)=2016
		```

		OFFSET
			- Inspect some index of an array
			`difference[OFFSET(0)].new_path`

		WITH
			- Sets up a table from a subquery that can be referenced

		```
			WITH
				commits AS(
			SELECT
				author.email,
				LOWER(REGEXP_EXTRACT(diff.new_path, r'\.([^\./\(-_\-#]*))])) lang,
				diff.new_path AS path,
				author.date
			FROM
				`bigquery_public_data.github_repos.commits`,
				UNNEST(difference) diff
			WHERE
				EXTRACT(YEAR
				FROM
					author.date)=2016
			)

			SELECT
				lang,
				COUNT(path) AS numcommits
			FROM
				commits
			WHERE
				LENGTH(lang) < 8
				AND lang IS NOT NULL
				AND REGEXP_CONTAINS(lang, '[a-zA-Z]')
			GROUP BY
				lang
			HAVING
				numcommits > 100
			ORDER BY
				numcommits DESC
		```

## Cloud SQL, Cloud Spanner
	- Relational DBs
	- Support ACID properties
	- Too slow and too many checks for analytics/BI/warehousing (OLAP)

	- Cloud Spaner is Google proprietary, more advanced than Cloud SQL
		- Offers 'horizontal scaling' - i.e. bigger data, more instances, replication, etc...

	Cloud SQL
		- MySQL - fast and the usual
		- PostgreSQL - Complex Queries perform better

		- Instances need to be created explicitely
			- Specify region
		- First vs. second generation instances
			- Second gen MySQL instances allow:
				- Proxy Support (Cloud Proxy)
					- Provides secure access to your Cloud SQL Second Generation instances without having to whitelist IP addresses or configure SSL.
					- Secure connections: The proxy automatically encrypts traffic to and from the database; SSL certificates are used to verify client and server identities.
					- Easier connection management: The proxy handles authentication with Google Cloud SQL, removing the need to provide static IP addresses.
					- Starting Proxy:
						- Tell it what cloud sql instances it should establish connections to.
						- Tell it where it will listen for data coming from your application to be sent to cloud sql.
						- Tell it where it will find credentials it will use to authenticate your application to cloud sql.
				- Higher availability configuration
					- Has a failover replica
					- Must be in a different zone than original.
					- All changes made to the data on the master, including to user tables, are replicated to the failover replica during semisynchronous replication.
				- Maintenance won't take down the server

		Hard Disk (Persistent Disk) for a Production Instance? No use SSD

		Enable binary logging (for point-in-time recovery and replication)
			- See what the database was like on X day

		GCloud
			Connect
				- gcloud beta sql connect instance-name --user=root
			Run Queries on Instance
				- gcloud beta sql connect instance-name --user=root < queries.sql

		Bulk Load Data
			- Copy data to Cloud Storage Bucket
				- Optimal for bucket to be in same location as db
			- Whitelist CloudShell IP to access the SQL instance (add network)
				- Get this by running `bash ./find_my_ip.sh` in the cloudshell.
			- `mysql --host=$DB_IP --user=root --password=testpassword`

	Cloud Spanner
		Horizontal Scaling?
			- Just add more machines
		Use when
			- Need high availability
			- Strong consistency
			- Transactional reads and writes (especially writes!)
		Don't Use if
			- Data is not relational, or not even structured.
			- Want an open source RDBMS
			- Strong consistency and availability is overkill

		Data Model
			- Has tables
				- Look relational - rows, columns, strongly typed schemas

			- Specifies a parent-child relationship for efficient storage
			- Interleaved representation (like HBase)

			Parent Child relationship
				- Between tables
				- These cause physical location for fast access
				- If you query Students and Grades together, make Grades child of Student.
				- Primary key of parent table has to be part of the key in the interleaved child table.

				- Every table must have primary keys
				- To declare table is child of another...
				- Prefix parent's primary key onto primary key of child.

			Interleaving
				- Rows are stored in sorted order of primary key values
				- Child rows are inserted between parent rows with that key prefix

		Hotspotting
			- As in HBase - need to choose Primary keys carefully
			- Do not use monotonically increasing values, else writes will be on same locations - hot spotting
			- Use hash of key value if you naturally monotonically ordered keys (serial in postgres)

		Splits
			- Parent-child relationships can get complicated - 7 layers deep
			- CloudSpanner is distributed - uses "splits"
			- A split is a range of rows that can be moved around independent of others
			- Splits are added to distribute high read-write data (to break up hotspots)

		Secondary Indices
			- Like in HBase, key-based storage ensures fast sequential scan of keys
			- Unlike HBase, can also add secondary indices
				- Might cause same data to be stored twice
					- Grades -> Course table | Grades -> Students table
			- Fine-grained control on use of indices
				- Force query to use a specific index (index directives)
				- Force column to be copied into secondary index (use a STORING clause)

		Data Types
			- Remember that tables are strongly-typed (schemas must have types)
			- Non-normalized types such as ARRAY and STRUCT available too
				- STRUCTS are not OK in tables, but can be returned by queries
					- i.e. if query returns ARRAY of ARRAYs
				- ARRAYs are OK in tables, but ARRAYs of ARRAYs are not

		Transactions
			- Supports serialisability
				- All transactions appear as if they executed in a serial order, even if some of the reads, writes, and other operations of distinct transactions actually occurred in parallel.
			- Transaction support is stronger than traditional ACID
				- Transactions commit in an order that is reflected in their commit timestamps
				- These commit timestamps are "real time" so you can compare them when your watch

			- Two Transaction Modes
				- Locking read-write (slow)
					- Only one that supports writing data
				- Read-only (fast)
					- Only requires read locking
			- If making a one-off read, use "Single Read Call"
				- Fastest, no transaction checks needed!

		Staleness
			- Can set timestamp bounds
			- Strong - "read latest data"
			- Bounded Staleness - "read version no later than..."
				- (could be in past or future)

		Lab
			- At least 3 nodes in production
			- Best performance when each node CPU is under 75%

			- Primary key specified outside the closing bracket of create table statement
			- Add , INTERLEAVE IN PARENT parent_table ON DELETE CASCADE after primary key section.

## BigTable (HBase equivalent)
	- Fast scanning of sequential key values
	- Columnar database
		- Good for sparse data
	- Sensitive to hot spotting (like Cloud Spanner)
		- Data is sorted on key value and then sequential lexicographically similar values are stored next to each other.
		- Need to design key structure carefully.

	BigTable is basically GCP's managed HBase
		- Much stronger link than say Hive and BigQuery

	Advantages over HBase
		- Scalability
		- Low ops/admin burden
		- Cluster resizing without downtime
		- Many more column families before performance drop (~100k)

	Sparse Tables
		- Can't ignore with petabytes of data.
		- The null cells still occupy space.

	4-Dimensional Data model
		Row 
			- Uniquely identifies a row
			- Can be primitives, structures, arrays
			- Represented internally as a byte array
			- Sorted in ascending order
		Column Family
			- Table name in an RDBMS
			- All rows have the same set of column families
			- Each column family is stored in a separate data file
			- Set up at schema definition time.
				- Columns can be added on the fly
			- Can have different columns for each row.
		Column
			- Columns are units within a column family.
			- New columns can be added on the fly.
		Timestamp
			- Support for different versions based on timestamps of same data item. (like Spanner)
			- Omit timestamp gets you the latest data.

	Avoid BigTable When
		- Need transaction support (OLTP) - Use Cloud SQL or Spanner
		- Don't use for data less than 1 TB (can't parallelize)
		- Don't use if analytics/business intelligence/data warehousing - use BigQuery instead
		- Don't use for documents or highly structured hierarchies - Use DataStore instead.
		- Don't use for immutable blobs like movies each > 10MB - Cloud Storage instead

	Use BigTable When
		- Use for very fast scanning and high throughput
		- Use for non-structured key/value data
		- Where each data item < 10MB and total data > 1 TB
		- Use where writes are infrequent/unimportant (no ACID) but fast scans crucial
		- Use for Time Series data

	Avoiding Hotspotting (Row keys to Use)
		- Field Promotion: Use in reverse URL order like Java package names
			- THis way keys have similar prefixes, different endings
		- Salting
			- Hash the key value
		- Timestamps as suffix in key

	Row Keys to Avoid
		- Domain names (as opposed to field promotion)
			- Will cause common portion to be at end of row key leading to adjacent values to not be logically related
		- Sequential numeric values
		- Timestamps alone
		- Timestamps as prefix of row-key
		- Mutable or repeatedly updated values

	Size Limits:
		- Row keys: 4KB per key
		- Column Families: ~100 per table
		- Column Values: ~10 MB each
		- Total Row Size: ~100 MB

	"Warming the Cache"
		- BigTable will improve performance over time
		- Will observe read and write patterns and redistribute data so that shards are evenly hit
		- Will try to store roughly same amount of data in different nodes
		- This is why testing over hours is important to get true sense of performance

	SSD or HDD Disks
		- Use SSD unless skimping on costs
		- SSD can be 20x faster on individual row reads
			- Less important with batch reads or sequential scans
		- More predictable throughput too (no disk seek variance)
		- Don't even think about HDD unless storing > 10 TB and all batch queries
		- The more random access, the stronger case for SSD
			- Purely random -> maybe use datastore.

	Reason for Poor Performance
		- Poor schema design
		- Inappropriate workload
			- too small (< 300 GB)
			- used in short bursts (needs hours to tune perf. internally)
		- Cluster too small
		- Cluster just fired up or scaled up
		- HDD used instead of SSD
		- Development v. Production instance

	Schema Design
		- Each table has just 1 index - row key. Choose well
		- Rows sorted lexicographically by row key
		- All operations are atomic at row level
		- Related entities in adjacent rows

	DEMO
		```
			#Get HBase shell
			curl -f -O ...quickstart/GoogleCloudBigtable...
			#unzip it
			#cd into it
			gcloud beta bigtable instances list
			# Get into HBase (BigTable) shell
			./quickstart.sh

			#IN THE SHELL
			# Creates student table and personal column family
			create 'students', 'personal'
			# Show tables
			list
			# Add to Table (row key is 12345)
			put 'students', '12345', 'personal:name', 'john'
			put 'students', '12345', 'personal:state', 'CA'
			# List all information in students table
			scan 'students'
			# Delete table
			drop 'students'
		```

## Datastore (MongoDB equivalent)
	- Document data (XML or HTML) - has a characteristic pattern
	- Key value structure, i.e. structured data
	- Typically not used either for OLTP or OLAP
		- Fast lookup on keys is the most common use-case

	- Speciality is that query execution depends on size of returned result and not the size of the data set.
		- Datastore is best in situations where it is necessary to lookup non sequential keys (needle in haystack)

	- Fast to read/slow to write

	Comparison
		Traditional RDBMS
			- Atomic
			- Indices for fast lookup
			- Some queries use indices - not all
			- Query time depends on both size of data set and size of result set

			- Structured relational data
			- Rows stored in tables
			- Rows consist of fields
			- Primary keys for unique ID

			- Rows of table have same properties (Schema strongly enforced)
			- Types of all values in a column are the same
			- Multiple inequeality conditions

		DataStore
			- Atomic
			- Indices for fast lookup
			- All queries use indices!
			- Query time depends on size of result set alone

			- Structured hierarchical data (XML, HTML)
			- Entities (row) of different Kinds (table) (like HTML tags)
				- Entities are of different Kinds, not stored
			- Entities consist of Properties
			- Keys for unique ID

			- Entities of a kind can have different properties
				- Think optional tags in HTML
			- Types of different properties with same name in an entity can be different
			- Only 1 inequality filter per query

	Avoid DataStore When
		- Don't use if you need very strong transaction support (OLTP)
			- It's okay for basic ACID support though
		- Don't use for non-hierarchical or unstructured data
			- BigTable is better
		- Don't use for analytics/business intelligence/data warehousing
			- BigQuery instead
		- Don't use if application has a lot of writes and updates on key columns

	Use DataStore When
		- Scaling of read performance - to virtually any size
		- Use for hierarchical documents with KV data

	Full Indexing
		- Built in indices on each property (~field) of each entity kind (~table row)
		- Composite indices on multiple property values
		- If you are certain a property will never be queried, can explicitely exclude it from indexing
		- Each query is evaluated using its "perfect index"

	Perfect Index
		- Given a query, which is the index that most optimally returns query results?
		- Depends on following (in order)
			- equality filter
			- inequality filter (only 1 allowed)
			- sort conditions if any specified

	Implications of Full Indexing
		- Updates are really slow
		- No joins possible
		- Can't filter results based on subquery results
		- Can't include more than one inequality filter (1 is OK)

	Multi-Tenancy
		- Separate data partitions for each client organizations
		- Can use the same schema for all clients, but vary the values
		- Specified via a namespace (inside which kinds and entities can exist)

	Transaction Support
		- Can optionally use transactions - not required
		- Stronger than BigQuery and BigTable

	Consistency
		- Two consistency levels possible
			- Strongly consistent	
				- Return up to date result, however long it takes
			- Eventually consistent
				- Faster, but might return stale data

	Serverless

# Machine Learning

## Concepts

	        ML Based                |         Rule Based
	Dynamic                         | Static
	Experts optional                | Experts required
	Corpus (training data) required | Corpus optional
	Training step                   | No training step

	Types of Machine Learning Problems
		
		- Classification
		- Regression
		- Clustering
		- Rule-Extraction

	Deep Learning
		- Algorithms that learn what features matter
	
	Neural Networks
		- The most common class of deep learning algorithms
		- Used to build representation learning systems

	Neurons
		- Simple building blocks that actually "learn"

	Representation Learning Algorithm
		- Algorithm identifies important features on its own

	Supervised Learning
		- Labels associated with the training data are used to correct the algorithm.

	Unsupervised Learning
		- The model has to be set up right to learn the structure in the data.

	Neurons as Learning Units
		- Deep Learning based binary classifier

	Learning Arbitrarily Complex Functions
		- XOR function ex.

	Each Neuron only applies 2 simple functions to its inputs
		- A linear (affine) transformation
			- Like linear regression

			x1 * W1 + b
			x2 * W2 + b
			...
			xn * Wn + b

				- W = Weights
					- Shape of W
						- First dimension is equal to number of dimensions of feature vector.
						- Second dimension is equal to the number of parameters required to be tuned per neuron. (same goes for b)
							- # of constants that need to be calculated for each neuron.
				- b = Bias
					- Determined during training process

		- An activation function
			- Helps to model non linear functions
				- Like XOR, logistic regression

			- INTRODUCES non-linearity into the network

			ex. ReLu (Rectified Linear Unit)

			max(Wx + b, 0)

		Finding the "best" values of W and b for each neuron is crucial.

		The "best" values are found using the cost function, optimizer, and training data.

		During training, the output of the deeper layers may be "fed back" to find the best W, b.
			- Back propogation
				- Standard algorithm for training neural networks

		Most common activation function: ReLu

		Logistic Regression: SoftMax

		Modeling non linear functions: Chain output from affine transformation into activation function.

		Linear Regression requires just 1 neuron with just an affine transformation

		Simple Regression

		y = Ax + B

				y1         1         x1       e1
				y2         1         x2       e2
			[ y3 ] = A [ 1 ] + B [ x3 ] + [ e3 ]
				.          .         .        .
				.          .         .        . 
				yn         1         xn       en
		
		Minimize Least Square Error

			ei = yi - y'i (Residual)

		Optimizers for the "Best fit"
			- Method of Moments
			- Method of Least Squares
			- Maximum likelihood estimator

		"Learning" XOR function
			- 3 neurons, 2 layers, non-linear activation (ReLu) function
			- 2 layer feed forward neural network

		Choice of Activation Function
			- Input layers use identity function as activation: f(x) = x
				- Output is some sort of weighted output of data
			- Inner hidden layers typically use ReLU
			- Output layer in XOR is identity function
				- Only makes sense to use with linear functions like regression

## TensorFlow

	https://tensorflow.org/

	Numerical computation using data flow graphs.

	Advantages
		- Distributed
			- Runs on a cluster of machines or multiple CPUs/GPUs on the same machine.
		- Suite of Software
			- TensorFlow, TensorBoard, TensorFlow Serving (deploying)
	
	Uses
		- Research and development of new ML algorithms
		- Taking models from traing to production
		- Large scale distributed models
		- Models for mobile and embedded systems

	Strengths
		- Easy to use, stable Python API
		- Runs on large as well as small systems
		- Efficient and performant
		- Great support from Google

	Challenges
		- Distributed support still has a ways to go
		- Libraries still being developed
		- Writing custom code is not straightforward

	Everything is a Graph
		- Nodes = Computations
		- Edges = Data Items => Tensors

	1st Step => Build Computation Graph
	2nd Step => Use tf.Session() to execute
		with tf.Session() as sess:
			sess.run(comp)

	Need to close session when done with it
		- This is why it is usually done with `with` block.

	DAGs can not complete any of its computations if it has a cycle.

	Modelling Cyclical Dependencies
		- Backpropogation feedback
		- Output of computation graph is input into same computation graph, but updates parameters. (unroll)

	Any node in a DAG can be thought of as the output of a smaller DAG
		- Smaller DAGs be computed in parallel over many machines in a cluster.

	Tensor
		- Central unit of data in TensorFlow.
		- Set of primitive values shaped into an array of any number of dimensions.
			- Scalars are 0-D(imensional) tensors
			- Vectors ([1, 2]) are 1-D tensors
			- N-Dimensional matrices are N-D tensors
		- Characterization
			- Rank = Number of dimensions in a tensor
			- Shape = Number of elements in each dimension
				- Is a vector with 1 element corresponding to each dimension in the tensor.
				- Shape of a scalar = []
				- Shape of [[[1,2,3],[3,2,1]]] = [3,2,1]
			- Data Type = Type of each element in Tensor
				- int, float, string, boolean

	Constants
		- Immutable values that do not change

	Placeholders
		- A way to specify inputs into a graph
		- Values you specify while training

		Hold the place for a Tensor that will be fed at runtime, in effect becoming an "input" node

		Ex. In linear regression: A model should have the ability to accept different X and Y values

		- Use feed_dict to input data into placeholder

	Variables
		- Parameters of the ML model
		- These are altered during the training process

		Ex. Constants A and B in the formula y = A + Bx

		Mutable tensor values that persist across multiple session calls to Session.run()

		Need to be initializes as follows:
			init = tf.global_variables_initializer()

			with tf.Session() as sess:
				sess.run(init)

		Can also initialize specific variables likes so:
			init = tf.variables_initializer([W])

	TensorBoard
		- Named Scopes = Shaded with gray and have (+) sign
			- Represents a logical grouping of several computations

	Linear Regression
		X causes Y
		Best fitting line: Minimizing least square error

	Working with Images (Spatially Arranged Data)
		Convolutional Neural Networks (CNNs)

		Input -> Pixels -> Edges -> Corners -> Object Parts -> ML Classifier

		Visible or Hidden Layers

		Drop Out Factor
			- Break at random, a bunch of interconnections in your network

		Inception Algorithm uses CNNs

		Images as Tensors
			- Pixels
				- Each holds a value based on the type of image
				- Grayscale -> 0.0 - 1.0 (intensity) 1 channel
				- RGB -> R,G,B: 0-255
					- 3 values to represent 3 colors: 3 channels

			(# pixels on x, # of pixels on y, # of channels)

		List of Images as 4-D Tensors
			- Images need to be the same size

			Ex. Shape = (10, 6, 6, 3)
				=> 10 Images, Each 6x6, RGB

		An image is expressed using CMYK colors (Cyan, Magenta, Yellow and Key). How many channels are needed for a tensor representing this image? 4 channels

	Text Processing
		- RNNs

	Sequences of Data (like timeseries)
		- RNNs

	tf.train.string_input_producer
		- Takes in a list and creates a queue with the items of that list

	tf.WholeFileReader()
		- Reads entire file at once (image)

	Coordinator
		- Allows to manage and coordinate multiple threads
		- A method call on coordinator will wait for all of your threads to finish processing.
		- Can also stop processing with a single call as well.

		coord = tf.train.Coordinator()

		coord.request_stop()
		coord.join(threads)

	QueueRunner
		- Allows you to process elements in a queue in parallel using threads
		
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	Image Transformations

		Converts all tensors to a single tensor with a 4th dimension
			images_tensor = tf.stack(image_list)
				4 images of (224,224,3) can be accessed as (0,224,224,3)...


		tf.stack()
			- Converts a list of tensors of Rank R to a single tensor of rank R+1.
			- The tensors in the list must have the same shape

	K-Nearest-Neighbors
		- Supervised learning
		- Uses all training data
		- Find out what data is most similar to input. Use that similar item to classify the input.

		Defining Similarity => Distance measures
			Euclidean Distance
			Hamming Distance
			Manhatten Distance
				
				Other names:
					L1 Distance
					Snake Distance
					City Block Distance
				Corresponds to # of grid blocks that must be traversed
				Can only move along x and y axis (no diagonal)

				(1,0) -> (5,4)
					L1 = abs(5-1) + abs(4-0) = 8

	One-Hot Vectors
		Vector [0,0,0,0,1,0,0,0,0,0] # The label of image
		Index   0 1 2 3 4 5 6 7 8 9

		Represents the labels in MNIST

	Implementing Regression in Tensorflow
		
		- Baseline
			- Non-Tensorflow implementation

			Pandas for dataframes
			NumPy for arrays
			Statsmodels for regression
			Matplotlib for plots

		- Computation Graph
			- Neural netowrk with 1 neuron
				- Affine transformation suffices

		- Cost function
			- MSE
				- Quantifying goodness of fit

		- Optimizer
			- Gradient Descent optimizers
				- Improving goodness of fit

		- Training
			- Invokes optimizer in epochs
				- Batch size for each epoch

			- Decisions
				- Initial values
				- Types of optimizers
				- Number of epochs
				- Batch size

		- Converged Model
			- Values of W and b
				- Compare to baseline

		ML Based Regression Model
			- Treat all x variables as features
				- Use numpy reshape(-1,1)
					- Takes 1 list and makes a list of lists
					- Goes from coordinate geometry form to feature vectors
			- Labels are y values
			
		Stochastic Gradient Descent
			- 1 point at a time

		Mini-Batch Gradient Descent
			- Some subset in each batch

		Batch Gradient Descent
			- All training data in each batch

		1 Step towards optimal = 1 epoch

		data1 = [1,2,3] data2 = [4,5,6]
		np.vstack((data1, data2)).T
		[[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]

		Logistic Regression
			- Fits an S curve (logic curve)
			- Given causes, predict probability of effects
			- Can be used for categorical y-variables

			Uses S curve
				p(yi) = 1/ (1 + e^-(A + Bxi))

				A is intercept
				B is regression coefficeint

				Curve flattens out at the extremes.

			Categorical variables with only 2 values => Binary variables

			By default, logistic regression assumes the intercept (A) is not included in the equation.
				Set that value in a new column 'intercept' in a dataframe

			Cost Function
				- Cross Entropy = -Sum(Yactual * log[Ypredicted])
					- Similarity of distribution
					- Checks if sets of numbers have been drawn from similar probability distributions or not.

			Softmax activation function required
				- Outputs probability of label being true or false
						p(y = True) = 1/ (1 + e^-(b + Wx'))
						p(y = True) = 1/ (1 + e^(b + Wx'))

						Use argmax(y_, 1) to set output to 1 hot format
							- The second input (1 in this case) tells what axis of the tensor (y_) to look at.
							- The index of the max arg is returned in one hot format.

						Check for accuracy
							correct = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
							# Cast turns True => 1 and False => 0
							accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

			N classification labels => N-1 neurons needed
				- With an M dimensional feature vector
					Shape(W) = [M,N]
					Shape(b) = [N]

		Linear vs. Logistic
			- Linear
				- Given causes, predict effect
				- Y must be continuous
				- Linear by assumption
			- Logistic
				- Given causes, predict probability of effect
				- Y must be categorical
				- Can be made linear (by log transformation)
					ln( p(yi)/(1 - p(yi)) ) = A + Bxi

					Odd(p) = p / 1 - p = e^(A+Bx)
						- Probability of p happening

					Logit(p) = A + Bx
						- ln(Odds(p))
						- Logistic regression can be solved by linear regression on the logit function

		Estimators
			- TensorFlow APIs for many standard problems
			- Reside in `tf.learn` and `tf.contrib.learn`
			- Estimators can be extended by plugging custom models into a base class.
			- That extension relies on composition rather than inheritance.

			How they work
				- Instantiate an estimator object
				- Specify list of all possible features
				- Create an input function which tells estimator how and what features it cares about.
					- Ex. tf.contrib.learn.io.numpy_input_fn
					- Input function also specifies the number of epochs and batch size

				Estimator object
					- Instantiate optimizer
					- Fetch data from feature vector
					- Define cost function
					- Run optimization

## Cloud ML

Minor Topics on Data Engineer Test, but major on Cloud Architect

	Predicting taxicab demand in NYC
		Inject a parameter into BQ and turn result into dataframe
			%bq query -n placeholderquery
			WITH trips AS (
				SELECT EXTRACT (DAYOFYEAR from pickup_datetime) AS daynumber FROM `nyc-tlc.green.trips_*` WHERE _TABLE_SUFFIX = @YEAR
			)
			SELECT daynumber, COUNT(1) AS numtrips FROM trips
			GROUP BY daynumber ORDER BY daynumber

			params = [
				{
					'name': 'YEAR',
					'parameterType': {'type':'STRING'},
					'parameterValue': {'value': 2015}
				}
			]
			trips = placeholderquery.execute(query_params=params).result().to_dataframe()

	Selecting what API to use
		from googleapiclient.discovery import build
		service = build('translate', 'v2', developerKey=APIKEY)

	Look at the features you can get from cloud ml.

# Compute Choices

## App Engine (PaaS)
	- Serverless and ops-free

	Environments
		- Standard
			- Pre-configured with: Java 7, Python 2.7, Go, PHP
			- Standard Container
			- Each runtime includes libraries that support App Engine Standard APIs
		- Flexible
			- More choices: Java 8, Python 3.x, .NET
			- Compute Engine VM (use dockerfiles to configure)

		Cloud Functions
			- Serverless execution environment for building and connecting cloud services.
			- Write simple, single purpose functions.
			- Attached to events emitted from your cloud infrastructure and services.
			- Cloud Function is triggered when an event being watched is fired.
			- Code executes in a fully managed environment.
			- No need to provision any infrastructure or worry about managing any servers.
			- Cloud Functions are written in JS.

## Compute Engine (IaaS)
	- Fully controllable down to OS

	- Google Cloud Launcher
		- LAMP stack or WordPress in a few minutes
	- Cost estimates before deployment

	- Can choose machine types, disk sizes before deployment
	- Can customize configuration, rename instances etc
	- After deployment, have full control of machine instances

	Creating an Instance
		- Name, Zone, Type of Machine, Disk Size, HTTP/HTTPS?

	Machine Types
		- Standard
		- High-memory
		- High-CPU
		- Shared core (small, non resource intensive)

		- Can attach GPU dies to most machine types

	Preemtible VM instance
		- A VM that Google can take back with just 30 seconds notice
		- Much cheaper than regular Compute Engine
		- Use for fault-tolerant applications
			- Processing only node in Hadoop Cluster
		- Terminated after 24 hours
		- Probability of termination varies by day/zone
		- Cannot live migrate (stay alive during updates) or auto-restart on maintenance

		- Step 1: Compute Sends Soft Off signal
		- Step 2: Hopefully, you have a shutdown script to clean up and give up control within 30s.
		- Step 3: If not, Compute sends Mechanical Off signal

	Projects and Instances
		- Each instance belongs to a project
		- Projects can have any number of instances
		- Projects can have up to 5 VPC (Virtual Private Networks)
			- Internal network separation inside of a project
		- Each instance belongs in one VPC
			- Instance within same VPC communicate on LAN
			- Instances across VPC communicate on internet

	Storage Options
		- Small root persistent disk containing the OS
		- Cloud Storage Buckets (Global Scope)
			- Cheapest
		- Persistent disks (Zone Scope)
			- Standard
			- SSD (solid state disks)
		- Local SSD (Instance Scope)
			- Most expensive
			- Does not offer Data redundancy
			- No custom encryption keys
			- Supports most machine types, not all like the rest
			- Data persists only till the instance is stopped or deleted.

		All have encryption at rest

	Can label resources to see usage patterns and billing for each group.


## Containers
	- Cluster of machines running Kubernetes and hosting containers
	- Balance between AE and CE

	Containers vs Virtual Machines
		- Each container does not abstract the operating system.
		- VMs have a Guest OS
		- VMs sit on top of (Virtual machine monitor/hypervisor)

		- Containers virtualize OS
		- VMs virtualize hardware

		- Container: more portable, quicker to boot, smaller in size

	Kubernetes
		- Has a Master which controls multiple Nodes (Kubernetes runs here)
		- Each Node has a Pod and a Kubelet
			- Machines in the same Node Pool all have same configuration
			- Node Pools useful for customizing instance profiles in your cluster.
			- Can run multiple Kubernetes node versions on each node pool in cluster, update each node pool independently, and target different node pools for specific deployments.
		- Each Pod has multiple containers

		- DevOps - need largely mitigated
		- Can use Jenkins for CI/CD if you want.

	GKE
		Storage options same as with Compute
			- Container Disks are emphemeral
				- If a container restarts after a crash, the data is lost.
				- Can use gcePersistentDisk abstraction to circumvent this.
		Load Balancing
			- Can make use of Network load Balancing (out of the box)
			- For HTTP load balancing, need to integrate with Compute Engine
		Container Builder
			- Tool that executes container image builds on GCP's infrastructure.
			- Working:
				- Import source code from a variety of repositories of cloud storage spaces
				- Executes a build to your specifications.
		Autoscaling
			- Automatic resizing of clusters with Cluster Autoscaler
			- Periodically checks whether there are any pods waiting, resizes cluster if needed.
			- Also monitors usage of nodes and deletes nodes if all its pods can be scheduled elsewhere.

		Setting up GKE instance
			`gcloud config set compute/zone us-central1-a`
			`gcloud container clusters create my-first-cluster --num-nodes 1`
			`gcloud compute instances list` -- get list of compute instances
			`kubectl run wordpress --image=tatum/wordpress --port=80`
			`kubectl get pods` -- gets details about pods
			`kubectl expose pod wordpress-323424-3e2e2 --name=wordpress --type=LoadBalancer`
				- Exposes the pod as a service
				- Type means set up load balancer for the container.
					- Creates an external IP address that the pod can use to accept traffic.
			`kubectl describe services wordpress`
				- Use the name given above

## Contrasting Compute Options

	App Engine
		- Flexible, zero ops platform for building highly available apps.
		- Focus on writing code only.
		- Neither know or care about the OS running your code.

		- Web sites; Mobile app and gaming backends.
		- RESTful APIs
		- IoT apps

	Container
		- Logical infrastructure powered bu Kubernetes.
		- Deploy fast, separate app from the OS.
		- Don't have dependencies on a specific operating system.
		
		- Containerized workloads.
		- Cloud native distributed systems
		- Hybrid applications

	Compute
		- Virtual machines running on Google's global data center network
		- Need complete control over infrastructure and direct access to high-performance hardware such as GPUs and local SSDs.
		- Need to make OS-level changes, such as providing your own network or graphic drivers, to squeeze out the last drop of performance.
		
		- Any workload requiring a specific OS or OS configuration.
		- Currently deployed, on-premises software that you want to run in the cloud.
		- Can't be containerised easily; or need existing VM images.

# Cloud Use-Cases
	- Hosting a website (Basic)
	- Running a Hadoop Cluster (Big Data)
	- Serving a TensorFlow model (Machine Learning)

# Logging and Monitoring
	Stackdriver

# Security, Networking
	API keys
	Load balancing

# Dataproc (~ Managed Hadoop)

	Managed Hadoop + Spark
		Hadoop, Spark, Hive and Pig

	"No-ops": Create cluster, use it, turn it off
		- Use GCS, not HDFS - else billing will hurt
			HDFS would require a VM instance
	
	Ideal for moving existing code to GCP

	Cluster Machine Types
		- Build using Compute Engine VM instances
		- Cluster - need at least 1 master and 2 workers
		- Preemptible instances - OK if used with care
			- Used for processing only - not data storage
			- No preemptible-only clusters - at least 2 non
			- Minimum persistent disk size - smaller of 100GB and primary worker boot disk size (for local caching - not part of HDFS)
	
	Initialization Actions
		- Can specify scripts to be run from GitHub or Cloud Storage
		- Can do so via GCM console, gcloud CLI or programatically
		- Run as root (i.e. no sudo required)
		- Use absolute paths
		- Use shebang line to indicate script interpreter

	Scaling Clusters
		- Can scale even when jobs are running
		- Operations for scaling are:
			- Add workers
			- Remove workers
			- Add HDFS storage
	
	High Availability
		- Will run 3 masters rather than 1
		- 3 masters will run in an Apache Zookeeper cluster for automatic failover

	Single Node Clusters
		- Just 1 node - master as well as worker
		- Can't be a preemptible VM
		- Use for prototyping/education

	Restartable Jobs
		- Jobs do NOT restart on failure (default setting)
		- Can change this - useful for long running and streaming jobs (eg Spark Streaming)
		- Mitigates out-of-memory errors, unscheduled reboots
	
	Connectors
		- BQ/BigTable/CloudStorage

	Networking
		- Firewall Rules
			- Name | Target (can be a tag) | Source Filter (whitelisted IPs)

			Direction of traffic
				- Ingress
			
			Action on match
				- Allow (whitelist)
			
			Specify protocols and ports (more restrictive)
				- tcp:8088 (cluster manager)
				- tcp:50070 (connect to HDFS name node)
				- tcp: 8080

			Paste in Browser
				`<master node ip>:8088`
				`<master node ip>:50070`

	Pyspark (SSH into Master Node)
		```
			pyspark
			data = [0,1,2,3,4]
			distData = sc.parallelize(data)
			squares = distData.map(lambda x: x*x)
			res = squares.reduce(lambda a, b: a + b)
			print res # RDDs evaluate here
		```
	
	hadoop fs -mkdir /dir
	hadoop fs -ls /
	# Copies from local instance machine to HDFS
	hadoop fs -put file.txt /dir

	Running a pig command from a .pig file
		`pig < pig-details.pig`

	# Copies output of pig jobs to local machine
	hadoop fs -get /dir/part* .

	Submitting a Spark Jar
		Specify job type as Spark
		Specify Jar file location and main class that is on the file system of the cluster

	GCloud
		```
			# Submit a job
			gcloud dataproc jobs submit spark --cluster ${CLUSTERNAME} \
				--class org.apache.spark.examples.SparkPi \
				--jars file:///usr/lib/spark/examples/jars/spark-examples.jar -- 1000

			# View jobs
			gcloud dataproc jobs list --cluster ${CLUSTERNAME}

			# Connects to job and shows output logs
			gcloud dataproc jobs wait ${JOB_ID}
				- Can get job id from `jobs list`

			# Get metadata regarding cluster
			gcloud clusters describe ${CLUSTERNAME}

			# Expand cluster cheaply
			gcloud dataproc clusters update ${CLUSTERNAME} --num-preemptible-workers=2
				- they use same image as worker nodes on the cluster

			# SSH into Master
			gcloud compute ssh ${CLUSTERNAME}-m --zone=us-central1-c

			# Describe a specific node
			gcloud compute instances describe ${CLUSTERNAME}-m --zone us-central1-c
		```

# Required Context
	- Hadoop, Spark, MapReduce (granular level)
	- Hive, HBase, Pig (use cases and some implementation quirks)
		- Hotspotting
		- HBase is a Columnar database
		- Poor choice of row keys can impact performance
	- RDBMS, indexing, hashing

# Datalab (Jupyter)
	- Interactive code execution for notebooks
	- Basically, Jupyter or iPython
	- Packaged as a container running on a VM instance.
	- Need to connect from browser to Datalab container on GCP

	Notebooks
		- Can be in Cloud Storage Repository (git repository)

	Persistent Disk
		- Notebook can be cloned from Cloud Storage to VM persistent disk
		- This clone => workspace => add/remove/modify files
		- Notebooks autosave, but you need to commit.

	Kernel
		- Opening a notebook => backend kernel process manages session and variables
		- Each notebook has 1 python kernel
		- Kernels are single-threaded
		- Memory usage is heavy - execution is slow - pick machine type accordingly
	
	Connecting to Datalab
		- gcloud SDK
			- Running the Docker container in a Compute VM and connecting to it through a SSH tunnel
		- Cloud Shell
		- Local machine
			- Running Docker on your local machine

	APIs and Services
		- Enable Compute Engine API

	Getting Started
		gcloud components update
		gcloud components install datalab
		datalab create <vm instance name>
			- specify zone
		datalab connect <vm instance name>
		Click web preview on top right, change to port 8081

	Importing and Exporting Data
		/datalab/docs/tutorials/BigQuery/ Importing and Exporting.ipynb

	Charting API in Datalab
		/datalab/docs/tutorials/Data/ Interactive Charts...

		When to use annotation chart vs. line chart while visualizing data?
			- If you want to provide the ability to zoom in.

	datalab create --no-create-repository <name>
		- Don't want to commit our files to github

## VMs and Images

VMs

	Live Migrations
		
		- Keep VM instances running even during maintenance events
		- Migrates your instance to another host in the same zone WITHOUT rebooting VMs
			- Infrastructure maintenance and upgrades
			- Network and power grid maintenance
			- Failed hardware
			- Host and BIOS updates
			- Security changes etc...
		
		- VM gets a notification that it needs to be evicted
		- A new VM is selected for migration, the empty "target"
		- A connection is authenticated between the two

		Stages of Migration
			1. Pre-migration Brownout
				- VM executing on source when most of the state is sent from source to target
			2. Blackout
				- A brief moment when the VM is not running anywhere
			3. Post-migration Brownout
				- VM is on target, the source is present and might offer support (forwards packets from source to target VMs till networking is updated)

		Instance with GPUs cannot be live migrated

		Get a 60 minute notice before termination

		Instance with local SSDs attached can be live migrated

		Preemptible instances cannot be live migrated, they are always terminated.

	Machine Types and Billing
		Cloud Platform Free Tier
			- 1 f1-micro VM instance per mongth (US regions, excluding Northern Virginia)
			- 30GB of Standard persistent disk storage per month
			- 5 GB of snapshot storage per month
			- 1 GB egress from North America to other destinations per month (excluding Australia and China)

		Pre-defined or Custom
			- Use custom to specify VCPU and Memory

		Shared Core
			- Ideal for applications that do not require a lot of resources
			- Small, non-resource intensive applications
			- Bursting
				- f1-micro machine types offer `bursting` capabilities that allow instances to use additional physical CPU for short periods of time
				- Bursting happens `automatically` when needed
				- The instances will automatically take advantage of available CPU in bursts
				- Not permanent, only possible periodically
		
		High Memory Machines (n1-highmem-${number of vCPUs})
			- More memory per vCPU as compared with regular machines
			- Useful for tasks which require more memory as compared to processing
			- Regular Machines: 3.75 GB RAM per vCPU (core)
			- Hgih Memory Machines: 6.5 GB RAM per vCPU (core)

		High CPU Machines (n1-highcpu-${number of vCPUs})
			- Less memory per vCPU as compared with regular machines

		Custom Machine Types
			- If none of the predefined machine types fit your workloads, use a custom machine type
			- Save the cost of running on a machine which is more powerful than needed.
			- Billed according to the number of vCPUs and the amount of memory used (no fixed billing since no standard configs)

		Sustained use and Committed use discounts
			- Save more if you use it longer.
			- Can purchase committed use contract for a larger discount.
				- Billed for each month of the contract regardless of whether billing occurs or not.

			Sustained Use
				- Discounts for running a VM instance for a significant portion of the billing month
				- Usage Levels (% of month)
					0 - 25% -> 100% base rate
					25 - 50% -> 80% base rate
					50 - 75% -> 60% base rate
					75 - 100% -> 40% base rate

			Sustained Discounts for Custom Machines
				- Calculates sustained use discounts by combining memory and CPU usage separately
				- Tries to combine resources to qualify for the biggest sustained usage discounts possible
				- Ex:
					Custom 2 vCPU and 4GB memory + Custom 4 vCPU and 6 GB memory

		Inferred Instances
			- Compute engine gives you the maximum available discounts by clubbing instance usage together.
			- Different instances running the same predefined machine type are combined to create inferred instances

		Billing Model
			- All machine types are charged for a minimum of 1 minute
			- After 1 minute instances are charged in 1 second increments

	Rightsizing Recommendations
		- Allows you to use the right sized machine for your workloads
		- Automatically generated based on system metrics gathered by Stackdriver monitoring
		- Uses last 8 days of data for recommendations

	RAM Disk
		- Allocate high performance memory to use as a disk
		- A RAM disk has very low latency and high performance
		- Used when application expects a file system structure and can't store data in memory
			- Writing files to disk
		- No storage redundancy or flexibility
		- SHARES meory with you applications
		- Contents stays only as long as the VM is up

Images

	An image in Compute Engine is a cloud resource that provides a reference to an immutable disk.

	- Used to create boot disks for VM instances
	- Public Images:
		- Provided and maintained by Google, open source communities, third party vendors
		- All projects have access and can use them.
	- Custom Images:
		- Available only to your project.
		- Create from boot disks and other images

	- Most public images can be used for no cost
	- Some premium images may have additional cost
		- Ex. Image of OS that needs a license in real world
	- Custom images that you import add no cost to your instance
	
	- They incur an image storage charge when stored in your project (tar and gzipped file in cloud storage)

	- Images are configured as part of the instane template of a managed instance group.
		- Managed instance groups can autoscale

	Image Contents
		- Boot loader
		- OS
		- File system structure
		- Software
		- Customizations

	- Master boot record and bootable partition are required for the image to be bootable.
	- For a disk to be imported as a compute engine image, the disk bytes must be written to a `disk.raw` file
	- Then the disk.raw is tar'd and compressed using gzip
	- Upload to Cloud Storage
	- Register it with Compute Engine
	- Can use it to create exact replicas of the original disk in any region of the platform (persistent disk)
	- Common use case: Have the images act as boot volumes for compute instances

	Premium Images
		- Additional per second charges, same across world
		- Changes based on the machine type used.
		- SQL Server images are charged per minute.

	Startup Scripts
		- Used to customize the instance created using a public image.
		- The script runs commands that deploys the application as it boots.
		- Should be idempotent to avoid inconsistent or partially configured state

	Baking
		- A more efficeint way to provision infrastructure.
		- Create a custom image with your configuration incorporated into the public image.

	Baking vs. Startup Scripts
		- Baking is faster to go from boot to application readiness.
		- Bakeing is much more reliable for deployments.
		- Version management is easier, easier to rollback
			- Startup scripts: Rollback has to be handled for applications and image separately
		- Fewer external dependencies during application bootstrap
		- Scaling up creates instances with identical software versions

	Image Lifecycle
		- DEPRECATED
			- No longer live version, but can still be launched
		- OBSOLETE
			- Should not be launched by users or automation.
			- Can use this image state to archive images so their data is still available when mounted to a non-boot disk.
		- DELETED
			- Have already been deleted or marked for deletion in future.
			- Cannot be launched and should be deleted as soon as possible.

	Images can be used across multiple projects
		- User can upload to cloud storage.
		7:20 in lecture 161

## VPCs and Interconnecting Networks

	Resources in GCP projects are split across VPCs

	Routes and forwarding rules must be configured to allow traffic within a VPC and with the outside world

	Traffic flows only after firewall rules are configured specifying what traffic is allowed or not

	VPN, peering, shared VPCs are some of the ways to connect VPCs or a VPC with an on premise network

	VPC (Virtual Private Cloud)
		- A global private isolated virtual network partition that provides managed networking functionality
		- 5 VPCs per project
		- 7000 instances per VPC

		- Global
			- Instances can be in different regions/zones
		- Multi-tenancy
			- VPCs can be shared across GCP projects
				- Firewall rules and traffic routes can span across projects
				- Across different billing units
		- Private and Secure
			- IAM, firewall rules
		- Scalable
			- Add new VMs, containers to the network

	- Since it is virual, the location of the machines in the network does not matter.

	Subnets
		- Logical partitioning of the network
			- Defined by a IP address prefix range
			- Specified in CIDR notation
				- 10.123.9.0/24
					- 24 is network prefix
						- first 24 bits are associated with the network that the subnet belongs to
					- Contains all IP addresses in the range
						- 10.123.9.0 - 10.123.9.255
					- Every subnet has a contiguous private RFC1918 IP space
			- IP range cannot overlap between subnets
			- Regional
				- Instances can be in different zones.
		- Subnets in the same VPC can communicate via Internal IP addresses
			- For different VPCs to communicate, they must use Internet	
				- Not private IP addresses like subnets

	GCP networks are a collection of subnets which have their own IP ranges.
		- Rather than the traditional approach of a network being allocated a range of IPs which gets split amongst the subnets.
		- Each dept. in a company can have their own subnets (logical groupings)

	Types of VPC Networks
		Auto Mode
			- Automatically sets up a single subnet in each region - can manually create more subnets
		Custom Mode
			- No subnets are set up by default, we have to manually configure all subnets
	
	Default Network
		- Created when intitializing a project
		- Auto Mode network
		- Comes with a number of routes and firewall rules preconfigured

	IP Addresses
		- Can be assigned to resources. Ex. VMs
		- Each VM has an internal IP address
		- One or more secondary IP addresses
		- Can also have an external IP

		Internal IP
			- Use within VPC
			- Cannot be used across VPCs unless we have special configuration (like shared VPCs or VPNs)
			- Can be ephemeral or static, typically ephemeral
				- Okay with address changing during reboot, or every 24hrs etc.
			- VMs know their internal IP address (VM name and IP is availbale to the network DNS)
				"instance-1.c.test-project123.internal"
			- VPC networks automatically resolve internal IP addresses to host names

		External IP
			- Use to communicate across VPCs
			- Traffic using external IP addresses can cause additional billing charges
			- Can be ephemeral or static
				- Static: Reserved - charged when not assigned to VM
			- VMs are not aware of their external IP address
			- Need to publish public DNS records to point to the instance with the external IP
			- Can use Cloud DNS

		Ephemeral
			- Available only till the VM is stopped, restarted or terminated
			- No distinction between regional and global IP addresses
			- Can turn ephemeral into static

		Static
			- Permanently assigned to a project and available till explicitely detached
			- Regional or global resources
				- Regional
					- Allows resources of the region to use the address
				- Global
					- Used only for global forwarding rules in global load balancing
			- Unassigned static IPs incur a cost
				- Once assigned, free to use
	
		Alias IP Ranges
			- A single service on a VM requires just one IP address
			- Multiple services on the same VM may need different IP addresses
			- Subnets have a primary and secondary CIDR range
				- Primary IP of VM drawn from primary
				- Containers/services in the VM can use secondary range
			- Using IP aliasing can set up multiple IP addresses drawn from the primary or secondary CIDR ranges
			- VPCs automatically set up routes for the IPs
			- Containers don't need their own routing, simplifies traffic management
			- Can separate infrastructure from containers (infra will draw from primary, containers from secondary)

	Working With Static Addresses
		- Go to VPC network
		- Go to external IP addresses
		- Reserve static address
			- Give a name
			- Select region
			- IP version (IPv4 or IPv6)
		- Create a VM instance
			- Networking
				- Specify external IP to address created previously
	
	Routes
		- A route is a mapping of an IP range to a destination. Routes tell the VPC network where to send packets destined for a particular IP address.

		2 Default Routes for a Network
			- Direct packets to destinations to specify destinations which carry it to the outside world (uses external IP addresses)
			- Allow instances on a VPC to send packets directly to each other (uses internal IP addresses)

			The existence of a route does not mean that a packet will get to the destination
				- Firewall rules have to be configured

		Creating a Network
			- Default route for internet traffic
			- One route for every subnet that is created
				- Ensures traffic from rest of network can reach that subnet

		What is a route made of?
			- name: User friendly name
			- network: Name of the network it applies to
			- destRange: The destination IP range that this route applies to
			- instanceTags: Instance tages that this route applies to, applies to all instances if empty
				- Can use IF you want route to apply to one specific instance
			- priority: Used to break ties in case of multiple matches
			AND one of
				- nextHopInstance: Fully qualified URL. Instance must already exist
				- nextHopIP: The IP address
				- nextHopNetwork: URL of network
				- nextHopGateway: URL of gateway
				- nextHopVpnTunnel: URL of VPN tunnel
			
		Instance Routing Tables
			- Every route in a VPC might map to 0 or more instances
			- Routes apply to an instance if the tag of the route and insance match
			- If no tag, then the route applies to all instances in a network
			- All routes together form a routes collection

		Using Routes
			- Many-to-one NATs (network address translations)
				- Multiple hosts mapped to one public IP
			- Transparent proxies
				- Direct all external traffic to one machine

	Firewall Rules
		- Allow or deny specific connections based on a combination of IP addresses, ports, and protocol
		- Rules exist between instances in the same network

		- Action: Allow or Deny
		- Direction: Ingress or Egress
		- Source IPs (ingress), Destination IPs (egress)
		- Protocol and Port
		- Specific instance names
			- Subset of machines
			- Can specify instance tags and service acct as well.
		- Priorities and tiebreakers

		GCP firewall rules are stateful
			- If a connection is allowed, all traffic in the flow is allowed, in both directions.
		
		Rule Assignment
			- Every rule is assigned to every instance in a network
			- Rule assignment can be restricted using tags or service accounts
				- Allow traffic from instances with source tag "backend"
				- Deny traffic to instances running as service account "blah@appspot.gcpserviceaccount.com"
		
		Service Accounts
			- Represents the identity that the instance runs with (use these if possible for firewall)
			- An instance can have just one service account
			- Restricted by IAM permissions, permissions to start an instance with a service account has to be explicitely given
			- Changing a service account requires stopping and restarting an instance
		
		Tags
			- Logically group resources for billing or applying firewalls
			- An instance can have any number of tags
			- Tags can be changed by any user who can edit an instance
			- Changing tags is metadata update and is a much lighter operation
			
		- Only IPv4 addresses are supported in a firewall rule
		- Firewall rules are specific to a network. They cannot be shared between networks (shared VPC this works tho)
		- Tags and service accounts cannot be used together in the same firewall rule

		Implied Rules
			- A default "allow egress" rule
				- Allows all egress connections.
				- Rule has a priority of 65535 (lowest priority)
			- A default "deny ingress" rule
				- Deny all ingress connection.
				- Rule has a priority of 65535 (lowest priority)
		
		Firewall Rules for "Default" Network
			- default-allow-internal
				- Allows ingress network connections of any protocol and port between VM instances on the network
			- default-allow-ssh
				- Allows ingress TCP connections from any source to any instance on the network over port 22
			- default-allow-icmp
				- Allows ingress ICMP (internet control message protocol) traffic from any source to any instance on the network
					- Used for error reporting
				- icmp is used with ping
			- default-allow-rdp
				- Allows ingress remote desktop protocol traffic to TCP port 3389

		Egress Connections
			- Destination CIDR ranges, protocols, ports
			- Destinations with specific tags or service accounts
				- Allow: Permit matching egress
				- Deny: Block matching egress

		Ingress Connections
			- Source CIDR ranges, protocols, ports
			- Sources with specific tags or service accounts
				- Allow: Permit matching ingress
				- Deny: Block matching ingress

	Cannot ping instance in other networks with internal reference (DNS scoped to network)
	Can however ping it with external ip

	sudo apt-get install traceroute
	sudo traceroute <instance name> -I
		- Use this to determine where pings fail

	Why convert automode to custom mode?
		- No option to select subnets. Can only delete entire network.

	Expand a Subnet (Ex. /26 -> /23)
		`gcloud compute networks subnet \
		expand-ip-range <subnet name> \
		--prefix length 23 \
		--region us-east1`

	Connecting via Cloud Shell SSH uses a google IP, so restricting access to a VM to only your actual IP will not work.

	Typically need external IP to SSH into VM.
		- Can use a VPN that connects from on prem network to the VPC
			gcloud compute ssh [INTERNAL_INSTANCE_NAME] --internal-ip
		- Or use a bastion host

	Bastion Host
		- Maintenance server

		# Create simple web app (webserver)
		sudo apt-get update
		sudo apt-get install apache2 -y
		echo '<!doctype html><html><body><h1>Hi</h1></body></html> | tee /var/www/html/index.html
		# Remove external IP

		Create an instance
			- SSH into host
			- ssh -a webserver
			# Can be used to connect to instances without external ip
			# Allows custom maintenance on machine

	3 Interconnection Options

		- Virtual Private Network using Cloud Router
			- Connects your on premise network to the Google Cloud VPC
			- Only IPsec gateway to gateway scenarios are supported, does not work with client software on a laptop
				- Peer gateway must have static external IP address (peer gateway to the on prem network)
					- If behind firewall, allow IKE traffic
				- CIDR range of on prem network should not conflict with cloud vpc
			- Offers 99.9% service availability
			- Traffic is encrypted by one VPN gateway and then decrypted by another VPN gateway
				- Uses IKE version 1 or 2
			- Supports both static and dynamic routes for traffic between on premise and cloud
			- Can have multiple tunnels to a single VPN gateway, site to site VPN
				- The same cloud vpn gateway can connect to multiple on prem networks
			- VPN traffic has to traverse internet
				- Higher latency and Lower throughput as compared to dedicated interconnect and peering options

			```
				# Create a VPN gateway
				gcloud compute target-vpn-gateways \
					create vpn-1 \
					--network vpn-network-1 \
					--region asia-east1
				# Link static IP address to that gateway
				gcloud compute addresses create --region asia-east1 vpn-1-static-ip
				# grab the new ip
				gcloud compute addresses list
				gcloud compute \
					forwarding-rules create vpn-1-esp \
					--region asia-east1 \
					--ip-protocol ESP \
					--address <address that was created> \
					--target-vpn-gateway vpn-1
				gcloud compute \
					forwarding-rules create vpn-1-udp500 \
					--region asia-east1 \
					--ip-protocol UDP \
					--ports 500 \
					--address <address that was created> \
					--target-vpn-gateway vpn-1
				gcloud compute \
					forwarding-rules create vpn-1-udp4500 \
					--region asia-east1 \
					--ip-protocol UDP \
					--ports 4500 \
					--address <address that was created> \
					--target-vpn-gateway vpn-1
				gcloud compute target-vpn-gateways list

				# Create a tunnel
				gcloud compute \
					vpn-tunnels create tunnel1to2
						--peer-address <address for second gateway (vpn-2)> \
						--region asia-east1 \
						--ike-version 2 \
						--shared-secret <secret> \
						--target-vpn-gateway vpn-1 \
						--local-traffic-selector 0.0.0.0/0 \
						--remote-traffic-selector 0.0.0.0/
				gcloud compute \
					vpn-tunnels create tunnel2to1
						--peer-address <address for first gateway (vpn-1)> \
						--region asia-south1 \
						--ike-version 2 \
						--shared-secret <secret> \
						--target-vpn-gateway vpn-2 \
						--local-traffic-selector 0.0.0.0/0 \
						--remote-traffic-selector 0.0.0.0/
				gcloud compute vpn-tunnels list

				# Create static routes
				gcloud compute \
					routes create route1to2 \
					--network  vpn-network-1 \
					--next-hop-vpn-tunnel tunnel1to2 \
					--next-hop-vpn-tunnel-region asia-east1
					--destination-range 10.1.3.0/24 # address range of vpc-network-2
				gcloud compute \
					routes create route2to1 \
					--network  vpn-network-2 \
					--next-hop-vpn-tunnel tunnel2to1 \
					--next-hop-vpn-tunnel-region asia-south1
					--destination-range 10.5.4.0/24
				
				# Can now go into the VMs and ping across VPCs
			```
			- Need to configure Cloud Router or Static Routes

			Cloud Router
				- Dynamically exchange routes between Google VPCs and on premise networks
				- Fully distributed and managed Google cloud service
				- Peers with on premise gateway or router to exchange route information
				- Uses the BGP or Border Gateway Protocol

			Static Routes
				- Create and maintain a routing table
				- A topology change in the network requires routes to be manually updated
				- Cannot re-route traffic automatically if link fails
				- Suitable for small networks with stable topologies
				- Routers do not advertise routes
				- If new rack in peer network
					- New routes need to be added to the cloud VPC to reach the new subnet
					- VPN tunnel will need to be torn down and re-established to include the new subnet
					- Static routes are slow to converge as updates are manual
			
			Dynamic Routes
				- Can be implemented using Cloud Router on GCP
				- Uses BGP to exchange route information between networks
				- Networks automatically and rapidly discover changes
				- Changes implemented without disrupting traffic
				- Ex.
					- A Cloud Router belongs to a particular network and a particular region.
					- The IP address of the Cloud Router and the gateway router should both be link local IP addresses (valid only for communication within the network link)
				
				Dynamic Routing Mode
					- Determines which subnets are visible to Cloud Routers
					- Global Dynamic Routing
						- Cloud Router advertises all subnets in the VPC network to the on-premise router
					- Regional Dynamic Routing
						- Advertises and propogates only those routes in its local region.
		
		- Dedicated Interconnect
			- Direct physical connection and RFC 1918 communication between on-premise network and cloud VPC
			- Can transfer large amounts of data between networks
				- Good for migrations to the cloud
			- More cost effective than using high bandwidth internet connections or using VPN tunnels

		- Direct and Carrier Peering

		