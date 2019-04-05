Hadoop

	Why cloud computing?
		- Data is getting bigger
		- Interconnections between data becoming more important.

		Nothing fits in-memory on a single machine

		Either have a super computer or a network of regular ones

		Distributed
			- Lots of cheap hardware
				- HDFS (distributed filesystem)
			- Replication and Fault Tolerance are important
				- YARN
			- Distributed computing
				- MapReduce

			- Each of which have corresponding configuration files

		Clusters consisting of Nodes live inside Server Farms
			- All of these servers need to be coordinated by a single piece of software
				- Need to partition data
				- Coordinate computing tasks
				- Handle fault tolerance and recovery
				- Allocate capacity to processes

		Coordinating between Hadoop Blocks
			- User defines map and reduce tasks using MapReduce API
			- A job is triggered on the cluster
			- Yarn figures out where and how to run the job, and stored the result in HDFS

HDFS

	Hadoop on GCP uses GCS. Why doesn't it use HDFS?
		- HDFS has a server which you would have to pay for with a VM on compute.

	- Built on commodity hardware
	- Highly fault tolerant, hardware failure is the norm
	- Suited to batch processing
		- Data access has high throughput rather than low latency
	- Supports very large datasets

	1 node is master node (Name Node)
		- Knows where the data is.
		- Manages the overall file system
		- Stores
			- The directory structure
			- Metadata of the files
	The other ones are Data Nodes
		- Physically stores the data in the files

	Storing a File in HDFS
		- Break the data into blocks of equal size
			- Different lenght files are treated the same way
			- Storage is simplified
			- Unit for replication and fault tolerance
		- Blocks are of size 128 MB
			- Larger -> Reduces parallelism
			- Smaller -> Increases overhead (more metadata)
		- Stores the blocks across the data nodes.
			- Each node contains a partition or a split of data.
			- How do we know where the splits of a particular file are?
				- Name Node (File 1 | Block 1 | Data Node)

	Reading a File
		- Use metadata in the name node to look up block locations
		- Read the blocks from respective locations

	For high availability applications
		- Can have multiple name nodes
		- Kept in sync using Zookeeper

	Failure Management
		Data Node
			- Define a replication factor
				- A given block of data will be stored in more than one data node.
				- Name node needs to store the replica locations as well.

	Default Replication Strategy
		Maximize Redundancy
			- First location chosen at random
			- Second has to be on a different rack (if possible)
			- Third will be on same rack as the second, but on a different node
				- Reduces inter-rack traffic and improves write performance
			- Read operations are sent to the rack closest to the client
		Minimize Write Bandwidth
			- Data is forwarded from first data node to the next replica location.
			- Forwarded further to the next replica location
			- Forwarding requires a large amount of bandwidth
				- Increases the costs of writes

MapReduce

	Why is there a strong need for a SQL interface a top MapReduce?
		- Hive and BigQuery offer this.
		Large number of data analysts that understand SQL but not Java code.

	Programmer defines 2 functions and Hadoop does the rest.

	Map
		- An operation performed in parallel, on small portions of dataset.
		- Outputs KV pairs

	Reduce
		- Mapper outputs become one final output.

	1. What {key, value} pairs should be emitted in the map step?
	2. How should values with the same key be combined?

	Params of both Map and Reduce
		Input key type
		Input value type
		Output key type
		Output value type

YARN (Yet Another Resource Negotiator)
	
	Why is managed hadoop such a great convenience? (Dataproc)
		- No admin, can turn on and off clusters, don't worry about data loss either.

	Coordinates tasks running on the cluster
	Assigns new nodes in case of failure

	Resource Manager
		- Runs on a single master node
		- Schedules tasks across nodes

	Node Manager
		- Run on all other nodes
		- Manages tasks on the individual node
		
		- All processes on a node ar run within a container and managed by Node Manager.
		- A container executes a specific application
		- One Node Manager can have multiple containers

		The Resource Manager starts off the Application Master within the Container.
			- Performs the computation required for the task.
			- If additional resources are required, the Application Master makes the request.

		Node Manager can request containers for mappers and reducers
			- This includes CPU and memory requirements

			Constraints when Requesting Help
				- Location
					- i.e. assign a process to the same node where the data to be processed lives.
						- if CPU, memory not available - WAIT!

		Scheduling Policies
			FIFO Scheduler
				- Queue
			Capacity Scheduler
				- Priority queue
			Fair Scheduler
				- Jobs assigned equal share of all resources


HBase
	- A database management system on top of Hadoop.
	- Integrates with your application just like a traditional database.
	- BigTable

	Columnar store
		Traditional (Id | To | Type | Content)
		Column (Id | Column | Value)

		Advantages
			Sparse tables
				- No wastage of space when storing data
			Dynamic attributes
				- Update attributes dynamically without changing storage structure
				- Do not need to change schema

	Denormalized storage
		Column names repeat across rows

		Normalization Reduces data duplication. Optimizes storage.
			- Storage is cheap in a distributed file system.
			- Optimize number of disk seeks instead (DENORMALIZE)

		Read a single record to get all details about an employee in one read operation (Denormalized).

	Only CRUD operations
		No comparisons/sorting/inequality checks across multiple rows
			No Joins
			No Group By
			No Order By

		NoSQL technology
			- Row is their basic unit of viewing the world.

		No operations involving multiple tables
		No indexes on tables
		No contraints

	ACID at the row level
		- Updates to a single row are atomic
			- All columns are updated, or none are
		- Updates to multiple rows are not atomic
			- Even if update is on the same column in multiple rows.

Hive (Analysis)
	- Provides a SQL interface to Hadoop.
	- Bridge to Hadoop for people without OOP exposure.

	- Not suitable for very low latency applications due to HDFS.

	Stores it's data in HDFS.

	HiveQL
		- Familiar to analysts.

	Wrapper on top of MapReduce.

	Metastore (?)
		- Bridge between HDFS and Hive
		- Stores metadata for all the tables in Hive
		- Maps the files and directories in Hive to tables
		- Holds table definitions and the schema for each table

		- Any database with a JDBC driver can be used as a metastore.
		- Development use built-in Derby database (embedded metastore)
			- Same Java process as Hive itselt
			- One Hive session to connect to the database

		Product Envs
			- Local metastores: Allows multiple sessions to connect to Hive
			- Remote metastore: Separate processes for Hive and the metastore


Hive vs. RDBMS

	Data Size
		Hive - Large datasets (giga or peta)
			Calculating trends
		RDBMS - Small datasets (mega or giga)
			Accessing and updating individual records

	Computation
		Hive - Parallel computations
			Semi-structured data files partitioned across machines
		RDBMS - Serial computations
			Structured data in tables on one machine
	
	Latency
		Hive - High latency
		RDBMS - Low latency
	
	Operations
		Hive - Read operations
		RDBMS - Read/write operations
	
	ACID compliance
		Hive - Not ACID compliant by default
		RDBMS - ACID compliant

	Query Language

HiveQL vs. SQL

	HQL
		High Latency
			Records not indexed - Cannot be accessed quickly

			Fetching a row will run a MapReduce that might take minutes

			Not the owner of the data
				They are in HDFS
				Hive files can be read and written by many technologies
					- Hadoop, Pig, Spark
				Hive database schema cannot be enforced on these files

			Schema-on-read
				Number of columns, column types, constraints specified at table creation.
				Hive tries to impose this schema when data is read
				It may not succeed, may pad data with nulls

		Not ACID compliant
			Data can be dumped into Hive tables from any source

		Row level updates, deletes as a special case
			- Not really supported

		Many more built in functions

		Only equi-joins allowed

	SQL

		Low Latency
			Records indexed, can be accessed and updated fast

			Queries can be answered in milliseconds or microseconds

			Sole gatekeeper for the data.

			Schema-on-write

		Acid compliant
			Only data which satisfies constraints are stored in the database


OLAP in Hive

	Why are windowing and partitioning so important in OLAP software?


	Partitioning
		State specific queries will run only on data in one directory.
			- i.e. Geographical data partitioned on states.
		Splits NOT of the same size

	Bucketing
		Size of each split should be the same
			- Hash of a column value
		Each bucket is a separate file
		Makes sampling and joining data more efficient
			- Greatly reduces the search space

	Join Optimizations

		Join operations are Map Reduce jobs under the hood
			- Optimize joins by reducing the amount of data held in memory

		Reducing data held in memory
			- On a join, one table is held in memory while the other is read from disk
				- Hold the smaller table in memory

		Structuring Joins as Map-Only Operation
			- Filter queries (only these rows)
				- Mapper needs to use null as key (?)

	Windowing Hive

		A suite of functions which are syntactic sugar for complex queries.
			- Make complex operations simple without needing many intermediate calculations

			Example:
				What revenue percentile did this supplier fall into this quarter?
					Window = one quarter
					Operation = Percentiles on revenue

Pig (ETL)
	- A data manipulation language
	- Transforms unstructured data into a structured format
	- Query this structured data using interfaces like Hive.

	Hive analyzes data, but where does the data come from??

	Characteristics of Data
		- Unknown schema
		- Incomplete data
		- Inconsistent records

	Apache Pig - High level scripting language to work with data with unknown or inconsistent schema.

	Used to get data into warehouse.

	Raw data -> Pig -> Warehouse -> HiveQL -> Analytics

	Pig Latin
		- A procedural, data flow language to extract, transform and load.
			Procedural
				- Uses a series of well-defined steps to perform operations.
				- No if statements or for loops.
			Data Flow
				- Focused on transformations applied to the data.
				- Written with a series of data operations in mind.
				- Nodes in a DAG

		- Data from one or more sources can be read, processed and stored in parallel.

		- Cleans data, precomputes common aggregates before storing in a data warehouse.

	Pig (Procedural) vs. SQL (Imperative)

		foreach						| select sum(revenue)
		(group revenues by dept)	| from revenues
		generate					| group by dept
		sum(revenue)				|

		Procedural
			- Specifies exactly how data is to be modifies at every step.

		Imperative
			- Abstracts away how queries are executed.

	Pig on Hadoop
		- Runs on top of Hadoop
		- Reads files from HDFS, stores intermediate records in HDFS and writes its final output to HDFS.
		- Decomposes operations into MapReduce jobs which run in parallel.
			- Provides non-trivial, built-in implementations of standard data operations, which are very efficient.
		- Pig optimizes operations before MapReduce jobs are run, to speed operations up.

	Pig on Apache Tez and Apache Spark
		Tez
			- Improves on MapReduce by making it faster.

Spark
	- A distributed computing engine used along with Hadoop
	- Interactive shell to quickly process datasets
	- Has a bunch of built in libraries for machine learning, stream processing, graph processing, etc...
	- Dataflow

	General Purpose
		Exploring
		Cleaning and Preparing
		Applying Machine Learning
		Building Data Applications

	Interactive
		Provides a REPL environment
			- Read-Evaluate-Print-Loop

	Reduces boilerplate of standard MapReduce Java code

	Resilient Distributed Datasets (RDDs)
		In memory collections of objects
		Can interact with billions of rows

		Properties
			Partitions
				- Distributed to multiple machines (nodes)
			Read-only
				- Immutable
				- Operations allowed on RDD
					- Transformations
						Transform into another RDD
					- Actions
						Request a result
			Aware of it's Lineage
				- When created, an RDD just holds metadata
					1. A transformation
					2. It's parent RDD
				- Implications of having Lineage
					- In built fault tolerance
						- If something goes wrong, reconstruct from source
					- Lazy Evaluation
						- Materialize only when necessary

	Lazy Evaluation
		- Spark keeps a record of the series of transformations requested by the user.
		- Waits until a result is requested before executing any of these transformations.

	Spark Core
		Basic functionality of Spark (RDDs)
		Written in Scala
			Run on JVM
			Has concept of Closures
		Runs on a Storage System & Cluster Manager
			These are plug and play components
			Can be HDFS and YARN

	How can RDDs be in-memory if they represent HDFS-scale data?
		- Files in HDFS can be petabytes in size
		- Lazy evaluation

		pyspark

		SparkContext (sc)
			Represents connection to Spark Cluster

		MLib

		Spark Streaming

		Hive:SQL :: Spark:Scala & Python

Streams Intro

	How can MapReduce be used to maintain a running summary of real-time data from sensors?
		- Send temp readings every 5 minutes

	Batches
		- Bounded datasets
		- Slow pipeline from data ingestion to analysis
		- Periodic updates as jobs complete
		- Order of data received unimportant
		- Single global state of the world at any point in time

	Streams
		- Unbounded datasets
		- Processing immediate, as data is received.
		- Continuous updates as jobs run constantly
		- Order important, out of order arrival tracked
		- No global state, only history of events received

		- Process the data one entity at a time or a collection of entities as a batch
			- Filter error messages (logs)
			- Find refrenece to latest movies (tweets)
			- Track weather patterns (sensor data)

		- Store, display, act on filtered messages
			- Trigger an alert
			- Show trending graphs
			- Warn of sudden squalls

	Stream-First Architechture
		- Data items can come from multiple sources
			- Files, Databases, but at least one from a Stream
		- All are aggregated and buffered in one way by a Message Transport (Queue)
			- i.e. Kafka
		- Passed to Stream Processing system
			- i.e Apache Flink or Spark Streaming

Microbatches

	Message Transport
		- Buffer for event data
		- Performant and persistent
		- Decoupling multiple sources from processing

	Stream Processing
		- High throughput, low latency
		- Fault tolerance with low overhead
		- Manage out of order events
		- Easy to use, maintainable
		- Replay streams

	A good approximation to stream processing is the use of micro-batches

	1. Group data items (time they were received)
	2. If small enough it approximates real-times stream processing.

	Advantaged of Microbatches
		- Exactly once semantics, replay micro-batches
			- items in stream only processed once
		- Latency-throughput trade-off based on batch sizes
			- Can adjust to your use case

	Spark Streaming, Storm Trident

Window Types

	Tumbling Window
		- Fixed window size (time window)
		- Non-overlapping time
		- Number of entities differ within a window

		Good for summations (number processing?)

	Sliding Window
		- Fixed window size
		- Overlapping time
		- Number of entities differ within a window

		Good for error spikes in logs
			- View errors occured in last N minutes

		Window Interval
			- How large the window is.

		Sliding Interval
			- By how much a window moves over.

	Session Window
		- Changing window size based on session data
		- No overlapping time
		- Number of entities differ within a window
		- Session gap determines window size


Oozie
	- A tool to schedule workflows on all the Hadoop ecosystem technologies

Kafka
	- Stream processing for unboudned datasets
	Apache Flink
	PubSub

Dataproc
	Hive
	Spark
	Pig
