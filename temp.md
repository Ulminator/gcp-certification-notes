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
		- VPN or Dedicated Interconnect or Direct and Carrier Peering

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
			- Capacity of a single connection is 10Gbps (also minimum deployment per location)
				- Maximum of 8 connections supported
			- No encryption (need in application layer)

			- Cross connect between the Google network and the on premise router in a common colocation facility	 (GOOGLE PEERING EDGE)
			- Can also use a Cloud Router too

			Benefits
				- Does not traverse the public internet.
					- Less hops, fewer points of failure
				- Can use internal IP addresses over a dedicated connection
					- Traffic that uses external IP is considered egress and may be billed higher.
				- Scale connection based on needs up to 80Gbps
				- Cost of egress traffic from VPC to on-premise network reduced

		- Direct and Carrier Peering
			Direct
				- Direct connection between on-premise network and Google at Google's `edge network locations`
				- BGP routes exchanges for dynamic routing
				- Can be used to reach all of Google's services including the full suite of GCP products
				- Special billing rate for GCP egress traffic, other traffic billed at standard GCP rates

			Carrier
				- Enterprise grade network services connecting your infrastructure to Google using a service provider
				- Can get higher availability and lower latency using one or more links
				- No Google SLA, the SLA depends on the carrier
				- Special billing rate for GCP egress traffic, other traffic billed at standard GCP rates
		
	Connecting VPC Networks
		- Shared VPC or VPC Network Peering
		- Shared VPC
			- XPN (Cross-Project Networking)
			- Multiple projects, 1 VPC
			- Creates a VPC network of RFC1918 IP spaces that associated projects can use
			- Firewall rules and policies apply to all projects on the network

			- User needs Compute Enginer | Compute Shared VPC Admin permissions

			Ex. 
				- Create a host project
					- Project that hosts sharable VPC networking resources within a Cloud Organization
				- Create multiple subnets that correspond to service projects that you want in the network
					- Service Project: Project that has permission to use the shared VPC networking resources from the host project
					- Each service project could correlate to departments in your organization. Each responsible for thier own billing

			Host and Service Projects
				- A service project can only be associated with a single host
				- A project cannot be a host as well as a service project at the same time
				- Instances in a project can only be assigned external IPs from the same project
				- Existing projects can join shared VPC networks
				- Instances on a shared VPC need to be created explicitely for the VPC
					- Instances cannot be migrated over

			Standard Use Case
				- Two Tier Web Service
					- A different team owns the Tier 1 and Tier 2 services
						- External clients go to external Load Balancer
						- This goes to Tier 1 (frontend) Service which then goes to Internal Load Balancer to Tier 2 (backend) services
					- Each team can deploy and operate its services independently
					- Each project billed separately
					- Each project admin can manage their own resources
					- A single group of network and security admins can be responsible for the shared VPC

		- VPC Network Peering
			- Allows private RFC 1918 connectivity across 2 VPC networks
			- Networks can be in the same or in different projects
			- Network peering can be broken unilaterally
			- IP addresses cannot conflict across all peering connections
			- Build SaaS ecosystems in GCP
				- Services can be made available privately across different VPC networks
			- Useful for organizations:
				- With several network administrative domains
				- Which want to peer with other organizations on the GCP

			Benefits
				- Lower latency as compared with public IP networking
				- Better security since services need not expose an external IP address
				- Using internal IPs for traffic avoids egress bandwidth pricing on the GCP

			Properties
				- Peered networks are administratively separate
					- Routes, firewalls, VPNs and traffic management applied independently
				- One VPC can peer with multiple networks with a limit of 25
					- Peering for 1 VPC network can be configured even before the other network is created.
				- Only directly peered networks can communicate with each other
					- If A peers with B and B peers with C, A cannot communicate with C using internal IPs
				- A load balancer in network A will apply automatically to network B (no additional configuration needed)
		
		- Peered Networks and Shared VPCs
			- A shared VPC with 1 host and 2 service projects
			- Host peers with another VPC
				- All VMs can communicate with each other via internal IP addresses
			- Can also have direct peering between 2 Shared VPCs

		- Peered Networks and Multiple Network Interface Cards (NICs)
			- VM has two network interfaces - one in network A (IP1) and one in Network B (IP2)
			- Network B and C are peered with each other.
				- Network A is standalone, C can only see IP2
			- IP1 on network A cannot see any instanes on Network B or C

	Cloud DNS
		- A high performance, resilient, global Domain Name System service that publishes your domain names to the global DNS in a cost-effective way.

		- Hierarchical distributed database that lets you store IP addresses and other data and look them up by name.
		- Publish zones and records in the DNS
		- No burden of managing your own DNS server

		Managed Zone
			- Entity that manages DNS records for a given suffix (example.com)
			- Maintained by Cloud DNS

		Record Types
			- A: Address record - Maps hostnames to IPv4 addresses
			- SOA: Start of Authority - Specifies authoritative information on a managed zone
			- MX: Mail Exchange used to route requests to mail servers
			- NS: Name Server record - Delegates a DNS zone to an authoritative server

		Resource Record Changes
			- The changes are first made to the authoritative servers and is then picked up by the DNS resolvers when their cache expires
	
	Legacy GCP Netowrks
		- Not recommended
		- Instance IP addresses are not grouped by region or zone
		- No subnets
		- Random and non-contiguous IP addresses
		- Only possible to create only through gcloud CLI and REST API

## Managed Instance Groups and Load Balancing

	Overview
		- Pool of similar machines which can be scaled automatically.
		- Load balancing can be external or internal, global or regional
		- Basic components of HTTP(S) load balancing - target proxy, URL map, backend service and backends
		- Use cases and architecture diagrams for all the load balancing types HTTP(S), SSL proxy, TCP proxy, network and internal load balancing

	Instance Groups
		- A group of machines which can be created and managed together to avoid controlling each instance in the project.

	Instance Template
		- Defines the machine type, image, zone and other properties of an instance. 
		- A way to save the instance configuration to use it later to create new instances or groups of instances.
		- Global resource not bound to a zone or a region.
		- Can reference zonal resources such as a persistent disk
			- In such cases can be used only within the zone.

	Managed Instance Groups
		- Uses an instance template to create a group of identical instances.
		- Changes to the instance group changes all instances in the group.
		- Can automatically scale the number of instances in the group.
		- Work with load balancing to distribute traffic across instances.
		- If an instance stops, crashes or is deleted the group automatically recreates the instance with the same template.
		- Can identify and recreate unhealthy instance in a group (autohealing)

		2 Categories
			Zonal
				- Choose if you want lower latency and avoid cross-zone communication.
			Regional
				- Prefer this so application load can be spread across multiple zones.
				- Protects against failures within a single zone.

		Health Checks and Autohealing
			- A MIG applies health checks to monitor the instances in the group.
			- If a service has fails on an instance, that instance is recreated (autohealing)
			- Similar to health checks used in load balancing but the objective is different
				- LB health checks are used to determine where to send traffic.
				- MIG health checks are used to recreate instances.
				- Typically configure health checks for both LB and MIGs
			- The new instance is recreated based on the template
			- Disk data might be lost unless explicitely snapshotted

		Configuring Health Checks
			- Check Interval
			- Timeout
			- Health Threshold
				- How many consecutive "healthy" responses indicate that the VM is healthy.
			- Unhealthy Threshold
				- How many consecutive "unhealthy" responses indicate that the VM is unhealthy.

	Unmanaged Instance Groups
		- Groups of dissimilar instances that you can add and remove from the group.
		- Do not offer autoscaling, rolling updates, or instance templates
		- Not recommended, used only when you need to apply `load balancing to pre-existing` configurations

	Load Balancing
		- Reserve an External Static IP for use with the Load Balancer
		- Load balancing and autoscaling for groups and instances
		- Scale your application to support heavy traffic
		- Detect and remove unhealthy VMs, healthy VMs automatically re-added
		- Route traffic to the closest VM
		- Fully managed service, redundant and highly available

		- Rule of thumb: Load balancer in the highest layer possible
			- Higher level encapsulates information at the lower levels

		External
			- Traffic from internet
			- Global or Regional
			Global
				- HTTP/HTTPS (application layer)
					- Cross-Regional or Content Based
					- Distributes traffic based on:
						- Proximity to user
						- Requested URL
						- or both
					- How it works:
						- Traffic from internet is sent to a global forwarding rule
							- This determines which proxy (HTTP in this case) the traffic should be directed to
						- Target proxy checks each request against a URL map to determine the appropriate backend service for the request
							- HTTPS requires the target proxy to have a signed certificate to terminate the SSL connection
						- Backend service directs each request to an appropriate backend based on serving capacity, zone, and instance health of its attached backends
							- Backend service also sets up Session Affinity
								- ALL requests from same client to same server based on either:
									- client IP
									- cookie 
						- The health check of each backend instance is verified using either an HTTP health check or an HTTPS health check - if HTTPS, request is encrypted
						- Actual request distribution to backend can happen based on CPU utilization or requests per instance
							- Can configure managed instance groups to scale as traffic slaces
						- MUST create firewall rules to allow requests from load balancer and health checker to get through to instances
				- SSL Proxy (session layer)
				- TCP Proxy (transport layer)
			Regional
				- Network (network layer)
				- These correspond to the OSI Network Stack

		Internal
			- Traffic from within network
			- Regional

		Health Checks
			- HTTP,HTTPS health checks
				- Highest fidelity check
					- Verify that web server is up and serving traffic, not just that the instance is healthy
			- SSL health check
				- Configure if your traffic is not HTTPS but is encrypted via SSL (TLS)
			- TCP health checks
				- For all TCP traffic that is not HTTP(S) or SSL(TLS)

		Gloabal Forwarding Rule
			- Route traffic by IP address, port, and protocol to a load balancing proxy
			- Can only be used with global load balancing HTTP(S), SSL Proxy, and TCP Proxy
			- Regional forwarding rules can be used with regional load balancing and individual instances

		Target Proxy
			- Referenced by one or more global forwarding rule
			- Route the incoming requests to a URL map to determine where they should be sent
				- TCP and SSL Proxies -> Routed directly to backend service (no url map)
			- Specific to a protocol (HTTP, HTTPS, SSL, and TCP)
			- Should have a SSL certificate if it terminates HTTPS connections (limit of 10 SSL cetificates)
			- Can connect to backend services via HTTP or HTTPS
				- For HTTPS, the VM instance also needs cert installed

		URL Map
			- Used to direct traffic to different instances based on incoming URL
				Ex.
					- http://example.com/audio -> backend service 1
					- http://example.com/video -> backend service 2
			- Default setting
				- Only /* path matcher is created automatically and directs all traffic to the same backend service
			- Typical settings
				- Host rules -> Path matcher -> Path rules
					- Host rules
						- example.com, customer.com
					- Path matcher and path rules
						- /video
					- Path rules
						- /video/hd, /video/sd
				- Traffic that does not match any of the rules are sent to default service

		Backend Service
			- Centralized service for managing backends
			- Contain one or more instance groups which handle user requests
			- Knows which instances it can use, how much traffic they can handle
			- Monitors the health of backends and does not send traffic to unhealthy instances

			Components
				- Health Check: Polls instances to determine which one can receive requests
					- HTTP(S), SSL and TCP
					- GCP creates redundant copies of the health checker automatically, so health checks might happen more frequently than you expect
				- Backends: Instance group of VMs which can be automatically scaled
					- Synonymous to Instance Group
					- Balancing Mode: Determines when the backend is at full usage.
						- CPU utilization, requests per second
					- Capacity Setting: A % of the balancing mode which determines the capacity of the backend.
					- Can also be Backend Buckets
						- Allow you to use Cloud Storage buckets with HTTP(S) load balancing
						- Traffic is directed to the bucket instead of a backend
						- Useful in load balancing requests to static content
						- A path of / static can be sent to the storage bucket and all other paths fo to the instances
				- Session Affinity: Attempts to send requests from the same client to the same VM
					- Client IP: Hashes the IP address to send requests from the same IP to the same VM
						- Requests from different users might look like its from the same IP
						- Users which move networks might lose affinity
					- Cookie: Issue a cookie named GCLB in the first request
						- Subsequent requests from clients with the cookie are sent to the same instance
				- Timeout: Time the backend service will wait fro a backend to respond

		Load Distribution
			- Uses CPU utilization of the backend or requests per second as the balancing mode.
			- Maximum values can be specified for both.
			- Short bursts of traffic above the limit can occur.

			- Incoming requests are first sent to the region closest to the user, if that region has capacity.
			- Traffic distributed amongst zone instances based on capacity.
			- Round robin distribution across instances in a zone.
			- Round robin can be overridden by session affinity.

		Firewall Rules
			- Allow traffic from 130.211.0.0/22 (load balancer) and 35.191.0.0/16 (health check) to reach your instances
			- Allow traffic on the port that the global forwarding rule has been configured to use.

		```
		# Create a static external route for the load balancer
		# Create a health check in compute/healthcheck section
		# Create target pool and connect it to the health check
		gcloud compute target-pools create <pool name> \
			--region us-central1
			--http-health-check <health check name>
		gcloud compute target-pools add-instances <pool name> \
			--instanes <instance 1>, <instance 2>, <instance 3> \
			--instances-zone=us-central1-a
		# Get the address of the static route created
		gcloud compute addresses list
		# Use it in forwarding rule
		gcloud compute forwarding-rules create <rule name> \
			--region us-central1 \ 
			--ports 80 \ 
			--address <previously created address>
			--target-pool <pool name>
		```

		Content Based Load Balancer
		```
		# Create VPC netowork
		# Create external ip for load balancer
		# Create two instance groups
		# Create 1 health checks for both instance groups
		# Create load balancer
		# Two backend services for this load balancer
		# Create path rules
		```

		SSL Proxy
			- Encrypted but not HTTPS
			- The usual combination is TCP/IP: network = IP, transpot = TCP, application = HTTP
			- For secure traffic: add session layer = SSL, and application layer = HTTPS

			- SSL Connections are terminated at the global layer, then proxied to the closest available instance group
				- Terminated means we need a certificate on this proxy
				- Makes fresh connections to backends - this can be SSL or non-SSL

				```
					# Create firewall rule on SSL load balancer for tcp:443
					# Create some VMs
					# Create an instance group
						# Port name mapping: name: ssl-lb number: 443
					# Create SSL Load Balancer
						# Creating a backend - use named port from above; protocol SSL
					# Create a certificate and apply to frontend of LB
				```
			
		TCP Proxy
			- Allows you to use a single IP address for all users around the world
			- Advantage of transport layer load balancing:
				- more intelligent routing possible than with network layer load balancing
				- better security - TCP vulnerabilities can be patched at the load balancer itself
		
		Network Load Balancer
			- Based on incoming IP protocol data, such as address, port, and protocol type
			- Pass-through, regional load balancer - does not proxy connections for clients
			- Use it to load balance UDP traffic, and TCP and SSL traffic
			- Load balances traffic on ports that are not supported by SSL and TCP proxy load balancers
				- SSL and TCP only support specified ports

			Load Balancing Algorithm
				- Picks an instance based on a hash of:
					- source IP and port
					- destination Ip and port
					- protocol
				- This means that incoming TCP connections are spread across instances and each new connection may go to a different instance.
				- Regardless of session affinity setting, all packets for a connection are directed to the chosen instance until the connection is closed and have no impact on load balancing decisions for new incoming connections.
				- This can result in imbalance between backends if long lived TCP connections are in use.
			
			Target Pools
				- Network load balancing forwads traffic to target pools
				- A group of instances which receive incoming traffic forwarding rules
				- Can only be used with forwarding rules for TCP and UDP traffic
				- Can have backup pools which will receive requests if the first pool is unhealthy
				- failoverRatio is the ratio of healthy instances to failed instances in a pool
				- If a primary target pool's ratio is below the failoverRatio traffic is sent to the backup pool

			Health Checks
				- Configure to check instance health in target pools
				- Network load balancing uses legacy health checks for determining instance health

			Firewall Rules
				- HTTP health check probes are sent from the IP ranges 209.85.152.0/22, 209.85.204.0/22, and 35.191.0.0/16
				- Load balancer uses the same ranges to connect to the instances
				- Firewall rules should be configured to allow traffic from these IP ranges

		Internal Load Balancing
			- Private load balancing IP address that only your VPC instances can access
				- All instances belong to the same VPC and region, but can be in different subnets
			- Less latency, more security
			- No public IP needed
			- Useful to balance requests from your frontend instances to your backend instances
			
			- Load balancing IP is from the same VPC network

			Load Balancing Algorithm
				- Backend instance for a client is selected using a hashing algorithm that takes instance health into consideration
				- Using a 5 tuple hash, five params for hashing:
					- Client source IP
					- Client Port
					- Destination IP (load balancing IP)
					- Destination Port
					- Protocol (TCP or UDP)

				- Introduce session affinity by hashing on only some of the 5 params
					- Hash based on 3-tuple (Client IP, Dest IP, Protocol)
					- 2-tuple (Client IP, Dest IP)
			
			GCP Internal Load Balancing
				- Not proxied - differs from traditional model
				- Lightweight load balancing built on top of Andromeda network virtualization stack
				- Provides software defined load balancing that directly delivers the traffic from the client instance to a backend instance

			Use Case: 3 Tier Web App
				- Clients connect to an external LB (HTTP(S))
				- Frontend instances are connected to the backend instances using an internal load balancer

	Autoscaling
		- Helps your applications gracefully handle increases in traffic
		- Reduces cost when load is lower
		- Feature of managed instance groups (not supported for unmanaged)

		- GKE has cluster autoscaling

		- Autoscaling Policy
			- Average CPU utilization
			- Stackdriver monitoring metrics
				- Built in or custom metrics
				- Not all standar metrics are valid
					- The metric must contain data for a single VM instance
					- Must define how busy the resource is.
			- HTTP(S) load balancing server capacity
				- CPU utilization
				- Maximum requests per second/instance
			- Pub/Sub queuing workload (alpha)
		- Target Utilization Level
			- The level at which you want to maintain your VMs
			- Interpreted differently based on the autoscaling policy that you've chosen.
			- If utilization reaches 100% during times of heavy usage the autoscaler might increase the number of CPUs by
				- 50% or 4 instances (whicher is larger)

		Autoscalers with Multiple Policies
			- Will scale based on the policy which provides the largest number of VMs in the group

		```
			# Create a VM
			# Install whatever you want to start on reboot
			sudo update-rc.d apache2 defaults
			# Delete VM but retain boot
			# Create custom image from boot
			# Use that image in an instance group definition.
		```

# Ops and Security

	StackDriver
		- Accounts
			- Holds monitoring and other configuration information for a group of GCP projects and AWS accounts that are monitored together.

		Types of Monitored Projects
			- Hosting Projects
				- Holds the monitoring configuration for the Stackdriver account - the dashboards, alert policies, uptime checks, and so on.
				- To monitor multiple GCP projects, create a new StackDriver account in an empty hosting project.
					- Don't use the hosting project for anything else.
			- AWS Connector Projects
				- When you add an AWS account to a StackDriver account, StackDriver Monitoring creates the AWS connector project for you, typically giving it a name beginning with AWS Link.
				- The Monitoring and Logging agents on your EC2 instances send their metrics and logs to this connector project.
				- If you use StackDriver logging from AWS, those logs will be in the AWS connector project (not in the host project of the Stackdriver account)
				- Don't put any GCP resources in an AWS connector project
					- Will not be monitored!
			- Monitored Project
				- Regular (non-AWS) projects within GCP that are being monitored.
			
		Metrics
			Built in:
				- CPU utilization of VM instances
				- Number of tables in SQL databases
				- Hundreds more...
			Custom
				- 3 Types
					- Gauge Metrics
					- Delta Metrics
					- Cumulative Metrics
			Metric Data available for 6 weeks.
		
		Metric Latency
			- VM CPU utilization
				- Once a minute, available with 3-4 minute lag
			- If writing data programatically to metric time series
				- First write takes a few minutes to show up
				- Subsequent writes visible within seconds
		
		Monitored Resources
			- VM instances
			- Amazon EC2 instances, RDS databases
			- Cloud Storage Buckets
			- Pretty much anything in GCP
		
		Alerts and Notification
			- Depends on service tier

		Error Reporting
			- App Engine
				- Stack trace and severity of ERROR or higher automatically show up.
			- App Engine Flexible Environment
				- Anything written to stderr automatically shows up
			- Compute Engine
				- Instrument - Throw error in exception catch block
					- Write to error stream using StackDriver client
			- Amazon EC2
				- Enable StackDriver logging
		
		Trace
			- Distributed tracing system that collects latency data from Google App Engine, Google HTTP(S) load balancers, and applications instrumented with the StackDriver Trace SDKs
			
			- How long does it take you application to handle incoming requests from users or other applications
			- How long it takes to complete operations like RPC calls performed when handling requests
			- Round trip RPC calls to App Engine services like Datastore, URL Fetch, and Memcache

		Logging
			- Includes a store for logs, a user interface (the Logs Viewer), and an API to manage logs programatically
			- Allows you to:
				- read and write log entries
				- search and filter
				- export your logs
				- create log-based metrics

			Types of Logs
				- Audit Logs
					- Permanent GCP logs (no retention period)
					- Admin Activity Logs
						- For actions that modify config or metadata
						- Always on
						- Not charged
					- Data Access Logs?
						- API calls that create modify or read user-provided data
						- Need to be enabled (can be big)
						- Always on for BigQuery
						- Charged
			
			Service Tiers and Retention
				- Basic -> No stackdriver account - free and 5GB cap
				- Retention period of log data depends on tier

			Using Logs
				- Exporting to sinks
					- Cloud Storage
					- BigQuery datasets
					- Pub/Sub topics
				- Create metrics to view in stackdriver monitoring

			```
			# Download stackdriver install script
			curl -O "https://repo.stackdriver.com/stack-install.sh"
			sudo bash stack-install.sh --write-gcm
			# Generate random data and force CPU to compress it (load strain)
			dd if=/dev/urandom | gzip -9 >> /dev/null &
			```

	# Replaces all instances of first with second in a file
	sed -i -e 's/first/'second/ file.txt
		
	Deployment Manager
		- Deployment is a collection of resources, deployed, and managed together.
		- Must enable Deployment Manager API

		Resource
			- Represents a single API resource and provided by Google-managed base types.
			- API resource provided by a Type Provider.
			- To specify a resource - provide a Type for that resource.

		Types
			- Represent a single API resource or a set of resources (composite type)
			- Base type - Creates single primitive resource and type provider used to create additional base types.
			- Composite base types - Containe one or more templates.
		
		Configuration
			- Describes all resources you want for a single deployment in a YAML
			- Resources must contain 3 components
				- Name: user defined string for id
				- Type: type of resource being deployed
				- Properties - params of the resource type
		
		Templates
			- Parts of the configuration and abstracted into individual building blocks. Written in python or jinja2
			- More flexible than individual configuration files and intended to support easy portability across deployments
			- The interpretation of each template eventually must result in the same YAML syntax

		Manifest
			- Read only object contians original configuration
			- At the time of updating Deployment Manager generates a manifest
		
		Runtime Configurator
			- Lets you define and store a hierarchy of KV pairs in GCP
			- Used to dynamically configure services, communicate service states, send notifications of changes to data and share information between multiple tiers of services
			- Offers watcher and waiter services
				watcher - return whenever KV pair changes
				waiter - pause until certain # of services are running

		```
			gcloud deployment-manager deployments create gcpingra --config=conf.yaml
			# Change config file and then update
			gcloud deployment-manager deployments update gcpingra --config=conf.yaml
			# see types of resources
			gcloud deployment-manager types list
		```

	Cloud Endpoints
		- Helps create, share, and secure your APIs
		- Uses the dostrobuted Extensible Service Proxy to provide low latency and high performance
		- Provides authentication, logging, monitoring
		- Host your API anywher Docker is supported as long as it has Internet access to GCP
			- But ideally use:
				- App Engine (flexible or some types of standard)
					- Endpoint service must be turned on for standard
				- Container Engine
				- Compute Engine
		- Is in a Docker container
		- Note - proxy and API containers must be on same instance to avoid network access

Identity and Security

	Authentication (Who are you?)
		Standard Flow
			- Service Accounts
				- Most flexible and widely supported authentication method
				- Different GCP APIs support diff credential types, but all support service accounts

				Why use?
					- Associated with project, not user

				- Create from: GCP Console or Programatically
				- Associated with credentials via GOOGLE_APPLICATION_CREDENTIALS
				- At any point, one set of credentials is 'active' (Application Default Credentials)

				- When your code uses a client library, the strategy checks for your credentials in the following order:
					- First ADC checks if GOOGLE_APPLICATION_CREDENTIALS is set. If so ADC uses the service account file that the variable points to.
					- If not, ADC uses default service account that Compute, Container, App Engine, or Cloud Functions provide

			- End-user accounts
				Why use?
					- If you'd like to differentiate between different end-users of the same project.
					- You need to access resources on behalf of an end user of your application. (Ex. BigQuery dataset that belongs to users)
					- You need to authenticate yourself (not as your application)
						- Ex. Cloud Resources Manager API can create and manage projects owned by a specific user.

		API Keys
		- API keys
			- Simple encrypted strings
			- Can be used when calling APIs that don't need to access private user data
			- Useful in clients such as browsers and mobile applications that don't have a backend server
			- API key is used to track API requests associated with your project for quota and billing
			- Can be used for ML APIs

			Beware
				- Can be used by anyone: MITM
				- Do not identify user or application (only project)

	OAuth 2.0
		- Application needs to access resources on behalf of a specific user
		- Application presents consent screen to user; user consents
		- Application requests credentials from some authorization server
		- Application then uses these credentials to access resources

		- Client ID Secrets are viewable by all project owners and editors, but not readers
		- If you revoke access to some user, remember to reset these secrets to prevent data exfiltration

		How to create?
			- GCP Console -> API Manager -> Credentials -> Create
			- Select "OAuth Client ID"
			- Can be used in application now.
		
		"Access API via GCP Project"
			- User wants to access some API
			- Project needs to access that API on behalf of user
			- Project requests GCP API Manager to authenticate user by passing client secret; API manager repsonds
			- Project has authenticated user, now gives API access

	Authorization (What can you do?)
		Identity and Access Management (IAM)
			- Identities
				- End-user (Google) account
				- Service account
				- Google group
				- G-Suite domain
				- Cloud Identity domain
				- allUsers, allAuthenticatedUsers
		- Roles
			- lots of granular roles
			- per resource

			Primitive Roles
				- Viewer
					- Read only
				- Editor
					- Modify and delete
					- Deploy
				- Owner
					- Create
					- Add/Remove members

			Custom Roles
				- Add permissions to role a la carte
				- Allow more granular access
				- Can only be used for with Project or Organization - not with folders
				- Project level custom roles only apply to resources in same project

		- Resources
			- Projects
			- Compute Engine instances
			- Cloud Storage buckets
			- Vrtual Networks
			- etc...

			Resource Hierarchy
				- Organization >> Project >> Resource
				- Can set an IAM access control policy at any level in the resource hierarchy
				- Resources inherit the policies of the parent resource
				- If a child policy is in conflict with the parent, the less restrictive policy wins

				Organization
					- Not required, but helps separate projects from individual users
					- Link with G-suite super admin
					- Root of hierarchy, rules cascade down
				
				Folders
					- Logical grouping of projects
					- May represent departments, legal entities or teams
					- Can have folder within folders

				G Suite Features
					- Allows an Organization resource
					- Administer users and groups
					- Can set up Google Cloud Directory Sync to synchronize G Suite accounts with LDAP/AD
					- SSO integration with third party identity providers

		Identity Aware Proxy (IAP)
			- An HTTPS based way to combine all of this.
			- Acts as an additional safeguard on a particular resource
			- Turning on IAP for a resource causes creation of an OAuth 2.0 Client ID and secret (per resource)
				- Don't go in and delete any of these! IAP will stop working

			- Central Authorization layer for applications accessed by HTTPS
			- Application level access control model instead of relying on network level firewalls
			- With Cloud IAP, you can set up group-based application access:
				- A resource could be accessible for employees and inaccessible for contractors, or only accessible to a specific department.

			- IAP is an additional step, not bypassing of IAM
				- Users and groups still need correct Cloud IAM role

			Work Flow
				- Requests come from 2 sources:
					- App Engine
					- Cloud Load Balancing (HTTPS)
				- Cloud IAP checks the user's browsers credentials
				- If none exist, the user is redirected to an OAuth 2.0 Google Account sign in
				- Those credentials sent to IAM for authorization

			IAP Limitations
				- Will not protect against activity inside VM, someone SSH-ing into a VM or App Engine flexible environment
				- Need to configure firewall and load balancer to disallow traffic not from serving infrastructure
				- Need to turn on HTTP signed headers

	Best Practices
		Resource Hierarchy
			- Use of Organization, Folders, and Projects to mirror the structure in your organization
			- Make use of inheritance when assigning permissions
			- Apply "principle of least privilege" while granting roles at any level
		Groups
			- Grant roles to groups rather than individuals
			- If a task requires multiple roles, create a new group and assign roles and users to it
			- Perform regular audits of group members
		Service Accounts
			- Define a naming convention for service accounts - the name should clearly identify its purpose
			- Be careful when granting `serviceAccountActor` role to user
				- That user will have all the permissions of that service account
			- Implement key rotation

	Data Protection
		Data Exfiltration
			- An authorized person extracts data from the secured systems where it belongs, and either shares it with unauthorized third parties or moves it to insecure systems. Can occur due to the actions of malicious or compromised actors, or accidentally.

			Types
				- Outbound email
				- Downloads to insecure devices
				- Uploads to external services
				- Insecure cloud behavior
				- Rouge admins, pending employee terminations

		Exfiltration Prevention
			Don'ts for VMs
				- Don't allow outgoing connections to unknown addresses
				- Don't make IP addresses public
				- Don't allow remote connection software e.g. RDP
				- Don't give SSH access unless absolutely necessary
			Do's for VMs
				- Use VPCs and firewalls between them.
				- Use a bastion host as a chokepoint for access.
					- Limit source IPs that can communicate with the bastion.
					- Configure firewall rules to allow SSH traffic to private instances from only the bastion host.
		
		Data Loss Prevention API
			- Understand and manage sensitive data in Cloud Storage or Datastore
			- Easily classify and redact sensitive data
				- Classify textual and image-based information
				- Redact sensitive data from text files and classify

			Classification
				- Input: Raw data
				- Output: 
					- Information type - specific types of sensitive information.
					- Likelihood
					- Offset - location where that sensitive information occurs
			
			Redaction
				- Input: Raw data
				- Output: Redacted data

			How?
				ML-based
					- Contextual analysis
					- Pattern matching
				Rule-based
					- Checksum
					- Word and phrase list