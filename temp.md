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

## Querying

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

# Required Context
	- Hadoop, Spark, MapReduce (granular level)
	- Hive, HBase, Pig (use cases and some implementation quirks)
		- Hotspotting
		- HBase is a Columnar database
		- Poor choice of row keys can impact performance
	- RDBMS, indexing, hashing

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