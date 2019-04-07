Static, No SSL
	- Just buy storage

	- Buy disk space
	- Automatic scaling

	- Google Cloud Storage

Secure Sockets Layer, CDN, etc.
	- Need HTTPS serving, release management etc.

	- SSL so that HTTPS serving is possible
	- CDN edges world-over
	- Atomic deployment, one-click rollback

	- Firebase Hosting + Google Cloud Storage

Load Balancing, Scaling
	- Get VMs, manage yourself

	- Control load balancing, scaling etc yourself
	- No need to buy machines or install OS, dev stack, languages etc

	- Compute Compute Engine (IaaS)

Lots of dependencies
	- Create containers, manage clusters

Heroku, Engine Yard
	- Just focus on code, forget the rest


DevOps
	Puppet
	Chef
	Salt
	Ansible

What do you do when Google shutsdown your machine for maitenance?
	- Migrate to a new machine

What is a necessary condition before I can attach a persistent disk to my VM?
	1. `gcloud compute disks create test-disk --size=100GB --zone us-central1-f`
		- Persistent disk created must be in the same zone.
		- At least 200GB in size for optimal read/write performance
		- In order to attach it to a VM it must be formatted first.
		https://gcloud.compute.com/compute/docs/disks/add-persistent-disk#formatting
	2. `gcloud compute instances attach-disk test-instance --disk test-disk --zone us-central1-f`

Get list of hard disks attached once ssh'd into a VM
	`ls -l /dev/disks/by-id/`

Rank the following storage options in order of maximum possible storage space per instance:
	Cloud storage > Persistent (SSD or standard) > Local SSD

How do storage options differ with container engine instances relative to compute engine instances?
	- Container disks are emphemeral by default; need to use a specific abstraction to make them persistent.

File Storage is an abstraction on Block Storage

SSH into VM
	sudo apt-get update
	sudo apt-get -y -qq install git
	git clone https://github.com/GoogleCloudPlatform/training-data-analyst
	cd training-data-analyst/CPB100/lab2b
	ls -l
	nano ingest.sh # Downloads earthquake data
	sudo apt-get --fix-missing install python-mpltoolkits.basemap python-numpy python-matplotlib

Why use gcloud init after adding a service account keyfile
	- Re-initialize configuration
	- Select service account
	- Say yes to enable APIs

Why is using a cloud proxy to access Cloud SQL a good idea?
	- Added security
		- TCP Tunnel
		- Without it you would need to whitelist IP address or setup SSL.

How does Clous SQL allow access via the Cloud Shell command line?
	- It sets up a temporary whitelist for IP of cloud shell instance.
	- Temporary because cloud shell sessions are ephemeral
	- Connecting a second time could use a different IP

Cloud Spanner has tremendous support for transaction processing. This makes even single read calls very slow.
	- False: Use Single Read Call

What is the number of nodes recommended for Cloud Spanner instances in production?
	- At least 3

Why would we never use Hive (or BigQuery) for OLTP?

Why never use Cloud SQL for OLAP?

Why are windowing and partitioning so important in Hive?
	Large datasets would cause memory overflowing errors.

In a typical ETL use-case, is Hive (or BigQuery) a source or a sink?
	Pig is a source | Hive is a sink

Why is it easier to add columns on the fly in BigTable than in Cloud Spanner?
	- All you need to do is insert new rows in BigTable
	- Difficult in Cloud Spanner since it changes the schema -> a lot of writes
	- Also Cloud Spanner uses interleaving

Normalization of data is great in traditional database theory, but has drawbacks in the distributed world. What are some?
	- Bandwidth is bottleneck.
		- Accessing lots of different nodes
		- Disk seeks rather than storage

BigTable supports equi-joins, but not constraints or indices. True or False?
	False

How does the row key affect physical storage in BigTable or HBase?
	Sorted in lexicographical order. Similar ones will be stored next to each other -> hotspotting.

What two operations are very low latency in Cloud BigTable?
	- Lookups using the rowkey
	- Scan operations

Do entities of a particular kind in DataStore all have the same properties?
	False

How does BigQuery take Hive's schema-on-read to the next level?
	Schema auto detection

How does BigQuery's take on partitioning vary from Hive's?
	Hive has dynamic and static partitioning (richer support)
		- Dynamic turned off by default
		- Static are explicitely specified by the user
	BQ is dynamic, automatic, and calculated based off of a pseudo-column which is calculated on the basis of when data is loaded into BQ

Where would you quickly look up metadata information for a table in BQ?

How can you reduce the amount of data processed by BQ?
	- Select less columns to return

What is the transformation which can produce zero or more outputs for every input record (Beam)?
	- beam.FlatMap(lambda line: func(line, searchTerm))

What is the reason you would use the ParDo class when running transforms in a Java pipeline?
	- Core element of tranformations in Java Beam pipeline
	- Process elements in parallel across multiple machines

How do you specify custom pipeline configuration parameters in a Dataflow?
	- Setup an Options interface that extends from PipelineOptions
	- Setup correct getters and setters along with corresponding annotations

What does the wall time represent for transforms in execution graph?
	- the total amount of time that a particular transformation took to process elements
	- helps identify bottlenecks in your pipeline

What Apache Beam library would you use to find the top N elements in a list?
	- beam.transforms.combiners.Top.Of(5, by_value)
	- by_value is a custom function that returns a boolean

What Apache Beam Java class would you use to represent a key-value pair?
	KV<>

Select Count(*) on BQ
	- Uses 0 bytes

What do we need to do to allow our local machine to connect to the Hadoop cluster manager console on Dataproc?
	Create a firewall rule

What is the environment variable on CloudShell which holds your project id?
	- $DEVSHELL_PROJECT_ID

How would you browse the HDFS directory on your Dataproc cluster?
	- Go to <master node ip>:50070
	- Click on Utilities
	- Then click Browse the file system

How to ensure Dataproc cluster can run gcloud commands on master node?
	```
		gcloud dataproc clusters create ${CLUSTERNAME} \
			--scopes=cloud-platform \  #THIS LINE DOES IT
			--tags codelab \
			--zone=us-central1-c
	```

Unbounded dataset source of truth
	- The stream /message transport layer

When would you choose to use the DataflowPipelineOptions to read configuration parameters for you Beam pipeline?
	- When you want to access project id, staging location, etc..

Only supervised learning algorithms have a training step. True or false?
	True

Can one-hot notation be used for continuous data such as heights or weights? Or is it restricted to categorical items?
	No it cannot represent continuous data.

In a multiple regression of GPA on IQ, gender and income for 1000 students, how many elements does each feature vector have?
	- 1000 feature vectors
	- 3 elements each

The output of a set of SoftMax classification neurons will be a set of probabilities

Why are estimators called a high level way to perform linear or logistic regression?
	- They abstract decisions
		- What optimizer?
		- Mini-batch or stochastic gradient descent?

What informs the estimator object of the training dataset, y labels and other properties such as the batch size of our training?
	- The input function

While using an estimator for logistic regression why do you NOT need to specify training labels in one-hot notation?
	- It will do it for you under the hood.

If you're working on an unfamiliar data set, what are some things you can do to check whether cause-effect relationships exist?
	- Exploratory visualizations

How would you save a trained model so you can use it later on for predictions?
	- Specify model directory when setting up an estimator
	- The estimator will write out the weights and bias values

	- Instantiate estimator by pointing to model directory to use it.

What APIs would you use if you want the ability to read the text in a sign in a foreign country?
	Vision
	Translate
	- Also need a API key for a service acct

Identify which images depict happy people and which ones depict unhappy people.
	- Look at joylikelihood and sorrowlikelihood using vision api

What feature type would we specify in our request to the vision API if we wanted to identify famous places like the Eiffel Tower?
	- Specify feature to be LANDMARK_DETECTION

Advantages obtained by the use of VPC
	- Convenience of DNS lookup within instances
	- Multi-tenancy
	- Firewall isolation between different components of an app architecture

Networking withing the same VPC in a project
	- Instances in the same VPC use private IP addresses to communicate with each other
	- Instances in the same VPC can use DNS to communicate with each other using instance names
	- Firewalls can be erected between subnets inside the same VPC project

Projects and VPCs are global; subnets within a VPC are regional