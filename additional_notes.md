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

How are audit logs and data access logs different from each other?

What role do templates play in GCP's deployment manager service?
	- Reuse of config files

Does the extensible service proxy always sit on the same VPC as your endpoint code?
	- Yes, so no network hop

Classifying user data as sensitive, and redacting if necessary, requires you to write complex machine learning programs. True or false?
	- False
	- Can use data loss prevention API

What is Cloud Interconnect
	- A service where GCP users with data-intensive or low-latency apps sign up with third party ISPs to connect to GCP

What is CDN Interconnect
	- A service where GCP users with high-volume egress data or frequent content updates to sign up with third party ISPs to connect to GCP.

Why is Cloud Router an essential part of any enterprise scale VPN?
	- When routes change on peer network, the VPN tunnel needs to be updated (deleted and re-created) disrupting network traffic.
	- The peer gateway is often a physical device which needs to be taken down for maintenance.
	- Static routes are slow to converge in routing protocols such as BGP; dynamic routes can be updated far more quickly.

Data Transfer Appliance vs. Storage Transfer Service

Cloud BigTable cbt tool?

Kafka on Compute Engine vs. PubSub?

You are building a data pipeline on Google Cloud. You need to select services that will host a deep neural network machine-learning model also hosted on Google Cloud. You also need to monitor and run jobs that could occasionally fail. What should you do?
	- B. Use Cloud Machine Learning to host your model. Monitor the status of the Jobs object for 'failed' job states.
	Feedback
	B (Correct Answer) - B is correct because of the requirement to host an ML DNN and Google-recommended monitoring object (Jobs); see the links below.

Cloud Dataprep?

You are building a data pipeline on Google Cloud. You need to prepare source data for a machine-learning model. This involves quickly deduplicating rows from three input tables and also removing outliers from data columns where you do not know the data distribution. What should you do?
	D. Use Cloud Dataprep to preview the data distributions in sample source data table columns. Click on each column name, click on each appropriate suggested transformation, and then click 'Add' to add each transformation to the Cloud Dataprep job.

You have data stored in a Cloud Storage dataset and also in a BigQuery dataset. You need to secure the data and provide 3 different types of access levels for your Google Cloud Platform users: administrator, read/write, and read-only. You want to follow Google-recommended practices.What should you do?
	D. Use the appropriate pre-defined IAM roles for each of the access levels needed for Cloud Storage and BigQuery. Add your users to those roles for each of the services.

You are developing an application on Google Cloud that will label famous landmarks in users’ photos. You are under competitive pressure to develop the predictive model quickly. You need to keep service costs low. What should you do?
	B. Build an application that calls the Cloud Vision API. Pass landmark locations as base64-encoded strings.

You are upgrading your existing (development) Cloud Bigtable instance for use in your production environment. The instance contains a large amount of data that you want to make available for production immediately. You need to design for fastest performance. What should you do?
 Change your Cloud Bigtable instance type from Development to Production, and set the number of nodes to at least 3. Verify that the storage type is SSD.

As part of your backup plan, you set up regular snapshots of Compute Engine instances that are running. You want to be able to restore these snapshots using the fewest possible steps for replacement instances. What should you do?
	D. Use the snapshots to create replacement instances as needed.

Data Studio?

Cloud Key Management Service?

You are working on a project with two compliance requirements. The first requirement states that your developers should be able to see the Google Cloud Platform billing charges for only their own projects. The second requirement states that your finance team members can set budgets and view the current charges for all projects in the organization. The finance team should not be able to view the project contents. You want to set permissions. What should you do?
	B. Add the finance team members to the Billing Administrator role for each of the billing accounts that they need to manage. Add the developers to the Viewer role for the Project.

You want to display aggregate view counts for your YouTube channel data in Data Studio. You want to see the video tiles and view counts summarized over the last 30 days. You also want to segment the data by the Country Code using the fewest possible steps. What should you do?
	B. Set up a YouTube data source for your channel data for Data Studio. Set Views as the metric and set Video Title and Country Code as report dimensions.

You are building storage for files for a data pipeline on Google Cloud. You want to support JSON files. The schema of these files will occasionally change. Your analyst teams will use running aggregate ANSI SQL queries on this data. What should you do?
	B. Use BigQuery for storage. Select "Automatically detect" in the Schema section.