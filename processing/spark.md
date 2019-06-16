# Apache Spark

-	Spark
o	A distributed computing engine used along with Hadoop
o	Interactive shell to quickly process datasets
o	Has a bunch of built in libraries for machine learning, stream processing, graph processing, …, etc.
o	Dataflow
o	General purpose
	Exploring
	Cleaning and Preparing
	Applying machine learning
	Building data applications
o	Interactive
	Provides a REPL environment
•	Read Evaluate Print Loop
o	Reduces boilerplate of standard MapReduce Java code.
o	Resilient Distributed Datasets (RDDs)
	In memory collections of objects.
	Can interact with billions of rows
	Properties
•	Partitions
•	Read-only
o	Immutable
o	Operations allowed on RDD
	Transformations
•	Transform into another RDD
	Actions
•	Request a result
•	Aware of it’s Lineage
o	When created, RDD knows
	A transformation
	It’s parent RDD
o	Implications of Lineage
	Built in fault tolerance
•	Reconstruct from source if something goes wrong
	Lazy Evaluation
•	Materialize only when necessary
o	Spark Core
	Basic functionality of Spark
	Written in Scala
	Runs on a Storage System and Cluster Manager
•	Plug and play components
•	Can be HDFS and YARN
o	Spark ML
	MLlib is Spark’s machine learning library.
	Provides tools such as:
•	ML Algorithms: classification, regression, clustering, collaborative filtering.
•	Featurization: feature extraction, transformation, dimensionality reduction, and selection
•	Pipelines: tools for constructing, evaluating, and tuning ML Pipelines
•	Persistence: saving and load algorithms, models, and Pipelines
•	Utilities: linear algebra, statistics, data handling, etc.
