# Dataflow

Dataflow

-	Executes Apache Beam Pipelines
-	Can be used for batch or stream data
-	Scalable, fault-tolerant, multi-step processing of data.
-	Often used for data preparation/ETL for data sets.
-	Integrates with other tools (GCP and external)
o	Natively – PubSub, BigQuery, Cloud AI Platform
	BQ I/O connector can stream JSON through Dataflow to BQ
o	Connectors – BigTable, Apache Kafka
-	Pipelines are regional-based
-	Follows the Flink Programming Model
o	Data Source -> Transformations -> Data Sink
-	Use when:
o	No dependencies on Apache Hadoop/Spark
o	Favor hands-off/serverless
o	Preprocessing for machine learning with Cloud ML Engine
-	Apache Beam Architecture
o	Pipeline
	Entire set of computations
	Not linear, it is a DAG
	Beam programs start by constructing a Pipeline object.
o	A single, potentially repeatable job, from start to finish, in Dataflow.
o	Defined by driver program.
	Actual computations run on a backend, abstracted in the driver by a runner.
•	Driver: Defines DAG
•	Runner: Executes DAG
o	Supports multiple backends
	Spark
	Flink
	Dataflow
	Beam Model
o	Element
	A single entry of data (e.g. table row)
o	PCollection
	Distributed data set in pipeline (immutable)
	Specialized container classes that can represent data sets of virtually unlimited size.
•	Fixed size: Text file or BQ table
•	Unbounded: Pub/Sub subscription
	Side inputs
•	Inject additional data into some PCollection
•	Can inject in ParDo transforms.
o	PTransform
	Data processing operation (step) in pipeline
•	Input: 1 or more PCollection
•	Processing function on elements of PCcollection
•	Output: 1 or more PCollection
o	ParDo
	Core of parallel processing in Beam SDKs
	Collects the zero or more output elements into an output PCollection.
	Useful for a variety of common data processing operations, including:
•	Filtering a data set.
o	Better than a Filter transform in the sense that a filter transform can only filter based on input element => no side input allowed.
•	Formatting or type-converting each element in a data set.
•	Extracting parts of each element in a data set.
•	Performing computations on each element in a data set.
-	Dealing with late/out of order data
o	Latency is to be expected (network latency, processing time, etc.)
o	PubSub does not care about late data, that is resolved in Dataflow.
o	Resolved with Windows, Watermarks, and Triggers.
o	Windows = Logically divides element groups by time span.
o	Watermarks = Timestamp
	Event time – When data was generated
	Processing time – when data processed anywhere in pipeline
	Can use Pub/Sub provided watermark or source generated.
o	Trigger = Determine when results in window are emitted.
	(Submitted as complete)
	Allow late-arriving data in allowed time window to re-aggregate previously submitted results.
	Timestamps, element count, combinations of both.
-	Templates
o	Google provided templates.
	WordCount
	Bulk compress/decompress GCS Files
	PubSub to (Avro, PubSub, GCS Text, BigQuery)
	Datastore to GCS Text
	GCS Text to (BigQuery, PubSub, DataStore)
-	Requires a Staging Location where intermediate files may be stored.
-	IAM
o	Project-level only – all pipelines in the project (or none)
o	Pipeline data access separate from pipeline access.
o	Dataflow Admin
	Full pipeline access
	Machine type/storage bucket config access
o	Dataflow.developer
	Full pipeline access
	No machine type/storage bucket access (data privacy)
o	Dataflow Viewer
	View permissions only.
o	Dataflow.worker
	Enables service account to execute work units for a Dataflow pipeline in Compute Engine.
	Dataflow API also needs to be enabled.
-	Stopping a Dataflow Jobs
o	Cancelling
	Immediately stop and abort all data ingestion and processing.
	Buffered data may be lost.
o	Draining
	Cease ingestion but will attempt to finish processing any remaining buffered data.
	Pipeline resources will be maintained until buffered data has finished processing and any pending output has finished writing.
-	Pipeline Update
o	Replace an existing pipeline in-place with a new one and preserve Dataflow’s exactly-once processing guarantee.
o	When updating pipeline manually, use DRAIN instead of CANCEL to maintain in flight data.
	Drain command is supported for streaming pipelines only
o	Pipelines cannot share data or transforms.
-	Handling Invalid Inputs in Dataflow
o	Catch exception, log an error, then drop input.
	Not ideal
o	Have a dead letter file where all failing inputs are written for later analysis and reprocessing.
	Add a side output in Dataflow to accomplish this.
-	Converting from Kafka to PubSub
o	CloudPubSubConnector is a connector to be used with Kafka Connect to publish messages from Kafka to PubSub and vice versa.
o	Provides both a sink connector (to copy messages from Kafka to PubSub) and a source connector (to copy messages from PubSub to Kafka.
-	Windowing
o	Can apply windowning to streams for rolling average for the window, max in a window etc.
o	Types
	Fixed Time Windows (Tumbling Window)
•	Fixed window size
•	Non-overlapping time
•	Number of entities differ within a window
	Sliding Time Windows (overlapped)
•	Fixed window size
•	Overlapping time
•	Number of entities differ within a window
•	Window Interval: How large window is
•	Sliding Interval: How much window moves over
	Session Windows
•	Changing window size based on session data
•	No overlapping time
•	Number of entities differ within a window
•	Session gap determines window size
•	Per-key basis
•	Useful for data that is irregularly distributed with respect to time.
	Single Global Window
•	Late data is discarded
•	Okay for bounded size data
•	Can be used with unbounded but use with caution when applying transforms such as GroupByKey and Combine
o	Default windowing behavior is to assign all elements of a PCollection to a single, global window even for unbounded PCollections.
-	Triggers
o	Determines when a Window’s contents should be output based on a certain being met.
	Allows specifying a trigger to control when (in processing time) results for the given window can be produced.
	If unspecified, the default behavior is to trigger first when the watermark passes the end of the window, and then trigger again every time there is late arriving data.
o	Time-Based Trigger
	Event Time Triggers
•	Operate on event time, as indicated by timestamp on each data elements.
•	This is the default trigger.
	Processing Time Triggers
•	Operate on the processing time – the time when the data element is processed at any given stage in the pipeline.
o	Data-Driven Trigger
	Operate by examining the data as it arrives in each window, and firing when that data meets a certain property.
	Currently, only support firing after a certain number of data elements.
o	Composite Triggers
	Combine multiple triggers in various ways.
-	Watermarks
o	System’s notion of when all data in a certain window can be expected to have arrived in the pipeline.
o	Tracks watermark because data is not guaranteed to arrive in a pipeline in order or at predictable intervals.
o	No guarantees about ordering.
o	Indicates all windows ending before or at this timestamp are closed.
o	No longer accept any streaming entities that are before this timestamp.
o	For unbounded data, results are emitted when the watermark passes the end of the window, indicating that the system believes all input data for that window has been processed.
o	Used with Processing Time
-	Tech
o	DirectPipelineRunner
	Allows you to execute operations in the pipeline directly and locally.
o	Create a cron job with App Engine Cron Service to run Dataflow job.
-	System Lag
o	Max time an element has been waiting for processing in this stage of the pipeline.
-	Wall Time
o	How long the processing takes.
-	Key Things 
o	Constraints you may have.
o	Why you would use JSON or Java related to Pipelines
-	How to improve performance?
o	TODO
