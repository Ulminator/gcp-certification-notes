# Streams Introduction

- How can MapReduce be used to maintain a running summary of real-time data from sensors?
    - Send temp readings every 5 minutes

## Batches
- Bounded datasets
- Slow pipeline from data ingestion to analysis
- Periodic updates as jobs complete
- Order of data received unimportant
- Single global state of the world at any point in time
- Typically small/singular source
- Low latency not important
- Often stored in storage services GCS, Cloud SQL, BigQuery

## Streams
- Unbounded datasets
- Processing immediate, as data is received
- Continuous updates as jobs run constantly
- Order important, but out of order arrival tracked
- No global state, only history of events received
- Typically many sources sending tiny (KB) amounts of data
- Requires low latency
- Typically paired with Pub/Sub (ingest) and Dataflow (real-time processing)

### Process data one entity at a time or a collection of entities as a batch
- Filter error messages (logs)
- Find a reference to latest movies (tweets)
- Track weather patterns (sensor data)
### Store, display, act on filtered messages
- Trigger an alert
- Show trending graphs
- Warn of sudden squalls

## Stream-First Architecture
- Data items can come from multiple sources
    - Files, DBs, but at least one from a Stream
- All files are aggregated and buffered in one way by a Message Transport (Queue)
    - i.e. Kafka, PubSub
- Passed to Stream Processing system
    - Flink or Spark Streaming

## Micro-batches
- Message Transport
    - Buffer for event data
    - Performant and persistent
    - Decoupling multiple source from processing
- Stream Processing
    - High throughput, low latency
    - Fault tolerant with low overhead
    - Manage out of order events
    - Easy to use, maintainable
    - Replay streams
- A good approximation of stream processing is the use of micro-batches
    - Group data items (time they were received)
    - If small enough it approximates real-time stream processing
- Advantages
    - Exactly once semantics, replay micro-batches
    - Latency-throughput trade off based on batch sizes
        - Can adjust to use case
        - Low latency better
        - High throughput better
- Spark Streaming or Storm Trident
