# PubSub

- Server-less messaging “middleware”
- Many to many asynchronous messaging
- Decouples sender and receiver
- Attributes can be set by sender (KV pairs)
- Glue that connects all components
- Order not guaranteed
- Encoding as a Bytestring (utf-8) required for publishing.
- Publishers: Any app that can make HTTPS requests to googleapis.com

## Message Flow
- Publisher app creates a topic object and sends a message to the topic.
- Messages persisted in message store until acknowledged by subscribers
- Messages forwarded from topic to all subscriptions individually.
- Subscriber receives pending messages from its subscription and acknowledges each one to the Cloud Pub/Sub service.
    - Push
        - WebHook endpoint (must accept POST HTTPS request)
        - Lower latency, more real time.
    - Pull
        - Subscriber explicitly calls pull method which requests messages for delivery.
        - More efficient message deliver/consume mechanism
        - Better for large volume of messages – batch delivery.
    - Acknowledgement Deadline
        - Per subscriber
        - Once a deadline has passed, an outstanding message becomes unacknowledged.
- When acknowledged, it is removed from the subscriptions message queue.

## Architecture
- Data Plane
    - Handles moving messages between publishers and subscribers
    - Forwarders
- Control Plane
    - Handles assignment of publishers and subscribers to server on the data plane.
    - Routers

## Use Cases
- Balancing workloads in a network cluster
- Implementing async workflows
- Distributing event notifications
- Refreshing distributed caches	
    - i.e. An app can publish invalidation events to update the IDs of objects that have changed
- Logging to multiple systems
- Data streaming from various processes or devices
- Reliability improvement
    - i.e. a single-zone GCE service can operate in additional zones by subscribing to a common topic, to recover from failures in a zone or region.

## Deduplicate
- Database table to store hash value and other metadata for each data entry.
- Message_id can be used to detect duplicate messages

## Out of Order Messaging
- Messages may arrive from multiple sources out of order.
- Pub/Sub does not care about message ordering
- Dataflow is where out of order messages are processed/resolved.
    - Ingest – Pub/Sub
    - Process - Dataflow
- Can add message attributes to help with ordering.
### Handling Order
- Order does not matter at all
    - i.e. queue of independent tasks, collection of statistics on events
    - Perfect for PubSub. No extra work needed.
- Order in final result matters
    - i.e. Logs, state updates
    - Can attach a timestamp to every event in the publisher and make the subscriber store the messages in some underlying data store (such as Datastore) that allows storage or retrieval by the sorted timestamp.
- Order of processed messages matters
    - i.e. transactional data where thresholds must be enforced
    - Subscriber must either:
        - Know the entire list of outstanding messages and the order in which they must be processed, or
            - Assigning each message a unique identifier and storing in some persistent place (Datastore) the order in which messages should be processed.
            - Subscriber check persistent storage to know the next message it must process and ensure that it only processes that message next.
        - Have a way to determine all messages it has currently received whether or not there are messages it has not received that it needs to process first.
            - Cloud Monitoring to keep track of the oldest_unacked_message_age metric.
            - Temporarily put all messages in some persistent storage and ack the messages.
            - Periodically check the oldest unacked message age and check against the publish timestamps of the messages in storage.
            - All messages published before the oldest unacked message are guaranteed to have been received, so those messages can be removed from the persistent storage and processed in order.
        - Single synchronous publisher/subscriber
            - Can just use a sequence number to ensure ordering.
            - Requires the use of a persistent counter.

## Cost
- Data volume used per month (per GB)

## IAM
- Control access at project, topic, or subscription level
- Resource types: Topic, Subscription, Project
- Service accounts are best practice.
- Pubsub.publisher
- Pubsub.subscriber
- Pubsub.viewer or viwer
- Pubsub.editor or editor
- Pubsub.admin or admin
