# Cloud Spanner

- Distributed and scalable solution for RDBMS (more expensive)
- Horizontal scaling: Add more machines
- Use when:
    - Need high availability
    - Strong consistency
    - Transactional support for reads and writes (especially writes)
- Don’t use when:
    - Data is not relational, or not even structured
    - Want an open source RDBMS
    - Strong consistency/availability is overkill

## Data Model
- Specifies a parent-child relationship for efficient storage
- Interleaved representation (like HBase)

### Parent Child Relationship
- Between tables
- Cause physical location for fast access
    - i.e. query Students and Grades together, make Grades child of Student
- Primary key of parent table must to be part of the key in the interleaved child table.

### Interleaving
- Rows are stored in sorted order of primary key values
- Child rows are inserted between parent rows with that key prefix

### Hotspotting
- Need to choose primary keys carefully (like HBase)
- Do not use monotonically increasing values, else writes will be on the same locations.
    - No timestamps (also sequential)
        - Use descending order if timestamps are required.
- Use hash of key value if using naturally monotonically ordered keys (serial in postgres)

### Splits
- Parent-child relationship can get complicated (i.e. 7 layers deep)
- Spanner is distributed – uses “splits”
- Split – Range of rows that can be moved around independent of other rows
- Added to distribute high read-write data (to break up hotspots)

### Secondary Indices
- Key-based storage ensures fast sequential scan of keys (like HBase)
- Can also add secondary indices (unlike HBase)
    - Can cause data to be stored twice
        - i.e. Grades -> Course table | Grades -> Students table
- Fine grained control on use of indices
    - Force query to use specific index: Index Directives
    - Force column to be copied into secondary index (use a STORING clause)

- Data Types
    - Non-normalized types such as ARRAY and STRUCT available too.
        - STRUCTs: NOT OK in tables, but can be returned in queries
        - ARRAYs: OK in tables, but ARRAYs of ARRAYs are not

## Transactions
- Supports serializability
    - All transactions appear if they were executed in a serial order, even if some operations of distinct transactions actually occurred in parallel.
- Stronger than traditional ACID
    - Transactions commit in an order that is reflected in their commit timestamps
    - Commit timestamps are “real time”
- 2 Transaction Modes
    - Locking read-write
        - Slow
        - Only one that supports writing data
    - Read-only
        - Fast
        - Only requires read locking
- If making a one-off read use “Single Read Call”
    - Fastest, no transaction checks needed!

## Staleness
- Can set timestamp bounds
    - Strong: Read latest data
    - Bounded Staleness: Read version no later than …
        - Could be in past or future
            
## Multitenancy
- Classic way is to create a separate database for each customer.
- Recommended way for Spanner: Include a CustomerId key column in tables.
    
## Replicas
- Paxos-based replication scheme in which voting replicas take a vote on every write request before it is committed.
- Writes
    - Client write requests always go to leader replica first, even if a non-leader is closer geographically.
    - Leader logs incoming write, forwards it in parallel to other replicas that are eligible to vote.
    - Replicas complete its write and then responds back to leader with a vote on whether the write should be committed.
    - Write is committed when a quorum agrees.
- Reads
    - Reads that are part of a read-write transaction are served from the leader replica, since the leader maintains the locks required to enforce serializability.
    - Single read and reads in a read-only transaction might require communication with leader, depending on concurrency mode.
- Single-region instances can only use read-write replicas. (3 in prod)
- Types
    - Read-write
        - Maintain a full copy of your data.
        - Can vote, can become leader, can serve reads
    - Read-only
        - Maintain a full copy of your data, which is replicated from read-write replicas.
        - Can serve reads
        - Do not participate in voting to commit writes -> location of read-only replicas never contribute to write latency.
        - Allow scaling of read capacity without increasing quorum size needed for writes (reduces total time of network latency for writes)
    - Witness
        - Can vote
        - Easier to achieve quorums for writes without the storage and compute resources required by read-write replicas to store a full copy of data and serve reads.

## Production Environment
- At least 3 nodes
- Best performance when each CPU is under 75% utilization

## Architecture
- Nodes handle computation for queries, similar to that of BigTable.
    - Each node serves up to 2 TB of storage.
    - More nodes = more CPU/RAM = increased throughput.  
- Storage is replicated across zones (and regions, where applicable).
    - Like BigTable, storage is separate from computing nodes.
- Whenever an update is made to a database in one zone/region, it is automatically replicated across zones/regions.
    - Automatic synchronous replications.
        - When data is written, you know it has been written.
        - Any reads guarantee data accuracy.

## IAM
- Project, instance, or database level
- Roles/spanner._____
    - Admin – Full access to Spanner resources
    - Database Admin – Create/edit/delete databases, grant access to databases
    - Database Reader – Read databases and execute SQL queries and view schema.
    - Database User – Read and write to DB, execute sql on DB including DML and Partitioned DML, view and update schema.
    - Viewer – View that instances and databases exist
        - Cannot modify or read from database.
