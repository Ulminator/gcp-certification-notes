# Datastore

- Typically, not used for either OLTP or OLAP
    - Fast lookup on keys is the most common use case.
    
- Serverless

- Specialty is that query execution depends on the size of the returned result and not the size of the data set.
    - Best for lookup of non-sequential keys (needle in haystack)
    - Built on top of BigTable
        - Non-consistent for every row.
        - Document DB for non-relational data.
        - MongoDB equivalent (JSON-oriented NoSQL).

- Suitable for:
    - Atomic transactions
        - Can execute a set of operations where all succeed, or none occur.
    - ACID transactions, SQL-like queries.
    - Structured data.
    - Hierarchical document storage such as HTML and XML

- Query
    - Can search by keys or properties (if indexed)
    - Key lookups somewhat similar to Amazon DynamoDB
    - Allow for SQL-like querying down to property level.
    - Does not support:
        - Join operations
        - Inequality filtering on multiple properties.
            - Only 1 inequality filter per query is allowed.
    - Filtering on data based on a result of a subquery.

- Performance
    - Fast to Terabyte scale, low latency.
    - Quick read, slow write as it relies on indexing every property (default) and must update indexes as updates/writes occur.

- Comparison to RDBMS
    - All Datastore queries use indices.
    - Query time depends on size of result set alone in Datastore whereas RDBMS also depends on size of data set.
    - Entities (rows) of the same kind (table) can have different properties (fields).
    - Different entities can have properties with same name, but different value type.

- Properties can vary between entities.
    - Think optional tags in HTML.

- Avoid DataStore when:
    - You need very strong transaction support.
    - Non-hierarchical or unstructured data (use BigTable instead)
    - Need extreme scale (10M+ read/writes per second) - BigTable
    - Analytics/BI/Data warehousing (BQ instead)
    - If application has a lot of writes and updates on key columns.
    - You need near zero latency (use in memory db Redis)

- Use DataStore when:
    - Scaling of read performance – to virtually any size.
    - Use for hierarchical documents with KV data.
    - Apps that need highly available structured data at scale.
    - Product catalogs, real time inventory
    - User profiles – mobile apps
    - Game save states

- Single Datastore database per project

- Where can you host?
    - Multi-regional for wide access
    - Single region for lower latency for single location
    - Cannot change after assignment… (have to delete project)

## Entity Groups
- Hierarchical relationship between entities.
- Ancestor Paths and Child Entities.

## Index Types
- Built in – default option
    - Allows single property queries
- Composite – specified with index configuration file (index.yaml)
    - gcloud datastore create-indexes index.yaml
        - Creating/updating

## Deleting Index
- datastore indexes cleanup
    - Deletes all indexes for the production Datastore mode instance that are not mentioned in the local version of index.yaml.

## Exploding Indexes
- Default – create entry for every possible combination of property values
- Results in higher storage and degraded performance
- Solutions
    - Use custom index.yaml file to narrow index scope
    - Do not index properties that don’t need indexing

## Full Indexing
- Built in indices on each property (~field) of each entity kind (~table row).
- Composite indices on multiple property values.
- Can exclude properties from indexing if certain it will never be queried.
- Each query is evaluated using its “perfect index”

## Perfect Index
- Given a query, which is the index that most optimally returns query results?
- Depends on following (in order)
    - Equality filter
    - Inequality filter
    - Sort conditions if any specified.

## Implications of Full Indexing
- Updates are really slow.
- No joins possible.
- Can’t filter results based on subquery results.
- Can’t include more than one inequality filter.

## Multi-Tenancy
- Separate data partitions for each client organizations.
- Can use the same schema for all clients, but vary the values.
- Specified via a namespace (inside which kinds and entities can exist)

## Transaction Support
- Can optionally use transactions – not required
- Stronger than BigQuery and BigTable

## Consistency
- Strongly consistent
    - Return up to date result, however long it takes
    - Ancestor query
        - Those that execute against an entity group
        - Can set the read policy of a query to make this eventually consistent.
    - key-value operations
- Eventually consistent
    - Faster, but might return stale data
    - Global queries/projections

## Deleting entities in bulk?
- Use Dataflow
    - Datastore delete template that can be used to delete entities selected by a GQL query.

## Exporting Entities
- Deploy App Engine service that calls Datastore mode managed export feature.
- Can run this service on a schedule with an App Engine Cron Service.

## Cloud Firestore
- Newest version of Datastore.
- Native Mode
    - New strongly consistent storage layer.
    - New data model:
        - Kind => Collection Group
        - Entity => Document
        - Property => Field
        - Key => Document ID
    - Real-time updates
    - Mobile and Web client libraries
        - Scales to millions of concurrent clients.
    - Datastore Mode
        - Removes previous consistency limitations of Datastore.
        - Strongly consistent queries across the entire database.
        - Transactions can access any number of entity groups.
        - Scales to millions of writes per second.

## IAM Roles
- Datastore.owner with Appengine.appAdmin
    - Full access to Datastore mode.
- Datastore.owner without Appengine.appAdmin
    - Cannot enable Admin access
    - Cannot see if Datastore mode Admin is enabled
    - Cannot disable Datastore mode writes
    - Cannot see if Datastore mode writes are disabled.
- Datastore.user
    - Read/write access to data in Datastore mode database.
    - Intended for application developers and service accounts.
- Datastore.viewer
    - Read access to all Datastore mode resources.
- Datastore.importExportAdmin
    - Full access to manage import and exports.
- Datastore.indexAdmin
    - Full access to manage index definitions.