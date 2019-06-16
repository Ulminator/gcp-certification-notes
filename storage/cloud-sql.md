# Cloud SQL

- Managed/No ops relational database (PostgreSQL, MySQL)
    - Complex queries perform better in postgresql

- Best for <b>gigabytes</b> of data with <b>transactional</b> nature
    - Low latency
    - Doesn’t scale well beyond GB’s
    - Data structures and underlying infrastructure required

- Too slow for analytics/BI/warehousing (OLAP)

- 2nd Generation Allow
    - Cloud Proxy Support
    - Higher availability configurations
    - Maintenance won’t take down the server

- Use SSD for production (instead of hard disk (persistent disk))

- Enable binary logging
    - For Point-in-time recovery and replication

- Bulk Loading Data
    - Copy data to GCS from SQL dump file or CSV files
        - SQL dump files cannot contain triggers, views, stored procedures
    - Import it into DB using copy from csv or something similar.
    - Use correct flags for dump file.
    - Compress data to reduce costs
        - Cloud SQL can import compressed .gz files
    - Use InnoDB for Second Generation instances

- Limited to 10 TB and is regional (not global)
    - Use Spanner if not good enough

- Use Case:
    - Medical Records
    - Blogs

- Read Replicas
    - In same region as master
    - Purpose to offload requests for analytics traffic from master.

- What to do when data size limits performance?
    - Many smaller tables perform better than one larger.

## IAM
- Cloudsql.admin
- Cloudsql.editor
    - Can’t see or modify permissions, users, or ssl certs.
    - No ability to import data or restore from backup, nor clone, delete, or promote instances.
    - No ability to delete databases, replicas, or backups.
- Cloudsql.viewer
    - Read only access to all Cloud SQL instances.
- Cloudsql.client
    - Connectivity access to Cloud SQL instances from App Engine and the Cloud SQL Proxy.
    - Not required for accessing an instance using IP addresses.
