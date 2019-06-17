# Google Cloud Storage

- Blob storage. Content not indexed.
- Virtually unlimited storage.
- Can have domain name buckets
- Can make requesters pay (ex. requester in different project)
- PubSub can have notifications based on operations to buckets/objects
- Objects are immutable
- Can set Cache-Control metadata for frequently accessed objects
- Keep in mind compliance requirements when storing data in certain regions. 

- No native directory support 
    - Forward slashes have no special meaning
    - Performance of a native filesystem is not present.
    
- Storage classes can change, but the objects (files) within them retain their storage class.
- Not ideal for high volume read/write
- A way to store data that can be commonly used by Dataproc and Bigquery

- Signed Urls
    - Provided limited permission and time to make a request.
    - Contain auth information in query string, allowing users without credentials to perform specific actions on a resource.
- Signed URL to give temporary access and users do not need to be GCP users
    - TODO

## Storage Classes

### Multi-Regional

- Serving website content, interactive workloads, mobile game/gaming applications
- Highest availability
- Geo-redundant: Stores data in at least 2 regions separated by at least 100 miles within the multi-regional location of the bucket.

### Regional

- Storing data used by Compute Engine
- Better performance for data-intensive computation
    
### Nearline

- Accessed once a month max
- 30 day min. storage duration
- Ex. Data backup, disaster recovery, archival storage

### Coldline

- Accessed once a year max
- 90 day min. storage duration
- Ex. Data stored for legal or regulatory reasons


## Versioning

- Needs to be enabled
- Things this enables:
    - List archived versions of an object
    - Restore live version of an object from an older state
    - Permanently delete an archived version
- Archived versions retain ACLs and does not necessarily have same permissions as live version of object.

## IAM vs ACLs
- IAM
    - Apply to all objects within a bucket.
    - Standard Roles
        - Storage.objectCreator
        - Storage.objectViewer
        - Storage.objectAdmin
        - Storage.admin – full control over buckets
            - Can apply to a specific bucket.
    - Primitive Roles
- ACL (Access Control List)
    - Use if need to customize access to individual objects within a bucket.
    - Permissions – What actions can be performed.
    - Scopes – Which defines who can perform the actions (user or group)
    - Reader/Writer/Owner

## Encryption

### Encryption at rest (Google-Managed Encryption Keys)

- Default (AES-256)
- Use TLS or HTTPS to protect data as it travels over Internet

### Server-side encryption
- Layers on top of default encryption
- Occurs after GCS receives data, but before written to disk

#### Customer-supplied encryption keys
- Provide key for each GCS operation
- Key purged from servers after operation is complete
- Stores only a cryptographic hash of key for future requests
- Transfer Service, Dataflow, and Dataproc do not support this currently
- Key rotation
    - Edit .boto config file
    - Encryption_key = [NEW_KEY]
    - Decryption_key1 = [OLD_KEY]
    - gsutil rewrite -k gs:://[BUCKET]/[OBJECT]
    
#### Customer-managed encryption keys
- Generate and manage keys using Cloud Key Management Service (KMS)
- KMS can be independent from the project that contains buckets (separation of duties)
- Uses service accounts to encrypt/decrypt
- Cloud SQL exports to GCS and Dataflow do not support this currently

### Client-side encryption
- Occurs before data sent to GCS
- GCS performs default encryption on it as well.
