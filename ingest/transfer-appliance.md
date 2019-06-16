# Transfer Appliance

- Transfer large amounts of data quickly and cost-effectively to GCP.
- Transfers directly to GCS or BQ
- Data Size >= 20TB
- Offline Data Transfer
- Takes more than 1 week to upload data.

## Workflow
- Receive Transfer Appliance and configure it and connect it to your network.
- Before data is stored, it is deduplicated, compressed and encrypted with AES 256 algorithm using a password and passphrase specified by user.
- Data integrity check is performed.
- Transfer Appliance is shipped back to Google.
- Encrypted data is copied to GCS staging bucket.
    - Still compressed, deduplicated, and encrypted.
- Email will be sent to user notifying the rehydration process can start.
- Transfer Appliance Rehydrator application is run specifying the GCS destination bucket.
    - This application is run on GCE.
    - Compared CRC32C hash value of each file being rehydrated.
    - If checksums don’t match, file is skipped and appears in skip file list with Data corruption detected.
- Data integrity check performed again.
- Appliance securely wiped and re-imaged.

## Use Cases
- Data Collection
    - Geographical, environmental, medical, or financial data for analysis.
    - Need to transfer data from researchers, vendors, or other sites to GCP.
- Data Replication
    - Supporting current operations with existing on prem infrastructure but experimenting with cloud.
    - Allows decommissioning duplicate datasets, test cloud infrastructure, and expose data to machine learning analysis.
- Data Migration
    - Offline data transfer is suited for moving large amounts of existing backup images and archives to ultra-low-cost, highly durable, and highly available archival storage (Nearline/Coldline).

**NOTE** - Should not be used for repeated data transfers.

