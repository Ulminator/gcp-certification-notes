# Transfer Appliance

-	Transfer large amounts of data quickly and cost-effectively to GCP.
-	Transfers directly to GCS or BQ
-	Data Size >= 20TB
-	Offline Data Transfer
-	Takes more than 1 week to upload data.
-	Workflow
o	Receive Transfer Appliance and configure it and connect it to your network.
o	Before data is stored, it is deduplicated, compressed and encrypted with AES 256 algorithm using a password and passphrase specified by user.
o	Data integrity check is performed.
o	Transfer Appliance is shipped back to Google.
o	Encrypted data is copied to GCS staging bucket.
	Still compressed, deduplicated, and encrypted.
o	Email will be sent to user notifying the rehydration process can start.
o	Transfer Appliance Rehydrator application is run specifying the GCS destination bucket.
	This application is run on GCE.
	Compared CRC32C hash value of each file being rehydrated.
	If checksums don’t match, file is skipped and appears in skip file list with Data corruption detected.
o	Data integrity check performed again.
o	Appliance securely wiped and re-imaged.
-	Use Cases:
o	Data Collection
	Geographical, environmental, medical, or financial data for analysis.
	Need to transfer data from researchers, vendors, or other sites to GCP.
o	Data Replication
	Supporting current operations with existing on prem infrastructure but experimenting with cloud.
	Allows decommissioning duplicate datasets, test cloud infrastructure, and expose data to machine learning analysis.
o	Data Migration
	Offline data transfer is suited for moving large amounts of existing backup images and archives to ultra-low-cost, highly durable, and highly available archival storage (Nearline/Coldline).
-	**NOTE** - Should not be used for repeated data transfers.
