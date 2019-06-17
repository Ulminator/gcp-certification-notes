# Storage Transfer Service

- Transfers data from an online data source (Amazon S3, HTTP/HTTPS location, GCS bucket) to a data sink (always GCS bucket).
- Schedule one-time transfer operations or recurring ones
- Delete existing objects in the destination bucket if they don’t have a corresponding object in source
- Delete source objects after transferring them
- Schedule periodic synchronization from data source to data sink with advanced filters based on file creation data, file-name filters, and the times of day you prefer to import data.

## Use cases:
- Backup data to GCS from other storage providers
- Move data from one GCS bucket to another (enables availability to different groups of users or applications)
- Periodically move data as part of a processing pipeline or analytical workflow

## Transfer Service vs. Gsutil
- On premise data source : gsutil
- Another cloud storage provider data source : Transfer Service
