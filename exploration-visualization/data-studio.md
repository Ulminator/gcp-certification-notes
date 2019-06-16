# Data Studio

- Easy to use data visualization and dashboards.
    - Drag and drop report builder.

- Part of G Suite, not GCP:
    - Uses G Suite access/sharing permissions, not Google Cloud (no IAM)
    - Google account permissions in GCP will determine data source access.
    - Files saved in Google Drive.

- Connect to many Google, Google Cloud, and other services:
    - BQ, Cloud SQL, GCS, Spanner
    - YouTube Analytics, Sheets, AdWords, local upload
        - Local
            - Stored in managed GCS bucket
            - First 2GB free
        - Many third party integrations

## Cost
- Free
- BQ access run normal query costs

## Basic Process
- Connect to data source
- Visualize data
- Share with others
    
## Creating Charts
- Use combinations of dimensions and metrics
- Create custom fields if needed
- Add date range filters with ease

## Caching (most relevant for BQ)
- Options for using cached data performance/costs
- 2 choices
    - Query Cache
        - Remembers query issues by reports components (i.e. charts)
        - When performing same query, pulls from cache.
        - If query cache cannot help, goes to prefetch cache.
        - Cannot be turned off.
    - Prefetch Cache (exam material?)
        - “Smart Cache” – predicts what might be requested
        - If prefetch cache cannot serve data, pulls from live data set
        - Only active for data sources that use owner’s credentials for data access
            - If I create table that pulls from BQ table that does not use my credentials for data access, prefetch will be disabled.
        - Can be turned off.
- When to turn caching off:
    - Need to view “fresh data” from rapidly changing data set.
