# Data Studio

-	Easy to use data visualization and dashboards.
o	Drag and drop report builder.
-	Part of G Suite, not GCP:
o	Uses G Suite access/sharing permissions, not Google Cloud (no IAM)
o	Google account permissions in GCP will determine data source access.
o	Files saved in Google Drive.
-	Connect to many Google, Google Cloud, and other services:
o	BQ, Cloud SQL, GCS, Spanner
o	YouTube Analytics, Sheets, AdWords, local upload
	Local
•	Stored in managed GCS bucket
•	First 2GB free
o	Many third party integrations
-	Price
o	Free
o	BQ access run normal query costs
-	Data Lifecycle
o	Visualization
-	Basic Process
o	Connect to data source
o	Visualize data
o	Share with others
-	Creating Charts
o	Use combinations of dimensions and metrics
o	Create custom fields if needed
o	Add date range filters with ease
-	Caching (most relevant for BQ)
o	Options for using cached data performance/costs
o	2 choices
	Query Cache
•	Remembers query issues by reports components (i.e. charts)
•	When performing same query, pulls from cache.
•	If query cache cannot help, goes to prefetch cache.
•	Cannot be turned off.
	Prefetch Cache (exam material?)
•	“Smart Cache” – predicts what might be requested
•	If prefetch cache cannot serve data, pulls from live data set
•	Only active for data sources that use owner’s credentials for data access
o	If I create table that pulls from BQ table that does not use my credentials for data access, prefetch will be disabled.
•	Can be turned off.
o	When to turn caching off:
	Need to view “fresh data” from rapidly changing data set.
