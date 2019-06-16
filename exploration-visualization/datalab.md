# Datalab

Datalab

-	Managed Jupyter notebooks
-	Great for use with a Dataproc cluster to write PySpark jobs
-	Supports Python, SQL (BQ), and JavaScript (for BQ user-defined functions)
o	In Cell
	%%bq query –name queryname
•	SQL underneath
	%%bq execute -q queryname
-	Runs on GCE instance, dedicated VPC and Cloud Source Repository
o	datalab create <instance name>
o	datalab-network (VPC) is created
o	datalab connect <instance name>
o	Cloud Source Repository
	Used for sharing notebook between users
-	3 Ways to Run:
o	Locally
	Good if only one person using
o	Docker on GCE
	Better
	Use by multiple people through SSH or CloudShell
	Uses resources on GCE
o	Docker + Gateway
	Best
	Uses a gateway and proxy
	Runs locally
-	Powerful interactive tool to explore, analyze, transform and visualize data and build machine learning models on GCP.
-	Notebooks
o	Can be in Cloud Storage Repository (git repo)
	Use ungit to commit changes to notebooks
-	Persistent Disk
o	Notebooks can be cloned from GCS to VM persistent disk.
o	This clone => workspace => add/remove/modify files
o	Notebooks autosave, but you need to commit.
-	Kernel
o	Opening a notebook => Backend kernel process manages session and variables.
o	Each notebook has 1 python kernel
o	Kernels are single-threaded
o	Memory usage is heavy – execution is slow – pick machine type accordingly
-	APIs and Services
o	Enable Compute Engine API
-	Sharing Notebook Data:
o	GCE access based on GCE IAM roles:
	Must have Compute Instance Admin and Service Account Actor roles to connect to datalab instance.
•	Service Account Actor role deprecated. Use Service Account Token Creator instead.
o	Notebook access per user only
o	Sharing data performed via shared Cloud Source Repository
o	Sharing is at the project level.
-	Creating Team Notebooks
o	2 Options
	Team lead creates notebooks for users using –for user option:
•	datalab create [instance] –for-user bob@blah.net
	Each user creates their own datalab instance/notebook
	Everyone accesses same shared repository of datalab/notebooks
o	NO web console option
o	Machine Type
	Standard n1 by default
	Multi-threading does not work, but can use high memory
	Custom machine types supported as well.
o	Can disable creating shared cloud repository
	--no-create-repository
-	Connecting
o	SSH tunnels to notebook on port 8081
o	datalab connect <instance name>
	RSA key is passphrase
o	Can configure idle timeouts on the actual webpage
-	Cost
o	Free
o	Only pay for GCE resources Datalab runs on and other GCP services you interact with.
