# Datalab

- Managed Jupyter notebooks
- Great for use with a Dataproc cluster to write PySpark jobs
- Powerful interactive tool to explore, analyze, transform and visualize data and build machine learning models on GCP.

- Supports Python, SQL (BQ), and JavaScript (for BQ user-defined functions)
    - In Cell
        - %%bq query –name queryname
            - SQL underneath
        - %%bq execute -q queryname

- Runs on GCE instance, dedicated VPC and Cloud Source Repository
    - datalab create <instance name>
    - datalab-network (VPC) is created
    - datalab connect <instance name>
    - Cloud Source Repository
        - Used for sharing notebook between users

## 3 Ways to Run:
- Locally
    - Good if only one person using
- Docker on GCE
    - Better
    - Use by multiple people through SSH or CloudShell
    - Uses resources on GCE
- Docker + Gateway
    - Best
    - Uses a gateway and proxy
    - Runs locally

## Notebooks
- Can be in Cloud Storage Repository (git repo)
    - Use ungit to commit changes to notebooks

## Persistent Disk
- Notebooks can be cloned from GCS to VM persistent disk.
- This clone => workspace => add/remove/modify files
- Notebooks autosave, but you need to commit.

## Kernel
- Opening a notebook => Backend kernel process manages session and variables.
- Each notebook has 1 python kernel
- Kernels are single-threaded
- Memory usage is heavy – execution is slow – pick machine type accordingly

## APIs and Services
- Enable Compute Engine API

## Sharing Notebook Data:
- GCE access based on GCE IAM roles:
    - Must have Compute Instance Admin and Service Account Actor roles to connect to datalab instance.
        - Service Account Actor role deprecated. Use Service Account Token Creator instead.
- Notebook access per user only
- Sharing data performed via shared Cloud Source Repository
- Sharing is at the project level.

## Creating Team Notebooks
- 2 Options
    - Team lead creates notebooks for users using –for user option:
        - datalab create [instance] –for-user bob@blah.net
    - Each user creates their own datalab instance/notebook
- Everyone accesses same shared repository of datalab/notebooks
- NO web console option
- Machine Type
    - Standard n1 by default
    - Multi-threading does not work, but can use high memory
    - Custom machine types supported as well.
- Can disable creating shared cloud repository
    - --no-create-repository

## Connecting
- SSH tunnels to notebook on port 8081
- datalab connect <instance name>
    - RSA key is passphrase
- Can configure idle timeouts on the actual webpage

## Cost
- Free
- Only pay for GCE resources Datalab runs on and other GCP services you interact with.
