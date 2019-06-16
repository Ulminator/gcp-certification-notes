# AI Platform

- Can use multiple ML platforms such as TensorFlow, scikit-learn and XGBoost

## Workflow
- Source and prepare data
    - Data analysis
        - Join data from multiple sources and rationalize it into one dataset.
        - Visualize and look for trends.
        - Use data centric languages and tools to find patterns in data.
        - Identify features in your data.
        - Clean the data to find any anomalous values caused by errors in data entry or measurement.
    - Data preprocessing
        - Transform valid, clean data into the format that best suits the needs of your model.
        - Examples
            - Normalizing numeric data to a common scale.
            - Applying formatting rules to data. Ex. removing HTML tagging from a text feature.
            - Reducing data redundancy through simplification. Ex. converting a text feature to a bag of words representation.
            - Representing text numerically. Ex. assigning values to each possible value in a categorical feature (or 1 hot).
            - Assigning key values to data instances.
    - Develop model
    - Train an ML model on your data
        - Benefits of Training Locally
            - Quick iteration
            - No charge for cloud resources
    - Deploy trained model
        - Upload to GCS bucket
        - Create a model resource in AI Platform specifying GCS path
        - Scenario: Maximize speed and minimize cost of model prediction and deployment:
            - Export trained model to a SavedModel format.
            - Deploy and run on Cloud ML Engine.
    - Send prediction requests to your model
        - Online
        - Batch
    - Monitor predictions on an ongoing basis
        - APIs to examine running jobs.
        - Stackdriver
        - Jobs that can occasionally fail
            - Monitor status of Jobs object for ‘failed’ jobs states.
    - Manage models and model versions
        - gcloud ai-platform

## Preparing Data
- Gather data
- Clean data
    - Clean data by column (attribute)
    - Instances with missing features.
    - Multiple methods of representing a feature.
        - Length measurement in different scale/format
    - Features with values far out of the typical range (outliers)
    - Significant change in data over distances in time, geographic location, or other recognizable characteristics.
    - Incorrect labels or poorly defined labeling criteria.
- Split data
    - Train, Validation, Test
    - Better to randomly sample the subsets from one big dataset than use pre-divided data. Otherwise could be non-uniform => overfitting.
    - Size of datasets: training > validation > test
- Engineer data features
    - Can combine multiple attributes to make one generalizable feature.
        - Address and timestamp => position of sun
    - Can use feature engineering to simplify data.
    - Can get useful features and reduce number of instances in dataset by engineering across instances. I.e. calculate frequency of something.
- Preprocess features

## Training Overview
- Upload datasets already split (training, validation) into something AI Platform can read from.
- Sets up resources for your job. One or more virtual machines (training instances)
    - Applying standard machine image for the version of AI Platform your job uses.
    - Loading application package and installing it with pip.
    - Installing any additional packages that you specify as dependencies.
- Distributed Training Structure
    - Running job on a given node => replica
    - Each replica given a single role or task in distributed training:
        - Master
            - Exactly 1 replica
            - Manages others and reports status for the job as a whole.
            - Status of master signals overall job status.
            - Single process job => the sole replica is the master for the job
        - Worker(s)
            - 1 or more replica
            - Do work as designated in job configuration.
        - Parameter Servers
            - 1 or more replicas
            - Coordinate shared model state between the workers.
    - Tiers
        - Scale tiers
            - Number and types of machines you need.
        - CUSTOM tier
            - Allows you to specify the number of Workers and parameter servers.
        - Add these to TrainingInput object in job configuration.
    - Exception
        - The training service runs until your job succeeds or encounters an unrecoverable error.
        - Distributed Case – status of the master replica that signals the overall status.
        - Running a Cloud ML Engine training job locally (gcloud ml-engine local train) is especially useful in the case of testing distributed models.
- Start training
    - Package application with any dependencies required
    - 2 ways
        - Submit by running `gcloud ai-platform jobs submit training`
        - Send a request to the API ar `projects.jobs.create`
            - Need `ml.jobs.create` permission.
    - Job ID
        - Define base name for all jobs associated with a given model and then append a data/time.
    - Job-Dir
        - Save model checkpoints to this GCS path.
        - Useful for VM restarts.
        - Used for job output.
    - GPUs
        - More effective at running certain operations on tensor data than adding another machine with one or more CPU cores.
        - Can specify GPU-enabled machines to run your job.
    - TPUs
        - Tensor Processing Units
        - Google’s custom developed ASICs used to accelerate machine learning workloads with TensorFlow.
        - Steps
            - Authorize Cloud TPU service account name associated with GCP project
            - Add service account as a member of your project with role Cloud ML Service Agent.
        - Only in us-central1 currently.

## Hyperparameter Tuning
- –config hptuning_config.yaml
- Hyperparameter: Data that governs the training process itself.
    - DNN
        - Number of layers
        - Number of nodes for each layer
- Usually constant during training.
- How it works:
    - Running multiple trials in a single training job.
    - Each trail is a complete execution of your training application with values for chosen hyperparameters, set within limits specified.
- Tuning optimizes a single target variable (hyperparameter metric)
    - Multiple params per metric.
- Default name is `training/hptuning/metric`
    - Recommended to change to custom name.
    - Must set `hyperparameterMetricTag` value in `HyperparameterSpec` object in job request to match custom name.
- How to actually tune?
    - Define a command line argument in main training module for each tuned hyperparameter.
    - Use value passed in those arguments to set the corresponding hyperparameter in application’s TensorFlow code.
- Types
    - Double
    - Integer
    - Categorical
    - Discrete – List of values in ascending order.
- Scaling
    - Recommended for Double and Integer types.
    - Linear, Log, or Reverse Log Scale
- Search Algorithm
    - Unspecified
        - Same behavior as when you don’t specify a search algo.
        - Bayesian optimization
    - Grid Search
        - Useful when specifying a number of trials that is more than the number of points in feasible space.
            - In such cases AI Platform default may generate duplicate suggestions.
        - Can’t use with any params being Doubles
    - Random Search

## Online and Batch Prediction
- Can process one or more instances per request.
- Can serve predictions from a TensorFlow SavedModel.
- Can make requests
    - Legacy Editor
    - Legacy Viewer (Online only)
    - AI Platform Admin or Developer
### Online
- Optimized to minimize the latency of serving predictions.
- Predictions returned in the response message.
- Input passed directly as a JSON string.
- Returns as soon as possible.
- Runs on runtime version and in region selected when deploying model.
- Can serve predictions from a custom prediction routine.
- Can generate logs if model is configured to do so. Must specify option when creating model resource.
    - onlinePredictionLogging or –enable-logging (gcloud)
- Use when making requests in responses to application input or in other situations where timely inference is needed.
### Batch
- Optimized to handle a high volume of instances in a job and to run more complex models.
- Predictions written to output files in Cloud Storage location that you specify.
    - Can verify predictions before applying them. (sanity check)
- Input data passed directly as one or more UIRs of files in Cloud Storage locations.
- Asynchronous request.
- Can run in any available region, using any runtime version.
    - Should run with defaults for deployed model versions.
- Only Tensorflow supported. (Not XGBoost or scikit)
- Ideal for processing accumulated data when you don’t need immediate results.
    - i.e. a periodic job that gets predictions for all data collected since the last job.
- Generates logs that can be viewed on Stackdriver.
- Slow because AI Platform allocates and initializes resources for a batch prediction job when the request is sent.

## Prediction Nodes and Resource Allocation
- Think of a Node as a VM
### Batch
- Scales nodes to minimize elapsed time job takes.
- Allocates some nodes to handle your job when you start it.
- Scales the number of nodes during the job in an attempt to optimize efficiency.
- Shuts down nodes as soon as job is done.
### Online
- Scales nodes to maximize number of requests it can handle without too much latency.
- Allocates some nodes the first time you request predictions after a long pause in requests.
- Scales number of nodes in response to request traffic, adding nodes when traffic increases, removing them when there are fewer requests.
- Keeps at least 1 node ready over a period of several minutes, to handle requests even when there are none to handle.
- Scales down to zero after model version goes several minutes without a prediction request.

## Predictions from Undeployed Models
- Batch only
- Specify URI of a GCS locations where the model is stored.
- Explicitly set runtime version in request.

## IAM
- Project Roles
    - Ml.admin
    - Ml.developer
    - Ml.viewer
- Model Roles
    - Ml.modelOwner
    - Ml.modelUser
