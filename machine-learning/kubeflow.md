# Kubeflow

Kubeflow 

-	Helps orchestrate machine learning training pipelines across on-prem and cloud-based resources.
-	Can containerize training and serving infrastructure.
-	Components include:
o	Support for distributed TensorFlow training via the TFJob CRD
	TFJob is a Kubernetes custom resource used to run TensorFlow training jobs on Kubernetes.
o	The ability to serve trained models using TensorFlow Serving
	Flexible, high performance serving system for machine learning models.
	Installed by default when deploying Kubeflow.
o	A JupyterHub installation with many commonly required libraries and widgets included in the notebook installation, included those needed for TensorFlow Model Analysis (TFMA) and TensorFlow Transform (TFT)
	JupyterHub – Multi user server for Jupyter notebooks
	TFMA – Library for evaluating TF models.
•	Uses Apache Beam
	TFT – Preprocessing/feature engineering
•	Can prevent Training Serving Skew
o	Difference between performance during training and performance during serving. This can be caused by:
	Discrepancy between how you handle data in the training and serving pipelines.
	Change in the data between when you train and when you serve.
	Feedback loop between model and algorithm.
•	Uses Apache Beam
o	Kubeflow Pipelines
	Has UI for managing and tracking experiments, jobs, and runs.
	An engine for scheduling multi-step ML workflows
	An SDK for defining and manipulating pipelines and components
	Notebooks for interacting with the system using the SDK
