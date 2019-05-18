Reducing Loss

  Hyperparmeters are the configuration settings used to tune how the model is trained.

  Convergence: When loss stops changing or at least changes extremely slowly.

  Gradient is a vector.

  Learning rate is a scalar.

  Gradient is multiplied by the learning rate.

  Stochastic Gradient Descent
    - Random samples from data set to estimate.
    - Uses a batch size of 1 per iteration.
    - Works (given enough iterations), but noisy

  Mini-Batch Stochastic Gradient Descent
    - Compromise between full batch and SGD.
    - Typically between 10 - 10K examples chosen at random.
    - Reduces noise, but still more efficient than full-batch.

Skipped `First Steps with TF` section.




SEND FEEDBACK
Solutions
Automating infrastructure with Cloud Composer
This tutorial demonstrates a way to automate cloud infrastructure by using Cloud Composer. The example shows how to schedule automated backups of Compute Engine virtual machine (VM) instances.

Cloud Composer is a fully managed workflow orchestration service on Google Cloud Platform (GCP). Cloud Composer lets you author workflows with a Python API, schedule them to run automatically or start them manually, and monitor the execution of their tasks in real time through a graphical UI.

Cloud Composer is based on Apache Airflow. Google runs this open source orchestration platform on top of a Google Kubernetes Engine (GKE) cluster. This cluster lets Airflow autoscale as needed, and opens up a host of integration opportunities with other GCP products.

This tutorial is intended for operators, IT administrators, and developers who are interested in automating infrastructure and taking a deep technical dive into the core features of Cloud Composer. The tutorial is not meant as an enterprise-level disaster recovery (DR) guide nor as a best practices guide for backups. For more information on how to create a DR plan for your enterprise, see the disaster recovery planning guide.

Defining the architecture
Cloud Composer workflows are defined by creating a Directed Acyclic Graph (DAG). From an Airflow perspective, a DAG is a collection of tasks organized to reflect their directional interdependencies. In this tutorial, you learn how to define an Airflow workflow that runs regularly to back up a Compute Engine virtual machine instance using Persistent Disk snapshots.

The Compute Engine VM used in this example consists of an instance with an associated boot persistent disk. Following the snapshot guidelines, described later, the Cloud Composer backup workflow calls the Compute Engine API to stop the instance, take a snapshot of the persistent disk, and restart the instance. In between these tasks, the workflow waits for each operation to complete before proceeding.

The following diagram summarizes the architecture:

Architecture for automating infrastructure

Before you begin the tutorial, the next section shows you how to create a Cloud Composer environment. The advantage of this environment is that it uses multiple GCP products, but you don't have to configure each one individually.

Cloud Storage: The Airflow DAG, plugin, and logs are stored in a Cloud Storage bucket.
Google Kubernetes Engine: The Airflow platform is based on a micro-service architecture, and is suitable to run in GKE.
Airflow workers load plugin and workflow definitions from Cloud Storage and run each task, using the Compute Engine API.
The Airflow scheduler makes sure that backups are executed in the configured cadence, and with the proper task order.
Redis is used as a message broker between Airflow components.
Cloud SQL Proxy is used to communicate with the metadata repository.
Cloud SQL and App Engine Flex: Cloud Composer also uses a Cloud SQL instance for metadata and an App Engine Flex app that serves the Airflow UI. These resources are not pictured in the diagram because they live in a separate Google-managed project.
For more details, see the Overview of Cloud Composer.

Scaling the workflow
The use case presented in this tutorial is simple: take a snapshot of a single virtual machine with a fixed schedule. However, a real-world scenario can include hundreds of VMs belonging to different parts of the organization, or different tiers of a system, each requiring different backup schedules. Scaling applies not only to our example with Compute Engine VMs, but to any infrastructure component for which a scheduled process needs to be run

Cloud Composer excels at these complex scenarios because it's a full-fledged workflow engine based on Apache Airflow hosted in the cloud, and not just an alternative to Cloud Scheduler or cron.

Airflow DAGs, which are flexible representations of a workflow, adapt to real-world needs while still running from a single codebase. To build DAGs suitable for your use case, you can use a combination of the following two approaches:

Create one DAG instance for groups of infrastructure components where the same schedule can be used to start the process.
Create independent DAG instances for groups of infrastructure components that require their own schedules.
A DAG can process components in parallel. A task must either start an asynchronous operation for each component, or you must create a branch to process each component. You can build DAGs dynamically from code to add or remove branches and tasks as needed.

Also, you can model dependencies between application tiers within the same DAG. For example: you might want to stop all the web server instances before you stop any app server instances.

These optimizations are outside of the scope of the current tutorial.

Using best practices for snapshots
Persistent Disk is durable block storage that can be attached to a virtual machine instance and used either as the primary boot disk for the instance or as a secondary non-boot disk for critical data. PDs are highly available—for every write, three replicas are written, but Google Cloud customers are charged for only one of them.

A snapshot is an exact copy of a persistent disk at a given point in time. Snapshots are incremental and compressed, and are stored transparently in Cloud Storage.

It's possible to take snapshots of any persistent disk while apps are running. No snapshot will ever contain a partially written block. However, if a write operation spanning several blocks is in flight when the backend receives the snapshot creation request, that snapshot might contain only some of the updated blocks. You can deal with these inconsistencies the same way you would address unclean shutdowns.

We recommend that you follow these guidelines to ensure that snapshots are consistent:

Minimize or avoid disk writes during the snapshot creation process. Scheduling backups during off-peak hours is a good start.
For secondary non-boot disks, pause apps and processes that write data and freeze or unmount the file system.
For boot disks, it's not safe or feasible to freeze the root volume. Stopping the virtual machine instance before taking a snapshot might be a suitable approach.

To avoid service downtime caused by freezing or stopping a virtual machine, we recommend using a highly available architecture. For more information, see Disaster recovery scenarios for applications.

Use a consistent naming convention for the snapshots. For example, use a timestamp with an appropriate granularity, concatenated with the name of the instance, disk, and zone.

For more information on creating consistent snapshots, see snapshot best practices.