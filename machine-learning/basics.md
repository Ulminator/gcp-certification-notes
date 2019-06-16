# Basics of Machine Learning

## Types of Problems:
- Classification
- Regression
- Clustering
- Rule Extraction

## Supervised Learning
- Labels associated with the training data are used to correct the algorithm.

## Unsupervised Learning
- The model has to be set up right to learn the structure in the data.

## Representation Learning Algorithms
- Feature learning. Algorithm identifies important features on its own.

## Deep Learning
- Algorithms that learn what features matter.
- Neural Networks
    - Most common class of deep learning algorithms.
    - Used to build representation learning systems.
    - Composed of neurons (binary classifiers)
    - Wide
        - Better for memorization
    - Deep
        - Better for generalization

## Neurons
- Apply 2 functions on inputs.
- Best values of W and b found by using cost function, optimizer, and training data.
- Back Propagation
## Linear (affine) transformation
- Like linear regression.
- X1 * W1 + b
    - W = Weights
        - Shape of W
            - First dimension is equal to number of dimensions of feature vector.
            - Second dimension is equal to the number of params required to be tuned. (same goes for b)
    - B = Bias
        - Determined during training process.
## Activation Function
- Helps to model non linear functions. (Logistic regression)
- Introduces non-linearity into the network.
- Ex.
    - ReLu (Rectified Linear Unit)
        - Max(Wx + b, 0)
    - SoftMax
        - Multi-class logistic regression
        - Layer right before output.
            - Must have same number of nodes as the output layer.
        - Use candidate sampling for large # of output classes.
    - Sigmoid
        - Binary classification logistic regression

## Failure Cases for Backpropagation
### Vanishing Gradients
- Gradients for lower layers (closer to input) can become very small.
- Leads to very slow training, if at all.
- ReLu as an activation function can help prevent vanishing gradients.
### Exploding Gradients
- Weights in a network are very large, the gradients for the lower layers involve products of many large terms.
- Gradients get too large to converge.
- Batch normalization and/or lowering learning rate can prevent this.
### Dead ReLU Units
- Weighted sum of a ReLU falls below 0, ReLU can get stuck.
- Outputs 0 activation, contributing nothing to network’s output, and gradients can no longer flow through it during backpropogation.
- Lowering learning rate can help keep ReLU from dying.

## Modeling Linear Regression
- 1 neuron with just an affine transformation.
- Y = Ax + b
- Minimize Least Square Error

## Optimizers for Best Fit
- Method of Moments
- Method of Least Squares
- Maximum likelihood Estimator

## Reducing Loss
- Convergence: When loss stops changing or at least changes extremely slowly.
- Gradient is a vector.
- Learning rate is a scalar.
- Gradient is multiplied by the learning rate.

### Hyperparameters
- Configuration settings used to tune hot the model is trained.
- Steps
    - Total number of training iterations. One step calculates the loss from one batch and uses that value to modify the model’s weights once.
- Batch Size
    - Number of examples (chosen at random) for a single step.
    - Total # of trained examples = Batch Size * Steps
- Learning rate

### Stochastic Gradient Descent
- Random samples from data set to estimate.
- Uses a batch size of 1 per iteration.
- Works (given enough iterations), but noisy
### Mini-Batch Stochastic Gradient Descent
- Compromise between full batch and SGD
- Typically between 10 – 10K examples chosen at random.
- Reduce noise, but still more efficient than full-batch.

## Periods
- the # of training examples in each period = batch size * steps / period
- Controls granularity of reporting.
    - If periods = 7 and steps = 70, the loss value will be output every 10 steps.
- Modifying period value does not alter what model learns.

## Generalization
- The less complex an ML model, the more likely that a good empirical result is not just due to the peculiarities of the sample.
- Overfitting occurs when a model tries to fit the training data so closely that it does not generalize well to new data.
- Identify Overfitting
    - Loss for the validation set is significantly higher than for the training set. (look at loss curve (loss/iterations))
    - Validation loss eventually increases with iterations.
- If the key assumptions of supervised ML are not met, then we lose important theoretical guarantees on our ability to predict new data.
- 3 Basic Assumptions
    - We draw examples independently and identically at random from the distribution. I.e. examples don’t influence each other.
    - The distribution is stationary; that is it does not change within the data set.
    - We draw examples from partitions from the same distribution.

## Training, Validation, and Test Sets
- Training set – a subset to train a model.
- Test set – a subset to test the trained model.
    - Must be large enough to yield statistically meaningful results.
    - Is representative of the data set as a whole. i.e. don’t pick a test set with different characteristics than the training set.
- Doing many rounds of just using a training and test set might cause implicit fitting to the peculiarities of the specific test set.
    - Use a validation set too!
    - Flow
        - Train model
        - Use model on validation set
        - Update hyperparams
        - Repeat
        - Finally test on test set

## Representation
- Process of mapping data to useful features.
- Discrete feature
    - A feature with a finite set of possible values.
    - Categorical feature are an example
### One-Hot Encoding
- A sparse vector in which:
    - One element is set to 1
    - All other elements are set to 0
- Commonly used to represent strings or identifiers that have a finite set of possible values.
### Feature Engineering
- Process of determining which features might be useful in training a model, and then converting raw data from log files and other sources into said features.
- Sometimes called feature extraction.
### Qualities of Good Features
- Avoid rarely used discrete feature values.
    - Should appear more than 5 or so times in a data set.
    - Having many examples with the same discrete value gives the model a chance to see the feature in different settings, and in turn, determine when it’s a good predictor for the label.
- Prefer clear and obvious meanings
    - Ex. house_age_years vs. house_age
    - Some cases, noisy data causes unclear values, such as data coming from sources that didn’t check for appropriate values.
        - Ex. user_age_years: 277
- Don’t mix “magical” values with actual data
    - Ex. quality_rating between 0 and 1.
        - If no value, it is set to -1
        - Create a Boolean feature to indicate if quality rating was defined.
    - Replace “magical” values as follows
        - For a variable that take a finite set of values (discrete variables), add a new value to the set and use it to signify that feature value is missing.
        - For continuous variables, ensure missing values do not affect the model by using the mean value of the feature’s data.
- Account for upstream instability
    - Definition of a feature shouldn’t change over time.
### Cleaning Data
- Scaling feature vectors
    - Converting floating point feature values from their natural range (100 to 900) to a standard range (0 to 1 or -1 to 1)
    - Scaling ~= Normalization
    - If only 1 feature, little to no practical benefit.
    - Multiple features, great benefits
        - Helps gradient descent convere more quickly
        - Helps avoid NaN traps
            - One number in the model becomes a NaN (value exceeds floating point precision limit during training) and due to math operations, every other number in the model also eventually becomes NaN.
        - Helps the model learn appropriate weights for each feature. Without scaling, the model pays too much attention to features having a wider range.
- Handling extreme outliers
    - Log scaling
        - Still leaves a tail on distribution
    - Cap or Clipping
        - Reduce feature values that are greater than a set maximum value down to that maximum value.
        - Also, increasing feature values that are less than a specific minimum value up to that minimum value.
    - Binning (Bucketing)
        - Converting a (usually continuous) feature into multiple binary features called buckets or bins, typically based on a value range.
- Scrubbing
    - Data can be unreliable due to:
        - Omitted values
        - Duplicate examples
        - Bad labels
        - Bad feature values
    - “Fix” by removing them from data set.
    - Omitted and duplicate easy to detect.
    - Detecting bad data in aggregate by using Histograms
    - Stats can also help identifying bad data:
        - Max and Min
        - Mean and Median
        - Standard Deviation
- Follow These Rules:
    - Keep in mind what your data should look like
    - Verify that the data meets these expectations
        - Or that you can explain why it doesn’t
    - Double check that the training data agrees with other soruces
        - i.e. dashboards

## Feature Crosses
- A synthetic feature formed by crossing (Cartesian product) individual binary features obtained from categorical data or from continuous features via bucketing.
- Helps represent nonlinear relationships.
- Encoding Nonlinearity
- Crossing One-Hot Vectors

## Regularization
- Minimize loss + complexity
    - Structural Risk Minimization
    - Penalizes complexity to prevent overfitting
- 2 Common Ways to Think About Model Complexity
    - As a function of the weights of all the features in the model
        - L2 Regularization
        - A feature weight with a high absolute value is more complex than one with a low absolute value.
        - L2 = w1^2 + w2^2 + … + wn^2
        - Consequences of L2 Regularization
            - Encourages weight values toward 0 (but not exactly 0)
            - Encourages the mean of the weights toward 0, with a normal (bell shaped or Gaussian) distribution.
    - As a function of the total number of features with nonzero weights
- Most developer tune the overall impact of the regularization term by multiplying it by a scalar known as lambda (regularization rate)
    - Minimize(loss function + lambda(regularization function))
    - When choosing a lambda value, the goal is to strike the right balance between simplicity and training-data fit
        - Lambda too high
            - Model will be simple but run the risk of underfitting data.
        - Lambda too low
            - Model will be more complex and run the risk of overfitting data.
## Early Stopping
- Ending training before the model reaches convergence (training loss finishes decreasing).
- End model training when loss on a validation dataset starts to increase, that is, when generalization performance worsens.
## Sparsity
- Sparse vectors often contain many dimensions.
- Creating a feature cross results in even more dimensions.
- High Dimensionality -> Large Model Size -> Large RAM reqs
- L1 Regularization
    - Penalizes absolute value of weights. (|weight|)
    - Derivative of L1 is a constant, k. (2 * weight for L2)
        - A force that subtracts some constant value from the weight every time.
    - Pushes weights toward 0
        - Efficient for wide models.
        - Reduces # of features -> smaller model size
    - May cause informative features to get a weight of exactly 0:
        - Weakly informative features
        - Strongly informative features on different scales
        - Informative features strongly correlated with other similarly informative features.
## Dropout
- Useful for neural networks.
- Randomly dropping out unit activations in a network for a single gradient step.
- 0.0 = No dropout regularization.
- 1.0 = Drop out everything. Model learns nothing.

## Logistic Regression
- A model that generates a probability for each possible discrete label value in classification problems by applying a sigmoid function to a linear prediction.
- Often used in binary classification problems, but can also be used in multi-class classification problems (multinomial regression)
- Sigmoid Function
    - Maps logistic or multinomial regression output (log odds) to probabilities, returning a value between 0 and 1.
    - Can serve as an activation function in neural networks.
- Loss and Regularization
    - Loss function is Log Loss
    - Regularization
        - L2 or Early Stopping
- One vs All
    - Classification problem with N possible solutions.
    - A one-vs-all solution consists of N separate binary classifiers.

## Classification
- Classification Threshold (Decision Threshold)
    - Determines what the probability output from logistic regression is classified as.
### Accuracy
- Number of correct predictions over total number of predictions
- TP + TN / (TP + TN + FP + FN)
### Class Imbalanced Dataset
- Labels have significantly different frequencies in a classification problem.
- Accuracy is not enough in this scenario.
### Confusion Matrix
- An NxN table that summarizes how successful a classification model’s predictions were.
- Useful when calculating precision and recall
### Precision
- Identifies the frequency with which the model was correct when predicting the positive class.
- TP/ (TP + FP)
- i.e. how many predicted cats are actually cats
- Raising classification threshold reduces FP, thus improving precision.
### Recall
- Out of all the possible positive labels, how many did the model correctly identify.
- TP / (TP + FN)
- i.e. number of predicted cats out of all cats
- Raising classification threshold will cause # of TP to decrease or stay the same and will cause the # of FN to increase or stay the same. Thus recall will either stay constant or decrease.
- Improving precision often reduces recall and vice versa.
### ROC Curve
- Receiver Operating Characteristic Curve
- Shows performance of classification model at all classification thresholds.
- TP rate (TP / TP + FN) vs. FP rate (FP / FP + TN)
- Lowering classification threshold increase TP and FP.
### AUC
- Area Under the ROC Curve
- Provides an aggregate measure of performance across all possible classification thresholds.
- 0 – worst model
- 1 – best model
- Desirable Because:
    - Scale Invariant
        - Measures how well predictions are ranked, rather than their absolute values.
    - Classification Threshold Invariant
        - Measures the quality of the model’s predictions irrespective of what classification threshold is chosen.
- Limitations
    - Scale invariance is not always desirable
        - We may need well calibrates probability outputs and AUC won’t tell us that.
    - Classification threshold invariance is not always desirable
        - In cases where there are wide disparities in the cost of false negatives vs. false positives, it may be critical to minimize one type of classification error.
### Prediction Bias
- = average of predictions – average of labels
- Different than bias, b, in wx + b
- Possible root causes of prediction bias:
    - Incomplete feature set
    - Noisy data set
    - Buggy pipeline
    - Biased training sample
    - Overly strong regularization
- Avoid Calibration Layer as a fix
    - Fixing symptoms rather than cause.
    - Built a more brittle system that you must now keep up to date.
- Examine prediction bias on a bucket of examples

## Embeddings
- D-Dimensional Embeddings
    - Assumes something can be explained by d aspects.
- Map items to low-dimensional real vectors in a way that similar items are close to each other.
