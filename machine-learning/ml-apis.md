# ML API's

Pre-trained ML API’s
-	For App Developers
-	Sight
o	Vision AI
	Image Recognition/analysis
	Label Detection
•	Extracts info in image across categories
	Text Detection (OCR)
•	Detect and extract text from images
	Safe Search
•	Recognize explicit content
	Landmark Detection
	Logo Detection
	Image Properties
•	Dominant colors, pixel counts
	Crop Hints
•	Crop coordinates of dominant object/face
	Web Detection
•	Find matching web entries
	Object Localizer
•	Returns labels and bounding boxes for detected objects.
	Product Search
•	Uses image and specific region(s) or largest object of interest to return matching items from product set.
o	AutoML Vision
	Object Detection
•	Bounding box smart multi-object detection, Google Vision API on steroids.
	Edge
•	The IoT version of Vision detection for Edge Devices.
•	Optimized to achieve high accuracy for low latency use cases on memory-constrained devices.
•	Use Edge Connect to securely deploy the AutoML model to IoT devices (such as Edge TPUs, GPUs, and mobile devices) and run predictions locally on the device.
o	Video Intelligence API
	Has pre-trained models that recognize a vast number of objects, places, and actions in stored and streaming video.
	Labels, shot changes, explicit content, subtitles
	Use cases:
•	Content moderation
•	Recommended content
•	Media archives
•	Contextual advertisements
o	AutoML Video Intelligence
	Video media tagging.
	Train custom video classification models.
	Ideal for projects that require custom labels which aren’t covered by the pre-trained Video Intelligence API.
	Detect shot changes
•	Detect scene changes in a segment or throughout the video.
-	Language
o	Natural Language API
	Syntax analysis
	Entity analysis
	Sentiment analysis
	Content classification
	Multi-language
o	AutoML Natural Language
	Handling things like domain specific sentiment analysis and more.
	Can classifies text using own custom labels.
o	Translation API
	Detect and translate languages
	Beta:
•	Glossary
•	Batch translations
o	AutoML Translation
	Upload translated language pairs -> Train -> Evaluate
-	Conversation
o	Cloud Speech-to-Text API
	Convert audio to text
	Multi-lingual support
	Understand sentence structure
o	Cloud Text-to-Speech API
	Convert text to audio
	Multiple languages/voices
	Natural sounding synthesis
o	Dialogflow Enterprise Edition
	Conversational experiences
	Virtual assistants
	Sentiment Analysis
•	Model chat-oriented conversations and responses, to assist you as you build interactive chatbots.
	Text-to-Speech
•	Chatbots trigger synthesized speech for more natural user interaction.
-	Cloud AutoML
o	Enables developers with limited machine learning expertise to train high-quality models specific to their business needs.
o	Relies on transfer learning and neural architecture search technology.
o	AutoML Tables
	Workflow:
•	Table input
•	Define data schema and labels
•	Analyze input features
•	Train (automatic)
o	Feature engineering
	Normalize and bucketize numeric features
	Create one-hot encoding and embeddings for categorical features
	Perform basic processing for text features
	Extract date- and time-related features from Timestamp columns.
o	Model selection
	Parallel model testing
•	Linear
•	Feedforward deep neural network
•	Gradient Boosted Decision Tree
•	AdaNet
•	Ensembles of various model architectures
o	Hyperparameter tuning
•	Evaluate model behavior
•	Deploy
	Structured Data
•	Can use data from BigQuery or GCS (CSV)
	AutoML Tables vs BigQuery ML
•	BQ
o	More focused on rapid experimentation or iteration with what data to include in the model and want to use simpler model types for this purpose.
	Can potentially return model in minutes
•	AutoML
o	Have finalized the data.
o	Optimizing for maximizing model quality without needing to manually do feature engineering, model selection, ensembling, and so on.
o	Willing to wait longer to attain that model quality.
	Takes at least an hour to train.
o	Have a wide variety of feature inputs (beyond numbers and classes) that would benefit from the additional automated feature engineering that AutoML Tables provides.
-	Cloud Job Discovery
o	More relevant job searches
o	Power recruitment, job boards
-	Basic Steps for Most APIs
o	Enable API
o	Create API key
o	Authenticate with API key
o	Encode in base64 (optional)
o	Make an API request
o	Requests and outputs via JSON
-	Structured Data
o	AutoML Tables
o	Cloud Inference API
	Quickly run large scale correlations over types time series data.
o	Recommendations AI (Beta)
o	BigQuery ML (beta)
-	Pricing
o	Pay per API request per feature
o	Feature as in Landmark Detection
-	How to convert images, video, etc for use with API?
o	Can use Cloud Storage URI for GCS stored objects
o	Encode in base64 format
-	How to combine API’s for scenarios?
o	Search customer service calls and analyze sentiment
	Speech to Text then Sentiment Analysis with Natural Language
