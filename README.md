![Logo](https://github.com/AKGanesh/Indian-Sign-Language-Prediction/blob/main/Indian-sign-language-dataset.png)

# Indian Sign Language Prediction

Sign languages are complete communication systems, just like spoken languages. Instead of sound, they use visual cues like hand gestures, facial expressions, and body movements. These gestures combine hand shapes, positions, and movements to convey meaning. Developed naturally by deaf communities, sign languages are not universal. Different regions have their own sign languages, like American Sign Language (ASL) and Indian Sign Language (ISL).

The objective of this project is to develop a machine learning model that can detect a hand gesture and translate to an equivalent alphabet (ex: C, K, G..)

Model deployed on: https://huggingface.co/spaces/akganesh/ISL

## Implementation Details

- Dataset: Please check the Dataset details below
- Model: SVM, KNN
- Input: Upload hand gesture image in jpeg format
- Output: Predicted Class (alphabet) and probabilities
- Scores : Accuracy and cross validation score
- Others : How to deal with images, hyperparameter tuning, predict_proba, Pickle to dump and read models, Huggingface spaces, Streamlit and Gradio and OTSU, HOG, SURF etc.,

## Dataset details

The project will utilize Indian Sign Language data
Files:

- train: the training set provided consists of different folders with alphabet names (jpegs inside) ex: A->200 jpegs, X->150 jpegs..
- test: Make use of test_train_split and cv of GridSearchCV
- realtime: you can take a pic of your hand gesture to upload

## Process

- Data and Pre-processing
  - Import the required libraries
  - Read the dataset
  - Preprocessing (Convert, Resize, Normalize)
  - Feature Extraction (Otsu)
  - Data Preparation (flatten, convert to np array, label enc)
- Model Development
  - Divide dataframe to Input and Output (X,y)
  - Test Train Split
  - Choose Model, Fit and Predict
  - Check the accuracy score
  - Hyperparameter tuning (GridSearchCV with cv)
  - Get the best params and fit to the model
- Test and Predict
  - Test against the ground truth to check the model perf
  - Test with another model like KNN and compare the performance

## Evaluation and Results

| Method | Accuracy Score | Comments                    |
| ------ | -------------- | --------------------------- |
| KNN    | 0. 97+         | Varies with K value         |
| SVM    | 0. 98+         | Varies with hyperparameters |

## Observations

- When you are dealing with images, resize to 32 or lower will improve the traning speed. Make use of different techniques that will help resize without losing the details.
- ClassicML struggles with images, slow and heavily biased towards the training set. DeepLearning is more suited for image related tasks.

## Libraries

**Language:** Python

**Packages:** Sklearn, Numpy, Seaborn, Matplotlib, Skimage, Streamlit

## Roadmap

- To apply and test HOG, SURF, SIFT etc.,
- Work on the same problem with DeepLearning techniques

## FAQ

#### Whats is Accuracy Score?

Imagine you have a model that classifies images as cats or dogs. You test your model on 100 images with known labels. If the model correctly identifies 80 images (40 cats and 40 dogs), the accuracy score would be:

Accuracy Score = (Number of Correct Predictions) / (Total Number of Predictions)
= 80 / 100
= 0.8 (or 80%)

Accuracy score is a simple and intuitive metric, but it's important to consider its limitations and use it in conjunction with other evaluation methods for a complete understanding of your model's performance. Check Precision, Recall, F1-Score and Confusion Matrix.

#### What is a classification problem?

A classification problem in machine learning is a task where the goal is to predict the category (or class) that a new data point belongs to. It's essentially about sorting things into predefined groups. Different types include Binary, Multi-Class and Multi-Label.

#### What is hyperparameter tuning?

Hyperparameter tuning is a crucial step in the machine learning workflow. It helps you fine-tune your model and unlock its full potential.
Imagine hyperparameters as the dials on a machine learning model. They influence how the model learns from data and makes predictions. This can significantly impact the accuracy and generalizability of your model.

Incase of SVM, pay attention to C, Gamma and Kernels.

#### What is Morphological Gradient?

A morphological gradient is an image processing technique that finds edges and boundaries by highlighting the difference between a dilated and eroded version of the original image.

Imagine inflating a balloon (dilation) and then deflating it (erosion). The leftover bulge is the edge, which is what the morphological gradient emphasizes. This makes it useful for tasks like edge detection, segmentation, and feature extraction.
Ex: cv2.morphologyEx(), skimage.morphology.binary_dilation,

#### What are HOG, SURF?

HOG and SURF are like detectives in the image processing world, searching for clues (features) to identify objects.

HOG focuses on the local shape and edges within an image. It breaks the image into grids, analyzes the direction of edges in each area, and builds a profile to recognize the object's overall form.

SURF is all about finding distinctive points (keypoints) in the image that won't change even if the object is rotated or zoomed. It identifies these keypoints and their surroundings, creating a unique fingerprint for each one.

Use HOG if: You're primarily interested in object shape and appearance, and robustness to illumination changes is important. (e.g., pedestrian detection)
Use SURF if: You need a fast and efficient way to identify keypoints in images that are invariant to scale and rotation. (e.g., object recognition in various poses)

#### How to save the model?

One can make use of libs like Pickle, Joblib, TF..

Though Pickle is simple, go to solution for PoCs, few concerns:

- Security concerns: Not recommended for models used in production due to potential security vulnerabilities when loading from untrusted sources.
- Limited portability: The pickled model might not be compatible with different Python versions or environments without additional considerations.

Alternatives to Pickle:

- Joblib: A popular library specifically designed for saving and loading machine learning models in Python. It addresses some security concerns and offers additional functionalities like model compression.
- TensorFlow SavedModel: If you're using TensorFlow, its built-in SavedModel format is a good option for saving and loading models in a portable way.
- ONNX: A vendor-neutral format for representing machine learning models, allowing for deployment across different frameworks.

## Acknowledgements

- https://scikit-learn.org/
- https://scikit-image.org/
- https://github.com/imRishabhGupta/Indian-Sign-Language-Recognition

## Contact

For any queries, please send an email (id on github profile)

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Screenshots

![Logo](https://github.com/AKGanesh/Indian-Sign-Language-Prediction/blob/main/huggingfaceISL.png)
![Logo](https://github.com/AKGanesh/Indian-Sign-Language-Prediction/blob/main/predict_proba.png)
