
# Product Classification Prototype with FastAPI and TensorFlow

This repository contains a prototype machine learning model developed using FastAPI and TensorFlow for classifying dummy products into predefined categories. The objective of this project is to demonstrate the ability to quickly prototype a solution for product classification and showcase basic machine learning principles.

## Task Overview

The task requirements include:

1.  **Data Generation**: A small dataset of dummy products with attributes such as name, description, price, and category was created.
2.  **Data Preprocessing**: Basic preprocessing steps were performed, including tokenization of text attributes and encoding of categorical attributes.
3.  **Model Development**: A simple text classification model was developed using TensorFlow and Python, integrated with FastAPI for seamless deployment and interaction.

## Prerequisites

To run the code, you need:

-   Python 3.9 and above.
-   TensorFlow.
-   Pandas.
-   NumPy.

## Running the Code

Here's how to get started:

1.  Clone this repository to your local machine.
2.  Install the required dependencies using `pip install -r requirements.txt`.
3.  Generate the dataset and preprocess the data by running backend utils `python data_generation.py`.
4.  Train the classification model by running scripts `python models.py`.
5.  Once the model is trained, predictions can be made using the FastAPI server by running `uvicorn app:app -- reload` and accessing the `/predict/` endpoint.

## Approach Overview

### Data Generation

-   A small dataset of dummy products with essential attributes was generated to facilitate faster processing.
-   The dataset was intentionally kept small for quicker prototyping and experimentation.

### Data Preprocessing

-   Basic preprocessing steps were applied to prepare the data for model training.
-   Text attributes were tokenized using standard techniques, and categorical attributes were encoded for model compatibility.

### Model Development

-   The model architecture incorporates embedding layers, LSTM layers, and dense layers for text classification.
-   Training involved splitting the dataset, training the model, and evaluating its performance using standard evaluation metrics.

## Conclusion

This prototype showcases the ability to rapidly develop a machine learning model for product classification using FastAPI and TensorFlow. While it serves as a solid foundation, further optimization and refinement may be necessary for real-world deployment.

**Note**: This project adheres to flake8/pep8 linting standards.