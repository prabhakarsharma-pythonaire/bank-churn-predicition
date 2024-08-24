# Churn Prediction Project


## Working of Project
![Output1](https://github.com/prabhakarsharma-pythonaire/bank-churn-predicition/blob/main/Screenshot%202024-08-25%20002441.png)
![Output2](https://github.com/prabhakarsharma-pythonaire/bank-churn-predicition/blob/main/Screenshot%202024-08-25%20002610.png)
This project is designed to predict customer churn using a machine learning model (Random Forest Classifier). It involves training a model with customer data, evaluating the model, and providing a graphical user interface (GUI) to make predictions based on user inputs.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Requirements](#requirements)
3. [Setup and Installation](#setup-and-installation)
4. [How to Run](#how-to-run)
5. [Usage](#usage)
6. [Files](#files)
7. [License](#license)

## Project Structure


- `app.py`: Contains the code for the GUI that loads the pre-trained model and allows for making predictions.
- `logger.py`: Handles logging configurations and logging functions.
- `main.py`: Responsible for training the machine learning model and saving it for future use.
- `utils.py`: Contains utility functions for data preprocessing, encoding, evaluating the model, and saving/loading the model.
- `Customer-Churn-Records.csv`: The dataset used for training the model.
- `model.pkl`: A serialized file to store the trained model, scaler, and label encoders (created after running `main.py`).
- `churn_prediction.log`: Log file to store log messages.
- `README.md`: Documentation file for the project.

## Requirements

- Python 3.6 or higher
- Required Python packages are listed in `requirements.txt`:

    ```
    pandas
    numpy
    scikit-learn
    joblib
    tkinter
    ```

## Setup and Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/churn-prediction.git
   cd churn-prediction

## env
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt

## Run 
main.py
app.py