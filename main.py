import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from utils import preprocess_data, evaluate_model, save_model
from logger import logger

# Load the dataset
data = pd.read_csv('Customer-Churn-Records.csv')

# Preprocessing
data = data.drop(columns=['RowNumber', 'CustomerId', 'Surname'])
categorical_columns = ['Geography', 'Gender', 'Card Type']
data, label_encoders = preprocess_data(data, categorical_columns)

# Splitting data into features and target
X = data.drop(columns=['Exited'])
y = data['Exited']

# Splitting into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Scaling the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Train the Random Forest Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
val_accuracy, val_f1, val_precision, val_recall = evaluate_model(model, X_val, y_val)
test_accuracy, test_f1, test_precision, test_recall = evaluate_model(model, X_test, y_test)

logger.info(f"Validation Accuracy: {val_accuracy:.2f}")
logger.info(f"Validation F1 Score: {val_f1:.2f}")
logger.info(f"Validation Precision: {val_precision:.2f}")
logger.info(f"Validation Recall: {val_recall:.2f}")

logger.info(f"Test Accuracy: {test_accuracy:.2f}")
logger.info(f"Test F1 Score: {test_f1:.2f}")
logger.info(f"Test Precision: {test_precision:.2f}")
logger.info(f"Test Recall: {test_recall:.2f}")

# Save the model
save_model(model, label_encoders, scaler)
