import pandas as pd
from tkinter import Tk, Label, Entry, Button, messagebox
from utils import load_model
from logger import logger

# Load the pre-trained model
model_data = load_model()
model = model_data['model']
label_encoders = model_data['label_encoders']
scaler = model_data['scaler']

# GUI for Predictions
def make_prediction():
    try:
        inputs = {
            'CreditScore': float(entry_creditscore.get()),
            'Geography': label_encoders['Geography'].transform([entry_geography.get()])[0]
                         if entry_geography.get() in label_encoders['Geography'].classes_ else label_encoders['Geography'].transform(['unknown'])[0],
            'Gender': label_encoders['Gender'].transform([entry_gender.get()])[0]
                      if entry_gender.get() in label_encoders['Gender'].classes_ else label_encoders['Gender'].transform(['unknown'])[0],
            'Age': int(entry_age.get()),
            'Tenure': int(entry_tenure.get()),
            'Balance': float(entry_balance.get()),
            'NumOfProducts': int(entry_numofproducts.get()),
            'HasCrCard': int(entry_hascrcard.get()),
            'IsActiveMember': int(entry_isactivemember.get()),
            'EstimatedSalary': float(entry_estimatedsalary.get()),
            'Complain': int(entry_complain.get()),
            'Satisfaction Score': int(entry_satisfaction_score.get()),
            'Card Type': label_encoders['Card Type'].transform([entry_card_type.get()])[0]
                         if entry_card_type.get() in label_encoders['Card Type'].classes_ else label_encoders['Card Type'].transform(['unknown'])[0],
            'Point Earned': int(entry_point_earned.get())
        }

        # Preprocessing inputs for prediction
        df_inputs = pd.DataFrame([inputs.values()], columns=inputs.keys())
        df_inputs = scaler.transform(df_inputs)

        # Making prediction
        prediction = model.predict(df_inputs)[0]
        result = "Churn" if prediction == 1 else "No Churn"
        messagebox.showinfo("Prediction", f"The customer is predicted to: {result}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        messagebox.showerror("Error", f"An error occurred: {e}")

# GUI setup
root = Tk()
root.title("Churn Prediction")
root.geometry("500x500")  # Set window size

# Creating entry fields and showing example values
labels = [
    ('CreditScore', 'Example: 650'),
    ('Geography', 'Example: France (France, Germany, Spain)'),
    ('Gender', 'Example: Male (Male, Female)'),
    ('Age', 'Example: 30'),
    ('Tenure', 'Example: 5'),
    ('Balance', 'Example: 10000.00'),
    ('NumOfProducts', 'Example: 2'),
    ('HasCrCard', 'Example: 1 (0, 1)'),
    ('IsActiveMember', 'Example: 1 (0, 1)'),
    ('EstimatedSalary', 'Example: 50000.00'),
    ('Complain', 'Example: 0 (0, 1)'),
    ('Satisfaction Score', 'Example: 3 (1-5)'),
    ('Card Type', 'Example: GOLD (GOLD, SILVER, PLATINUM, BLUE)'),
    ('Point Earned', 'Example: 200')
]
entries = {}

for idx, (label, example) in enumerate(labels):
    lbl = Label(root, text=label)
    lbl.grid(row=idx, column=0, padx=10, pady=5, sticky='W')
    entry = Entry(root)
    entry.grid(row=idx, column=1, padx=10, pady=5)
    example_lbl = Label(root, text=example, fg="grey")
    example_lbl.grid(row=idx, column=2, padx=10, pady=5, sticky='W')
    entries[label] = entry

# Assigning each entry to a variable for easier access
entry_creditscore = entries['CreditScore']
entry_geography = entries['Geography']
entry_gender = entries['Gender']
entry_age = entries['Age']
entry_tenure = entries['Tenure']
entry_balance = entries['Balance']
entry_numofproducts = entries['NumOfProducts']
entry_hascrcard = entries['HasCrCard']
entry_isactivemember = entries['IsActiveMember']
entry_estimatedsalary = entries['EstimatedSalary']
entry_complain = entries['Complain']
entry_satisfaction_score = entries['Satisfaction Score']
entry_card_type = entries['Card Type']
entry_point_earned = entries['Point Earned']

# Prediction button
btn_predict = Button(root, text="Predict Churn", command=make_prediction)
btn_predict.grid(row=len(labels), column=0, columnspan=3, pady=20)

# Running the GUI
root.mainloop()
