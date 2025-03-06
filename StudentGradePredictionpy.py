import joblib
import gradio as gr
import numpy as np
import os

# First, let's verify the model file exists and load it
model_path = r'C:\Users\NK\Desktop\MODELS\random_forest_model_NF.pkl'

# Check if model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found. Please ensure the file is in the correct directory.")

try:
    # Load the model
    model = joblib.load(model_path)
except Exception as e:
    raise Exception(f"Error loading the model: {str(e)}")

def predict_g3(school, sex, age, address, Medu, Fedu, traveltime, studytime, failures, absences):
    """
    Make predictions using the pre-loaded model
    """
    # Map categorical features to numerical values
    school_mapping = {'GP': 0, 'MS': 1}
    sex_mapping = {'M': 0, 'F': 1}
    address_mapping = {'U': 0, 'R': 1}
    
    # Convert categorical values to numerical
    school = school_mapping[school]
    sex = sex_mapping[sex]
    address = address_mapping[address]
    
    # Prepare the feature array
    features = np.array([[school, sex, age, address, Medu, Fedu, traveltime, studytime, failures, absences]])
    
    # Make prediction
    prediction = model.predict(features)
    
    return round(prediction[0], 2)

# Create Gradio interface
demo = gr.Interface(
    fn=predict_g3,
    inputs=[
        gr.Radio(['GP', 'MS'], label="School"),
        gr.Radio(['M', 'F'], label="Sex"),
        gr.Slider(15, 22, step=1, label="Age"),
        gr.Radio(['U', 'R'], label="Address"),
        gr.Slider(0, 4, step=1, label="Mother's Education (0-4)"),
        gr.Slider(0, 4, step=1, label="Father's Education (0-4)"),
        gr.Slider(1, 4, step=1, label="Travel Time (1-4)"),
        gr.Slider(1, 4, step=1, label="Study Time (1-4)"),
        gr.Slider(0, 3, step=1, label="Number of Failures"),
        gr.Slider(0, 75, step=1, label="Number of Absences"),
    ],
    outputs=gr.Number(label="Predicted Final Grade (G3)"),
    title="Student Final Grade Predictor",
    description="Enter student characteristics to predict their final grade."
)

if __name__ == "__main__":
    demo.launch()