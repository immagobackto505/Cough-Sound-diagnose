import numpy as np
import streamlit as st
import pandas as pd


def  output(prediction,predicted_index):
    value = None
    info = ''
    class_names = ["Bronchitis", "Covid-19", "Healthy", "Pnuemonai"]
    class_index = predicted_index[0]
    if class_index == 0:
        value =  class_names[0]
        info = "Bronchitis is an inflammation of the lining of your bronchial tubes, which carry air to and from your lungs. People who have bronchitis often cough up thickened mucus, which can be discolored. Bronchitis may be either acute or chronic."
    elif class_index == 1:   
        value =  class_names[1]
        info = "Coronavirus disease (COVID-19) is an infectious disease caused by the SARS-CoV-2 virus. The disease was first identified in December 2019 in Wuhan, the capital of China's Hubei province, and has since spread globally, resulting in the ongoing 2019–20 coronavirus pandemic."
    elif class_index == 2:
        value =  class_names[2]
        info = "yaay! you are healthy"
    elif class_index == 3:
        value =  class_names[3]
        info = "Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus (purulent material), causing cough with phlegm or pus, fever, chills, and difficulty breathing. A variety of organisms, including bacteria, viruses and fungi, can cause pneumonia."
    
    
    st.write(f"Predictions: {value}")
    st.write(f"Note: {info}")
    
     # Convert predictions to percentages and round up
    prediction_percentages = np.ceil(prediction * 100)
    
    # Display the percentage values for each class in a table
    data = {
        "Class [diagnose]": class_names,
        "Percentage [%]": prediction_percentages[0]
    }
    df = pd.DataFrame(data)
    st.write("Prediction Percentages:")
    st.table(df)