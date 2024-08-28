import streamlit as st
def display_sidebar():
    st.sidebar.image("https://images.squarespace-cdn.com/content/v1/5da2ef343b1ad128da926c53/1571314611701-SGDM0RAIYQQSKRKOHG32/curenetics_logo_t.png?format=1500w", width=300)
    
    st.sidebar.header('Instructions')
    st.sidebar.write("""
    1. Enter the patient's information in the form.
    2. Click the 'Predict' button to get the prediction and explanation.
    3. The prediction result shows whether the response is likely to be positive or negative.
    4. The SHAP plot explains which features contributed most to the prediction.
    """)
    st.sidebar.header('Model Information')
    st.sidebar.write("""
    This model predicts the likelihood of a positive response to immunotherapy in lung cancer patients.
    - Model type: Random Forest Classifier
    - Features used: PDL1, Gender, Regimen, Intent, TNM staging, Performance status, and Histology Type
    Please note that this model should be used as a supportive tool and not as a replacement for clinical judgment.
    """)