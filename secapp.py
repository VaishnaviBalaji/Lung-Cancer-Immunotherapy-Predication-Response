import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,precision_score,recall_score
from sklearn.preprocessing import StandardScaler
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors
import urllib.request
from io import BytesIO

import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

import os
from dotenv import load_dotenv
import hashlib
import time

# Load environment variables
load_dotenv()

# Set up authentication
def check_password():
    def password_entered():
        if st.session_state["password"] == os.getenv("APP_PASSWORD"):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input(
            "Enter the password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        st.text_input(
            "Enter the password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        return True

# Load the data
if check_password():



# Check if cache_data is available, otherwise use the older cache decorator
    if hasattr(st, 'cache_data'):
        cache_decorator = st.cache_data
    else:
        cache_decorator = st.cache

    @cache_decorator
    def load_data():
        return pd.read_csv('C:/Users/44755/Downloads/Lung Cancer/data.csv')


    df = load_data()
    def preprocess_data(df):
    # Handle missing values
        df['PDL1'] = pd.to_numeric(df['PDL1'], errors='coerce')
        df['PDL1'].fillna(df['PDL1'].median(), inplace=True)
    
    # Standardize Immunotherapy Response
        def process_response(x):
            if isinstance(x, str) and x.strip():
                parts = x.split()
                return parts[-1] if parts else np.nan
            return np.nan

        df['Immunotherapy Response'] = df['Immunotherapy Response'].apply(process_response)
        response_mapping = {'PR': 1, 'SD': 1, 'PD': 0}
        df['Immunotherapy Response'] = df['Immunotherapy Response'].map(response_mapping)
        df = df.dropna(subset=['Immunotherapy Response'])
    
    # Encode categorical variables
        df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
        df['EAS Regimen'] = df['EAS Regimen'].map({'Pembrolizumab': 0, 'Atezolizumab': 1, 'Nivolumab': 2})
        df['Intent'] = df['Intent'].map({'PALLIATIVE': 0, 'ADJUVANT': 1})
        df['Histology Type'] = df['Histology Type'].apply(lambda x: 0 if isinstance(x, str) and 'Squamous' in x else 1)
    
        # Convert T, N, M to numeric
        for col in ['T', 'N', 'M']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
        # Standardize numerical features
        scaler = StandardScaler()
        numerical_cols = ['PDL1', 'T', 'N', 'M', 'EAS Performance status']
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols].fillna(df[numerical_cols].mean()))
    
        return df
    df_processed = preprocess_data(df)    

    def create_pdf(input_data, prediction, probability, feature_importance, shap_values, input_df):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        # Add logo from URL
        logo_url = "https://images.squarespace-cdn.com/content/v1/5da2ef343b1ad128da926c53/1571314611701-SGDM0RAIYQQSKRKOHG32/curenetics_logo_t.png?format=1500w"
        logo_data = urllib.request.urlopen(logo_url).read()
        logo_img = Image(BytesIO(logo_data), width=2*inch, height=1*inch)
        
        
        # Create a table for logo alignment
        logo_table = Table([[logo_img]], colWidths=[6*inch])
        logo_table.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'RIGHT')]))
        story.append(logo_table)
        story.append(Spacer(1, 12))

        # Add title
        story.append(Paragraph("Immunotherapy Response Prediction Report", styles['Title']))
        story.append(Spacer(1, 12))

        # Add input data
        story.append(Paragraph("Input Data:", styles['Heading2']))
        for key, value in input_data.items():
                story.append(Paragraph(f"{key}: {value}", styles['Normal']))
        story.append(Spacer(1, 12))
        # Add prediction results
        story.append(Paragraph("Prediction Results:", styles['Heading2']))
        story.append(Paragraph(f"Predicted Response: {'Positive' if prediction[0] == 1 else 'Negative'}", styles['Normal']))
        story.append(Paragraph(f"Probability of Positive Response: {int(probability[0][1] * 100)}%", styles['Normal']))
        story.append(Spacer(1, 12))

        # Add feature importance plot
        story.append(Paragraph("Feature Importance:", styles['Heading2']))
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax)
        plt.title('Feature Importance')
        plt.tight_layout()
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        story.append(Image(img_buffer, width=450, height=300))
        plt.close()
        story.append(Spacer(1, 12))
        # Add feature importance summary
        top_features = feature_importance.head(3)['feature'].tolist()
        feature_summary = f"The top three most important features for this prediction are: {', '.join(top_features)}. These features have the strongest influence on the model's decision-making process."
        story.append(Paragraph(feature_summary, styles['Normal']))
        story.append(Spacer(1, 12))


        # Add SHAP summary plot
        story.append(Paragraph("SHAP Feature Importance:", styles['Heading2']))
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values[:, :, 1], input_df, plot_type="bar", show=False)
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        story.append(Image(img_buffer, width=450, height=300))
        plt.close()
        story.append(Spacer(1, 12))
        story.append(Paragraph("SHAP Analysis:", styles['Heading2']))

        # Debug information
        #story.append(Paragraph(f"SHAP values shape: {shap_values.shape}", styles['Normal']))
        #story.append(Paragraph(f"Input DataFrame shape: {input_df.shape}", styles['Normal']))
        #story.append(Paragraph(f"Input DataFrame columns: {', '.join(input_df.columns)}", styles['Normal']))

        try:
        # Extract SHAP values for the positive class (index 1)
                shap_values_array = shap_values.values[:, :, 1]
                shap_values_mean = np.abs(shap_values_array).mean(0)
        
                if len(shap_values_mean) == len(input_df.columns):
                    top_shap_features = pd.DataFrame({'feature': input_df.columns, 'importance': shap_values_mean})
                    top_shap_features = top_shap_features.sort_values('importance', ascending=False)
                    top_shap_features = top_shap_features.head(3)['feature'].tolist()
                
                    shap_summary = f"According to SHAP analysis, the top three features with the highest impact on the model's output are: {', '.join(top_shap_features)}. These features contribute most significantly to pushing the prediction away from the base value."
                else:
                    shap_summary = "Unable to determine top SHAP features due to shape mismatch between SHAP values and input features."
        except Exception as e:
                shap_summary = f"An error occurred while processing SHAP values: {str(e)}"

        story.append(Paragraph(shap_summary, styles['Normal']))
        story.append(Spacer(1, 12))
        

        # Add SHAP waterfall plot
        story.append(Paragraph("SHAP Waterfall Plot:", styles['Heading2']))
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_values[0, :, 1], show=False)
        plt.title('SHAP Waterfall Plot (Positive Class)')
        plt.tight_layout()
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        story.append(Image(img_buffer, width=450, height=300))
        plt.close()


        
        doc.build(story)
        buffer.seek(0)
        return buffer
def create_pdf(input_data, prediction, probability, feature_importance, shap_values, input_df):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
     # Add logo from URL
    logo_url = "https://images.squarespace-cdn.com/content/v1/5da2ef343b1ad128da926c53/1571314611701-SGDM0RAIYQQSKRKOHG32/curenetics_logo_t.png?format=1500w"
    logo_data = urllib.request.urlopen(logo_url).read()
    logo_img = Image(BytesIO(logo_data), width=2*inch, height=1*inch)
    
    
    # Create a table for logo alignment
    logo_table = Table([[logo_img]], colWidths=[6*inch])
    logo_table.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'RIGHT')]))
    story.append(logo_table)
    story.append(Spacer(1, 12))

    # Add title
    story.append(Paragraph("Immunotherapy Response Prediction Report", styles['Title']))
    story.append(Spacer(1, 12))

    # Add input data
    story.append(Paragraph("Input Data:", styles['Heading2']))
    for key, value in input_data.items():
        story.append(Paragraph(f"{key}: {value}", styles['Normal']))
    story.append(Spacer(1, 12))
    # Add prediction results
    story.append(Paragraph("Prediction Results:", styles['Heading2']))
    story.append(Paragraph(f"Predicted Response: {'Positive' if prediction[0] == 1 else 'Negative'}", styles['Normal']))
    story.append(Paragraph(f"Probability of Positive Response: {int(probability[0][1] * 100)}%", styles['Normal']))
    story.append(Spacer(1, 12))

    # Add feature importance plot
    story.append(Paragraph("Feature Importance:", styles['Heading2']))
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax)
    plt.title('Feature Importance')
    plt.tight_layout()
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    story.append(Image(img_buffer, width=450, height=300))
    plt.close()
    story.append(Spacer(1, 12))
    # Add feature importance summary
    top_features = feature_importance.head(3)['feature'].tolist()
    feature_summary = f"The top three most important features for this prediction are: {', '.join(top_features)}. These features have the strongest influence on the model's decision-making process."
    story.append(Paragraph(feature_summary, styles['Normal']))
    story.append(Spacer(1, 12))


    # Add SHAP summary plot
    story.append(Paragraph("SHAP Feature Importance:", styles['Heading2']))
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values[:, :, 1], input_df, plot_type="bar", show=False)
    plt.title('SHAP Feature Importance')
    plt.tight_layout()
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    story.append(Image(img_buffer, width=450, height=300))
    plt.close()
    story.append(Spacer(1, 12))
    story.append(Paragraph("SHAP Analysis:", styles['Heading2']))

    # Debug information
    #story.append(Paragraph(f"SHAP values shape: {shap_values.shape}", styles['Normal']))
    #story.append(Paragraph(f"Input DataFrame shape: {input_df.shape}", styles['Normal']))
    #story.append(Paragraph(f"Input DataFrame columns: {', '.join(input_df.columns)}", styles['Normal']))

    try:
    # Extract SHAP values for the positive class (index 1)
        shap_values_array = shap_values.values[:, :, 1]
        shap_values_mean = np.abs(shap_values_array).mean(0)
    
        if len(shap_values_mean) == len(input_df.columns):
            top_shap_features = pd.DataFrame({'feature': input_df.columns, 'importance': shap_values_mean})
            top_shap_features = top_shap_features.sort_values('importance', ascending=False)
            top_shap_features = top_shap_features.head(3)['feature'].tolist()
        
            shap_summary = f"According to SHAP analysis, the top three features with the highest impact on the model's output are: {', '.join(top_shap_features)}. These features contribute most significantly to pushing the prediction away from the base value."
        else:
            shap_summary = "Unable to determine top SHAP features due to shape mismatch between SHAP values and input features."
    except Exception as e:
        shap_summary = f"An error occurred while processing SHAP values: {str(e)}"

    story.append(Paragraph(shap_summary, styles['Normal']))
    story.append(Spacer(1, 12))
   

    # Add SHAP waterfall plot
    story.append(Paragraph("SHAP Waterfall Plot:", styles['Heading2']))
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0, :, 1], show=False)
    plt.title('SHAP Waterfall Plot (Positive Class)')
    plt.tight_layout()
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    story.append(Image(img_buffer, width=450, height=300))
    plt.close()


    
    doc.build(story)
    buffer.seek(0)
    return buffer

# Implement rate limiting
class RateLimiter:
    def __init__(self, max_calls, time_frame):
        self.max_calls = max_calls
        self.time_frame = time_frame
        self.calls = []

    def __call__(self):
        now = time.time()
        self.calls = [call for call in self.calls if call > now - self.time_frame]
        if len(self.calls) >= self.max_calls:
            return False
        self.calls.append(now)
        return True
    

# Create a rate limiter that allows 5 calls per minute
rate_limiter = RateLimiter(5, 60)

# Streamlit app
st.title('Immunotherapy Response Prediction')


   
    
    




# Split the data
X = df_processed.drop(['ID', 'Immunotherapy Response', 'ICD code'], axis=1)
y = df_processed['Immunotherapy Response']

# Train a simple model (Random Forest for demonstration)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# SHAP explainer
explainer = shap.TreeExplainer(model)
# Input form
st.header('Patient Information')
pdl1 = st.selectbox('PDL1 (%)', [20, 30, 50])
gender = st.selectbox('Gender', ['Male', 'Female'])
regimen = st.selectbox('Regimen', ['Pembrolizumab', 'Atezolizumab'])
intent = st.selectbox('Intent', ['Palliative', 'Adjuvant'])
t = st.number_input('T', min_value=0, max_value=4, value=0)
n = st.number_input('N', min_value=0, max_value=3, value=0)
m = st.number_input('M', min_value=0, max_value=1, value=0)
performance_status = st.selectbox('Performance Status', [0, 1])
histology_type = st.selectbox('Histology Type', ['Squamous', 'Adenocarcinoma'])

# Create a dictionary of input data
input_data = {
    'PDL1': pdl1,
    'Gender': 0 if gender == 'Male' else 1,
    'EAS Regimen': 0 if regimen == 'Pembrolizumab' else (1 if regimen == 'Atezolizumab' else 2),
    'Intent': 0 if intent == 'Palliative' else 1,
    'T': t,
    'N': n,
    'M': m,
    'EAS Performance status': performance_status,
    'Histology Type': 0 if histology_type == 'Squamous' else 1
}



# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Make prediction when button is clicked
if st.button('Predict'):
    if rate_limiter():
       prediction = model.predict(input_df)
       probability = model.predict_proba(input_df)
    
       st.header('Prediction Results')
       st.write(f"Predicted Response: {'Positive' if prediction[0] == 1 else 'Negative'}")
       st.write(f"Probability of Positive Response: {probability[0][1]:.2f}")
       # Generate and display feature importance
       st.header('Feature Importance')
       feature_importance = pd.DataFrame({'feature': input_df.columns, 'importance': model.feature_importances_})
       feature_importance = feature_importance.sort_values('importance', ascending=False)

       fig, ax = plt.subplots(figsize=(10, 6))
       sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax)
       plt.title('Feature Importance')
       st.pyplot(fig)

       st.write("This plot shows the importance of each feature in the model's decision-making process.")
       # SHAP values explanation
       st.header('SHAP Explanation')
       shap_values = explainer(input_df)
    
       st.write("SHAP (SHapley Additive exPlanations) values show how each feature contributes to pushing the model output from the base value (the average model output over the training dataset) to the model output for this specific prediction.")
    
       fig, ax = plt.subplots(figsize=(10, 6))
       shap.summary_plot(shap_values[:, :, 1], input_df, plot_type="bar", show=False)
       plt.title('SHAP Feature Importance')
       st.pyplot(fig)
    
       st.write("This bar plot shows the average impact of each feature on the model output. Features are ranked by importance, with the most important features at the top.")
    
       fig, ax = plt.subplots(figsize=(10, 6))
       shap.plots.waterfall(shap_values[0, :, 1], show=False)
       plt.title('SHAP Waterfall Plot')
       st.pyplot(fig)
    
       st.write("This waterfall plot shows how each feature pushes the prediction from the base value (average model output) to the final predicted value for this specific input.")
       pdf = create_pdf(input_data, prediction, probability, feature_importance, shap_values, input_df)
    else:
        st.error("Rate limit exceeded. Please try again later.")
    
    # Sanitize user inputs before using them
    def sanitize_input(input_string):
        # Remove any potentially harmful characters
        return ''.join(char for char in input_string if char.isalnum() or char in [' ', '-', '_'])

    # Use sanitized inputs when creating the PDF
    sanitized_input_data = {k: sanitize_input(str(v)) for k, v in input_data.items()}
    pdf = create_pdf(sanitized_input_data, prediction, probability, feature_importance, shap_values, input_df)

    # Add download button for PDF
    st.download_button(
        label="Download Prediction Report",
        data=pdf,
        file_name="prediction_report.pdf",
        mime="application/pdf"
    )
# Or add logo to the main area
st.sidebar.image("https://images.squarespace-cdn.com/content/v1/5da2ef343b1ad128da926c53/1571314611701-SGDM0RAIYQQSKRKOHG32/curenetics_logo_t.png?format=1500w", width=300)

st.sidebar.header('Instructions')
st.sidebar.write("""
1. Enter the patient's information in the form.
2. Click the 'Predict' button to get the prediction and explanation.
3. The prediction result shows whether the response is likely to be positive or negative.
4. The SHAP plot explains which features contributed most to the prediction.
""")

# Add information about the model
st.sidebar.header('Model Information')

st.sidebar.write("""
This model predicts the likelihood of a positive response to immunotherapy in lung cancer patients.

- Model type: Random Forest Classifier
- Features used: PDL1, Gender, Regimen, Intent, TNM staging, Performance status, and Histology Type

Please note that this model should be used as a supportive tool and not as a replacement for clinical judgment.
""")
