import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import fbeta_score, confusion_matrix
from sklearn.metrics import make_scorer
from streamlit_option_menu import option_menu

# Set Streamlit page configuration to wide layout
st.set_page_config(page_title="Auto Insurance Response Prediction", layout="wide")

# Load model and encoders
untreated_model = joblib.load('knn_tuned_model.pkl')
treated_model = joblib.load('pipe_tuned_pipeline.pkl')

# Load and preprocess data
def load_data():
    data = pd.read_csv('AutoInsurance.csv')
    return data

def load_new_obs():
    new_obs = pd.read_csv('new_obs_unseen_dummy3.csv')
    return new_obs

# Model Explanation with Interpretation and Hypothetical Cost Analysis
def explain_model(model, X_train):
    st.subheader('Model Explanation')
    st.markdown("""
    This section provides an interpretation of the model's performance and a hypothetical cost analysis based on the predicted outcomes.
    """)

    # Helper function to convert image to base64
    import base64

    def convert_image_to_base64(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()

    st.markdown(
        """
        <div style="text-align: center;">
            <img src="data:image/jpg;base64,{}" alt="ConfusionMatrix" width="600">
        </div>
        """.format(convert_image_to_base64("C:/Users/Hadi/Desktop/ConfusionMatrix.jpg")),
        unsafe_allow_html=True
    )

    st.markdown("""
    ## Interpretation and Hypothetical Cost Analysis

    If the model were used to filter potential customers to determine who would receive a "Yes" response, it would successfully predict 95% of the "Yes" responses (recall), and 90% of the "No" responses would be correctly filtered out. The precision of 61% indicates that when the model predicts a "Yes" response, it is correct 61% of the time, meaning that there is still a 39% chance of incorrectly predicting a "No" response as "Yes."
    """)

    st.markdown("""
    ### Hypothetical Cost Analysis:

    Given the classification report and confusion matrix, let's explore a cost scenario that accurately reflects the actual model's performance.

    **Assumptions:**

    - **Revenue gain from every 50 points in CLV:** $1
    - **Total Number of Customers:** 1462
    - **True Positives Data:** 199
    - **False Positives Data:** 128
    - **False Negatives Data::** 199
    """)

    st.markdown("""
    ### Without the Model [TP + FN]:
    - **Correctly Approached "Yes" Responses:** 209 (Total actual "Yes" responses)
    
    ### With the Model [TP + FP]:

    - **Predicted "Yes" Responses (True Positive):** 199
    - **Incorrectly Predicted "Yes" Responses (False Positives):** 128
                
    ### Revenue gain
    Gaining Revenue = model_revenue - revenue_without_model
    Gaining Revenue = 20474
                
    Percentage gaining revenue = (diff_rev/revenue_without_model)*100
    Percentage gaining revenue = 60.91%
    
    ### Summary of Revenue
    - **Without model (TP + FN): $33610**
    - **With model (TP + FP): $54084** 
    - **Difference:** $54084 - $33610 = **$20474**
    - **Percentage Revenue:** ($20474 / $33610) * 100 = **60.91%**

    Implementing the model resulted in a significant revenue increase, with total earnings rising from $33,610 (without the model) to $54,084 (with the model). This represents a difference of $20,474, translating to a 60.91% increase in revenue, highlighting the model's effectiveness in boosting financial performance.
    """)

    st.markdown("""
    ## Recommendations for Business and Project Model
    ------
    ### Business Recommendations

    1. **Targeted Strategy for Policyholder Growth:**
       - **Leverage the Model for Accurate Targeting:** Implement the KNN model to accurately identify and target customers who are likely to renew their policies. By focusing on these potential policyholders, the company can significantly increase the number of renewals, leading to substantial revenue growth.
       - **Maximize Policyholder Retention:** The model's ability to correctly predict renewals allows the company to enhance its retention efforts, ensuring that more customers choose to extend their policies. This increase in retained customers directly contributes to higher revenue.

    2. **Revenue Enhancement:**
       - **Achieve Substantial Revenue Gains:** The implementation of the model has demonstrated a significant revenue increase, with total earnings rising by 60.91%. This growth reflects the model's effectiveness in accurately identifying potential renewals and converting them into actual revenue.
       - **Focus on High-Impact Opportunities:** By concentrating efforts on customers identified by the model as likely to renew, the company can maximize revenue potential and achieve a higher return on investment.

    3. **Ongoing Model Optimization:**
       - **Continuous Model Refinement:** Regularly monitor and refine the model to ensure its ongoing accuracy in predicting policy renewals. As the model continues to evolve with new data, it will maintain its effectiveness in driving revenue growth.
       - **Adapt to Customer Behavior:** Stay responsive to changes in customer behavior by incorporating new insights into the model. This adaptability will help sustain revenue increases over time by ensuring that the company continues to target the most promising policyholders.

    ### Project Model Recommendations

    1. **Model Implementation:**
       - **Deploy the Model in Production ([Notebook](https://github.com/PurwadhikaDev/IgniteGroup_DTI_01_FinalProject/blob/main/Modeling_Auto%20Insurance_Ignite.ipynb)):** Integrate the KNN model into your customer relationship management (CRM) system or marketing automation platform. Ensure seamless integration so that predictions can be used to inform real-time marketing decisions.
       - **Conduct A/B Testing ([Notebook](https://github.com/PurwadhikaDev/IgniteGroup_DTI_01_FinalProject/blob/main/Prediction%20AB%20Test_Auto%20Insurance_Ignite.ipynb)):** Perform A/B testing to compare the performance of the model-based approach against traditional marketing methods. This will provide empirical evidence of the model’s effectiveness and help justify its use.

    2. **Data Management:**
       - **Maintain Data Quality:** Ensure high-quality and up-to-date data for the model to function effectively. Regularly clean and preprocess the data to avoid inaccuracies that could affect model performance.
       - **Expand Data Sources:** Consider incorporating additional data sources to enhance the model’s predictions. This could include behavioral data, social media interactions, or external market trends.

    3. **Evaluation and Metrics:**
       - **Track Key Performance Indicators (KPIs):** Monitor KPIs such as precision, recall, and overall accuracy to assess the model’s impact on marketing outcomes. Evaluate these metrics regularly to ensure the model continues to meet business objectives.
       - **Analyze Model Impact:** Use detailed analyses and reports to understand the model’s impact on marketing efficiency and customer engagement. Share insights with stakeholders to demonstrate the value and drive data-driven decision-making.

    4. **Risk Management:**
       - **Address False Positives:** Develop strategies to manage the impact of false positives (incorrect "Yes" predictions) and mitigate any potential negative effects on customer experience or resource allocation.
       - **Prepare for Model Limitations:** Acknowledge that no model is perfect. Be prepared for scenarios where the model’s predictions may not align with actual outcomes and have contingency plans in place.
    """)

def single_prediction(data):
    # Initialize session state variables if they do not exist
    if 'form_page' not in st.session_state:
        st.session_state['form_page'] = 'customer_segmentation'
    if 'input_df' not in st.session_state:
        st.session_state['input_df'] = None  # Initialize with None or an empty DataFrame if preferred
    if 'prediction_result' not in st.session_state:
        st.session_state['prediction_result'] = None

    def go_to_next_page(next_page):
        st.session_state['form_page'] = next_page

    if st.session_state['form_page'] == 'customer_segmentation':
        with st.form(key='customer_segmentation_form'):
            st.subheader('Customer Segmentation')
            customer_inputs = {}
            customer_inputs['Customer Lifetime Value'] = st.number_input(
                'Customer Lifetime Value', 
                min_value=float(data['Customer Lifetime Value'].min()), 
                max_value=float(data['Customer Lifetime Value'].max()), 
                value=float(data['Customer Lifetime Value'].mean())
            )
            customer_inputs['Gender'] = st.selectbox(
                'Gender', 
                options=['Male', 'Female']
            )
            customer_inputs['Education'] = st.selectbox(
                'Education', 
                options=['Bachelor', 'College', 'Doctor', 'High School or Below', 'Master']
            )
            customer_inputs['Marital Status'] = st.selectbox(
                'Marital Status', 
                options=['Married', 'Single', 'Divorced']
            )
            customer_inputs['EmploymentStatus'] = st.selectbox(
                'Employment Status', 
                options=['Employed', 'Unemployed', 'Medical Leave', 'Disabled', 'Retired']
            )
            customer_inputs['Income'] = st.number_input(
                'Income', 
                min_value=float(data['Income'].min()), 
                max_value=float(data['Income'].max()), 
                value=float(data['Income'].mean())
            )
            customer_inputs['State'] = st.selectbox(
                'State', 
                options=data['State'].unique()
            )
            customer_inputs['Location Code'] = st.selectbox(
                'Location Code', 
                options=['Urban', 'Suburban', 'Rural']
            )
            customer_inputs['Effective To Date'] = st.date_input(
                'Effective To Date', 
                value=pd.to_datetime(data['Effective To Date'].mode()[0])
            )

            # Add a submit button to the form
            next_button = st.form_submit_button(label='Next')
        
        if next_button:
            st.session_state['customer_inputs'] = customer_inputs
            go_to_next_page('insurance_details')

    elif st.session_state['form_page'] == 'insurance_details':
        with st.form(key='insurance_details_form'):
            st.subheader('Insurance Details')
            insurance_inputs = {}
            insurance_inputs['Coverage'] = st.selectbox(
                'Coverage', 
                options=['Basic', 'Extended', 'Premium']
            )
            insurance_inputs['Number of Open Complaints'] = st.number_input(
                'Number of Open Complaints', 
                min_value=float(data['Number of Open Complaints'].min()), 
                max_value=float(data['Number of Open Complaints'].max()), 
                value=float(data['Number of Open Complaints'].mean())
            )
            insurance_inputs['Number of Policies'] = st.number_input(
                'Number of Policies', 
                min_value=float(data['Number of Policies'].min()), 
                max_value=float(data['Number of Policies'].max()), 
                value=float(data['Number of Policies'].mean())
            )
            insurance_inputs['Monthly Premium Auto'] = st.number_input(
                'Monthly Premium Auto', 
                min_value=float(data['Monthly Premium Auto'].min()), 
                max_value=float(data['Monthly Premium Auto'].max()), 
                value=float(data['Monthly Premium Auto'].mean())
            )
            insurance_inputs['Months Since Last Claim'] = st.number_input(
                'Months Since Last Claim', 
                min_value=float(data['Months Since Last Claim'].min()), 
                max_value=float(data['Months Since Last Claim'].max()), 
                value=float(data['Months Since Last Claim'].mean())
            )
            insurance_inputs['Months Since Policy Inception'] = st.number_input(
                'Months Since Policy Inception', 
                min_value=float(data['Months Since Policy Inception'].min()), 
                max_value=float(data['Months Since Policy Inception'].max()), 
                value=float(data['Months Since Policy Inception'].mean())
            )

            # Add submit buttons to the form
            next_button = st.form_submit_button(label='Next')
            back_button = st.form_submit_button(label='Back')
        
        if next_button:
            st.session_state['insurance_inputs'] = insurance_inputs
            go_to_next_page('policy_information')
        if back_button:
            go_to_next_page('customer_segmentation')

    elif st.session_state['form_page'] == 'policy_information':
        with st.form(key='policy_information_form'):
            st.subheader('Policy Information')
            policy_inputs = {}
            policy_inputs['Policy Type'] = st.selectbox(
                'Policy Type', 
                options=['Personal Auto', 'Corporate Auto', 'Special Auto']
            )
            policy_inputs['Policy'] = st.selectbox(
                'Policy', 
                options=['L1', 'L2', 'L3']
            )
            policy_inputs['Renew Offer Type'] = st.selectbox(
                'Renew Offer Type', 
                options=['Offer1', 'Offer2', 'Offer3', 'Offer4']
            )
            policy_inputs['Sales Channel'] = st.selectbox(
                'Sales Channel', 
                options=['Agent', 'Call Center', 'Branch', 'Web']
            )
            policy_inputs['Total Claim Amount'] = st.number_input(
                'Total Claim Amount', 
                min_value=float(data['Total Claim Amount'].min()), 
                max_value=float(data['Total Claim Amount'].max()), 
                value=float(data['Total Claim Amount'].mean())
            )
            policy_inputs['Vehicle Class'] = st.selectbox(
                'Vehicle Class', 
                options=['Two-Door Car', 'Four-Door Car', 'SUV', 'Luxury SUV', 'Luxury Car', 'Sports Car']
            )
            policy_inputs['Vehicle Size'] = st.selectbox(
                'Vehicle Size', 
                options=['Small', 'Medsize', 'Large']
            )

            # Add submit buttons to the form
            predict_button = st.form_submit_button(label='Predict')
            back_button = st.form_submit_button(label='Back')

        if predict_button:
            st.session_state['policy_inputs'] = policy_inputs
            inputs = {**st.session_state['customer_inputs'], **st.session_state['insurance_inputs'], **st.session_state['policy_inputs']}
            
            # Exclude 'Effective To Date' and 'Customer' from input features
            # Instead of dropping, we simply don't include them in the dictionary
            relevant_inputs = {key: value for key, value in inputs.items() if key not in ['Effective To Date', 'Customer']}
            
            # Applying logarithmic transformations if not already applied
            relevant_inputs['CLV_log'] = np.log1p(relevant_inputs.pop('Customer Lifetime Value'))
            relevant_inputs['Income_Log'] = np.log1p(relevant_inputs.pop('Income'))
            relevant_inputs['TCA_Log'] = np.log1p(relevant_inputs.pop('Total Claim Amount'))

            # Prepare the input data in the expected format
            input_df = pd.DataFrame([relevant_inputs])

            # Store input_df in session state for later use
            st.session_state['input_df'] = input_df

            # Ensure correct data types and clean any non-numeric data
            input_df = input_df.apply(pd.to_numeric, errors='ignore')  # Directly apply numeric coercion if needed

            # Predict using the pre-trained pipeline
            prediction = treated_model.predict(input_df)[0]
            st.session_state['prediction_result'] = prediction
            go_to_next_page('results_visualization')

        if back_button:
            go_to_next_page('insurance_details')

    elif st.session_state['form_page'] == 'results_visualization':
        if st.session_state['input_df'] is not None and st.session_state['prediction_result'] is not None:
            prediction = st.session_state['prediction_result']

            # Display User Input Overview in Tables
            st.subheader('User Input Overview')

            # Create tables for each category of input
            if 'customer_inputs' in st.session_state:
                st.markdown("### Customer Segmentation")
                customer_inputs = st.session_state['customer_inputs']
                customer_df = pd.DataFrame.from_dict(customer_inputs, orient='index', columns=['Value'])
                st.table(customer_df)

            if 'insurance_inputs' in st.session_state:
                st.markdown("### Insurance Details")
                insurance_inputs = st.session_state['insurance_inputs']
                insurance_df = pd.DataFrame.from_dict(insurance_inputs, orient='index', columns=['Value'])
                st.table(insurance_df)

            if 'policy_inputs' in st.session_state:
                st.markdown("### Policy Information")
                policy_inputs = st.session_state['policy_inputs']
                policy_df = pd.DataFrame.from_dict(policy_inputs, orient='index', columns=['Value'])
                st.table(policy_df)
            
            # Create a table to show the prediction result
            st.subheader('Model Prediction')
            prediction_result = {"Output": ["Yes" if prediction == 1 else "No" if prediction == 0 else "Error"]}
            prediction_df = pd.DataFrame.from_dict(prediction_result, orient='index', columns=['Prediction'])
            st.table(prediction_df)

        else:
            st.write("Please complete the prediction process before viewing the results.")

        if st.button('Back'):
            go_to_next_page('policy_information')

# Batch prediction page
def batch_prediction():
    st.write("Please upload a CSV file for batch prediction.")
    uploaded_file = st.file_uploader("Upload CSV for batch prediction", type="csv")

    if uploaded_file is not None:
        batch_data = pd.read_csv(uploaded_file)
        
        # Drop unnecessary columns
        batch_data = batch_data.drop(columns=['Customer', 'Effective To Date', 'Unnamed: 0'], errors='ignore')
        
        # Mapping 'Yes'/'No' to 1/0 in the Response column (if applicable)
        if 'Response' in batch_data.columns:
            batch_data['Response'] = batch_data['Response'].map({'Yes': 1, 'No': 0}).astype(float)
        
        # Applying logarithmic transformation for prediction purposes
        if 'Customer Lifetime Value' in batch_data.columns:
            batch_data['CLV_log'] = np.log1p(batch_data['Customer Lifetime Value'])
        if 'Income' in batch_data.columns:
            batch_data['Income_Log'] = np.log1p(batch_data['Income'])
        if 'Total Claim Amount' in batch_data.columns:
            batch_data['TCA_Log'] = np.log1p(batch_data['Total Claim Amount'])
        
        # Prepare features for prediction by excluding original columns
        model_features = batch_data.drop(columns=['Customer Lifetime Value', 'Income', 'Total Claim Amount'], errors='ignore')

        if st.button('Predict'):
            # Perform predictions
            batch_predictions = treated_model.predict(model_features)
            
            # Prepare a DataFrame to show the results
            results_df = batch_data.copy()
            results_df['Prediction'] = batch_predictions

            # Map numeric predictions back to 'Yes'/'No'
            results_df['Prediction'] = results_df['Prediction'].map({1: 'Yes', 0: 'No'})

            # Add a count column for pivoting
            results_df['Count'] = 1

            # Create a pivot table that categorizes the data based on the prediction
            pivot_table = results_df.pivot_table(index=['Prediction'], values=['Count'], aggfunc='count')
            
            # Display the count of each prediction
            st.write("Summary of Batch Predictions (Pivot Table):")
            st.dataframe(pivot_table)
            
            # Exclude the log-transformed columns from the detailed view
            detailed_columns = [col for col in results_df.columns if col not in ['CLV_log', 'Income_Log', 'TCA_Log', 'Count']]
            
            # Prepare categorized data without log-transformed columns and without repeating the 'Prediction' column
            categorized_data = results_df[detailed_columns].groupby('Prediction').apply(lambda x: x.drop(columns=['Prediction']).reset_index(drop=True))
            
            st.write("Detailed Batch Predictions:")
            st.dataframe(categorized_data)
    else:
        st.write("Please upload a CSV file for batch prediction.")

# Main Streamlit app
def main():
    # Navigation
    selected_page = option_menu(
        menu_title="Auto Insurance Response Prediction", 
        options=["Home", "Single Prediction", "Batch Prediction", "Model Explanation"], 
        icons=["house", "robot", "archive", "info-circle"], 
        menu_icon="tools", 
        default_index=0, 
        orientation="horizontal"
    )
    
    st.session_state['page'] = selected_page

    # Load data
    data = load_data()
    new_obs = load_new_obs()

    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state['page'] = 'Home'

    # Page display
    if st.session_state['page'] == 'Home':
        st.title('Final Project Ignite Team')
        st.write('Welcome to the Auto Insurance Prediction App.')
        
        # Project Context
        st.subheader('Project Context')
        st.markdown("""
        This project aims to enhance the decision-making process in auto insurance through predictive analytics. Our model predicts whether customers will respond positively to different insurance offerings, helping the business optimize customer engagement and retention strategies.
        """)

        # Embed Tableau Dashboard
        tableau_html = """
        <div style='display: flex; justify-content: center; align-items: center;'>
            <div class='tableauPlaceholder' id='viz1724436836867' style='position: relative; width: 100%; max-width: 1000px;'>
                <noscript>
                    <a href='#'>
                        <img alt='Page 1' src='https://public.tableau.com/static/images/Au/AutoInsuranceDashboard_17241501812520/Page1/1_rss.png' style='border: none; width: 100%;' />
                    </a>
                </noscript>
                <object class='tableauViz' style='display:none; width: 100%;'>
                    <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> 
                    <param name='embed_code_version' value='3' />
                    <param name='site_root' value='' />
                    <param name='name' value='AutoInsuranceDashboard_17241501812520&#47;Page1' />
                    <param name='tabs' value='no' />
                    <param name='toolbar' value='yes' />
                    <param name='static_image' value='https://public.tableau.com/static/images/Au/AutoInsuranceDashboard_17241501812520/Page1/1.png' />
                    <param name='animate_transition' value='yes' />
                    <param name='display_static_image' value='yes' />
                    <param name='display_spinner' value='yes' />
                    <param name='display_overlay' value='yes' />
                    <param name='display_count' value='yes' />
                    <param name='language' value='en-US' />
                </object>
            </div>
        </div>
        <script type='text/javascript'>                    
            var divElement = document.getElementById('viz1724436836867');                    
            var vizElement = divElement.getElementsByTagName('object')[0];                    
            if ( divElement.offsetWidth > 800 ) { 
                vizElement.style.width='1000px';vizElement.style.height='850px';
            } else if ( divElement.offsetWidth > 500 ) { 
                vizElement.style.width='1000px';vizElement.style.height='850px';
            } else { 
                vizElement.style.width='100%';vizElement.style.height='2777px';
            }                     
            var scriptElement = document.createElement('script');                    
            scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    
            vizElement.parentNode.insertBefore(scriptElement, vizElement);                
        </script>
        """
        st.components.v1.html(tableau_html, height=850)

        # Data and Features
        st.subheader('Data and Features')
        st.markdown("""
        - ([EDA Notebook](https://github.com/PurwadhikaDev/IgniteGroup_DTI_01_FinalProject/blob/main/EDA_Auto%20Insurance_Ignite.ipynb))
        - ([Modeling Notebook](https://github.com/PurwadhikaDev/IgniteGroup_DTI_01_FinalProject/blob/main/Modeling_Auto%20Insurance_Ignite.ipynb))
        """)
        st.markdown("""
        The model is built using data from customer demographics, policy details, and historical response behavior. Important features include:
        - **Renew Offer Type**: The type of renewal offer received by the customer.
        - **Sales Channel**: The channel through which the policy was sold.
        - **Education**: The educational background of the customer.
        These features help in understanding customer behavior and preferences.
        """)

        # Convert 'Effective To Date' to datetime with flexible parsing
        data['Effective To Date'] = pd.to_datetime(data['Effective To Date'], format='mixed', dayfirst=False, errors='coerce')

        # Function to generate markdown explanation for all columns excluding certain ones
        def generate_feature_explanation(data, exclude_columns=[]):
            explanations = []

            for column in data.columns:
                if column in exclude_columns:
                    continue  # Skip columns that are in the exclude list

                if pd.api.types.is_numeric_dtype(data[column]):
                    min_value = data[column].min()
                    max_value = data[column].max()
                    explanations.append(f"- **{column}**: Numerical feature.\n    - **Range**: \n        - **Minimum**: {min_value}\n        - **Maximum**: {max_value}\n")
                elif pd.api.types.is_object_dtype(data[column]):
                    unique_values = data[column].unique()
                    explanations.append(f"- **{column}**: Categorical feature.\n    - **Possible Values**: {', '.join(map(str, unique_values))}\n")
                elif pd.api.types.is_datetime64_any_dtype(data[column]):
                    min_date = data[column].min().strftime('%m/%d/%Y')
                    max_date = data[column].max().strftime('%m/%d/%Y')
                    explanations.append(f"- **{column}**: Date feature.\n    - **Range**: {min_date} - {max_date}\n")

            return "\n".join(explanations)

        # List of columns to exclude from the explanation
        exclude_columns = ['Customer']  # You can add more columns to this list if needed

        # Use the function to generate the markdown
        feature_explanations = generate_feature_explanation(data, exclude_columns)

        st.markdown("""
        Features Input Limitation:
        """)
        st.markdown(feature_explanations)

        # Model Output
        st.subheader('Model Output')
        st.markdown("""
        The model outputs a probability score indicating the likelihood of a positive customer response to an insurance offer. This score ranges from 0 (least likely) to 1 (most likely). Based on the probability, decisions can be made regarding marketing and customer engagement strategies.
        """)

        # Model Limitations
        st.subheader('Model Limitations')
        st.markdown("""
        - **Data Imbalance**: The 'Response' variable shows a strong imbalance (85.7% 'No' vs. 14.3% 'Yes'). This imbalance might lead to biased predictions favoring the 'No' response. Techniques like oversampling, undersampling, or using class weights might be needed to address this issue.
        - **Feature Correlation**: There is a notable correlation between features like `Monthly Premium Auto` and `Total Claim Amount` (correlation of 0.632). This could lead to multicollinearity, making it challenging to interpret the impact of each feature independently and potentially biasing the model.
        - **Limited Feature Diversity**: The dataset primarily includes categorical features, with a few numerical features such as `Customer Lifetime Value`, `Income`, and `Total Claim Amount`. This imbalance may restrict the model's ability to understand complex interactions and capture nuanced customer behaviors that numerical features can reveal.
        - **Missing Temporal Trends**: The dataset has limited temporal data (e.g., `Effective To Date`) and lacks time-based features that could capture changes in customer behavior over time. This limitation can affect the model's ability to account for temporal trends, seasonality, or response to marketing campaigns.
        - **Assumption of Continuity**: The model is trained on historical data, assuming that past behavior patterns will continue in the future. Significant changes in customer behavior due to external factors (e.g., economic changes, new competitors) could reduce prediction accuracy, making the model less reliable over time.
        - **Categorical Feature Granularity**: Features like `Vehicle Class` and `State` are broad categories that may not capture important details. For instance, luxury cars and standard cars are treated similarly, potentially overlooking specific behavioral patterns. More granular features might improve the model's performance but also increase the complexity and risk of overfitting.
        """)

        # Interpretation of Results
        st.subheader('Interpretation of Results')
        st.markdown("""
        The results should be interpreted with the understanding that the model provides probabilities, not certainties. Features like 'Renew Offer Type' and 'Sales Channel' are significant indicators of customer behavior, as shown in feature importance analysis. Partial Dependence Plots provide insights into how changes in these features could impact customer response.
        """)

        # Usage Instructions
        st.subheader('Usage Instructions')
        st.markdown("""
        - **Single Prediction**: Use the form to input individual customer data and receive a prediction on their likelihood to respond positively to an insurance offer. This feature allows for personalized insights based on specific customer information.
        - **Batch Prediction**: Upload a CSV file containing multiple customer records to generate predictions for each record simultaneously. This feature is useful for analyzing and predicting customer behavior in bulk, making it easier to implement large-scale marketing and engagement strategies.
        - **Model Explanation**: Understand the inner workings of the predictive model by visualizing feature importance and partial dependence plots. This helps to see which features have the most influence on predictions and how they impact customer behavior, aiding in more informed decision-making.
        """)

        # Ethical Considerations
        st.subheader('Ethical Considerations')
        st.markdown("""
        - **Data Privacy**: All customer data used in the model is anonymized to protect individual privacy.
        - **Fairness**: The model is regularly evaluated to ensure it does not unfairly discriminate against any group of customers.
        """)

        # Future Improvements
        st.subheader('Future Improvements')
        st.markdown("""
        - **Incorporating More Data**: Expanding the dataset to include more customer interactions could improve model accuracy.
        - **Real-Time Predictions**: Implementing real-time data processing to provide up-to-date predictions.
        - **Enhanced Feature Engineering**: Exploring additional features and interactions to capture more nuances in customer behavior.
        """)

    elif st.session_state['page'] == 'Single Prediction':
        single_prediction(data)
    elif st.session_state['page'] == 'Batch Prediction':
        batch_prediction()
    elif st.session_state['page'] == 'Model Explanation':
        explain_model(treated_model, new_obs)

if __name__ == '__main__':
    main()
