import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import fbeta_score
from sklearn.metrics import make_scorer
from streamlit_option_menu import option_menu
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import permutation_importance

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

# Visualize data
def visualize_data(data):
    # Embed Tableau Dashboard
    st.subheader('Interactive Tableau Dashboard')
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

    st.subheader('Dataset Overview')
    st.dataframe(data.head())
    
    st.subheader('Data Description')
    st.write(data.describe())
    
    # 1. Distribution of Numerical Columns
    st.subheader('Distribution of Numerical Features')
    numerical_features = ['Customer Lifetime Value', 'Income', 'Total Claim Amount', 'Monthly Premium Auto']
    for feature in numerical_features:
        st.write(f'Distribution of {feature}')
        with st.markdown('<div class="plot-container">', unsafe_allow_html=True):
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            sns.histplot(data[feature], kde=True, ax=ax[0])
            ax[0].set_title(f'Histogram of {feature}')
            sns.boxplot(x=data[feature], ax=ax[1])
            ax[1].set_title(f'Boxplot of {feature}')
            st.pyplot(fig)

    # 2. Correlation Analysis with Response
    st.subheader('Correlation with Response')
    for feature in numerical_features:
        st.write(f'Scatter plot between {feature} and Response')
        with st.markdown('<div class="plot-container">', unsafe_allow_html=True):
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.scatterplot(x=data[feature], y=data['Response'], ax=ax)
            ax.set_title(f'Relationship between {feature} and Response')
            st.pyplot(fig)

    # Correlation Heatmap
    st.subheader('Correlation Heatmap')
    numerical_data = data.select_dtypes(include=[np.number])
    corr = numerical_data.corr()
    with st.markdown('<div class="plot-container">', unsafe_allow_html=True):
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    
    # 3. Distribution of Policyholders by State
    st.subheader('Distribution of Policyholders by State')
    with st.markdown('<div class="plot-container">', unsafe_allow_html=True):
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(y='State', data=data, ax=ax, order=data['State'].value_counts().index)
        ax.set_title('Number of Policyholders by State')
        st.pyplot(fig)

    # 4. Education Level Segmentation
    st.subheader('Education Level Segmentation')
    with st.markdown('<div class="plot-container">', unsafe_allow_html=True):
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(x='Education', data=data, order=data['Education'].value_counts().index, ax=ax)
        ax.set_title('Distribution of Education Levels')
        st.pyplot(fig)

    # 5. Purchase Timing (using Effective To Date if available)
    st.subheader('Timing of Insurance Purchases')
    if 'Effective To Date' in data.columns:
        data['Effective To Date'] = pd.to_datetime(data['Effective To Date'])
        data['Purchase Month'] = data['Effective To Date'].dt.month
        with st.markdown('<div class="plot-container">', unsafe_allow_html=True):
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.countplot(x='Purchase Month', data=data, ax=ax)
            ax.set_title('Number of Purchases by Month')
            st.pyplot(fig)

    # 6. Gender Proportion
    st.subheader('Gender Proportion')
    with st.markdown('<div class="plot-container">', unsafe_allow_html=True):
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(x='Gender', data=data, ax=ax)
        ax.set_title('Gender Distribution')
        st.pyplot(fig)

    # 7. Distribution by Police District
    st.subheader('Distribution by Police District')
    if 'Police District' in data.columns:
        with st.markdown('<div class="plot-container">', unsafe_allow_html=True):
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.countplot(y='Police District', data=data, ax=ax, order=data['Police District'].value_counts().index)
            ax.set_title('Number of Policyholders by Police District')
            st.pyplot(fig)

    # 8. Marital Status Distribution
    st.subheader('Marital Status Distribution')
    with st.markdown('<div class="plot-container">', unsafe_allow_html=True):
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(x='Marital Status', data=data, ax=ax)
        ax.set_title('Marital Status of Policyholders')
        st.pyplot(fig)

    # 9. Policy Types Distribution
    st.subheader('Policy Types Distribution')
    with st.markdown('<div class="plot-container">', unsafe_allow_html=True):
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(x='Policy Type', data=data, ax=ax)
        ax.set_title('Distribution of Policy Types')
        st.pyplot(fig)

    # 10. Renewal Offers
    st.subheader('Renewal Offers Distribution')
    with st.markdown('<div class="plot-container">', unsafe_allow_html=True):
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(x='Renew Offer Type', data=data, ax=ax)
        ax.set_title('Renew Offer Types Frequency')
        st.pyplot(fig)

    # 11. Sales Channels
    st.subheader('Sales Channels Distribution')
    with st.markdown('<div class="plot-container">', unsafe_allow_html=True):
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(x='Sales Channel', data=data, ax=ax)
        ax.set_title('Sales Channel Usage')
        st.pyplot(fig)

    # 12. Vehicle Types
    st.subheader('Vehicle Types Distribution')
    with st.markdown('<div class="plot-container">', unsafe_allow_html=True):
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(x='Vehicle Class', data=data, ax=ax)
        ax.set_title('Distribution of Vehicle Types')
        st.pyplot(fig)

    # 13. Vehicle Sizes
    st.subheader('Vehicle Sizes Distribution')
    with st.markdown('<div class="plot-container">', unsafe_allow_html=True):
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(x='Vehicle Size', data=data, ax=ax)
        ax.set_title('Distribution of Vehicle Sizes')
        st.pyplot(fig)

    # 14. Coverage Types
    st.subheader('Coverage Types Distribution')
    with st.markdown('<div class="plot-container">', unsafe_allow_html=True):
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(x='Coverage', data=data, ax=ax)
        ax.set_title('Distribution of Coverage Types')
        st.pyplot(fig)

    # 15. Relationship Between Categorical Features and Response
    st.subheader('Relationship Between Categorical Features and Response')
    categorical_features = ['State', 'Education', 'EmploymentStatus', 'Policy Type', 'Renew Offer Type', 'Sales Channel', 'Vehicle Class', 'Vehicle Size', 'Coverage', 'Gender', 'Marital Status']
    for feature in categorical_features:
        st.write(f'Impact of {feature} on Response')
        with st.markdown('<div class="plot-container">', unsafe_allow_html=True):
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.countplot(x=feature, hue='Response', data=data, ax=ax)
            ax.set_title(f'Relationship between {feature} and Response')
            st.pyplot(fig)

    # 16. Ratio of Each Feature to the Response and Segmentation Analysis
    st.subheader('Ratio and Segmentation Analysis')
    for feature in categorical_features:
        st.write(f'Ratio of {feature} Categories to Response')
        feature_response_ratio = data.groupby(feature)['Response'].value_counts(normalize=True).unstack().fillna(0)
        with st.markdown('<div class="plot-container">', unsafe_allow_html=True):
            fig, ax = plt.subplots(figsize=(10, 5))
            feature_response_ratio.plot(kind='bar', stacked=True, ax=ax)
            ax.set_title(f'Segmentation Analysis: {feature}')
            ax.set_ylabel('Proportion of Response')
            st.pyplot(fig)


# Preprocess data to match training
def preprocess_data(df):
    df = df.drop(columns=['Customer', 'Effective To Date', 'unnamed: 0'], errors='ignore')
    df['CLV_log'] = np.log1p(df['Customer Lifetime Value'])
    df['Income_Log'] = np.log1p(df['Income'])
    df['TCA_Log'] = np.log1p(df['Total Claim Amount'])
    df = df.drop(columns=['Customer Lifetime Value', 'Income', 'Total Claim Amount'], errors='ignore')
    return df

def f2_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=2)

f2_scorer = make_scorer(f2_score)

# Model Explanation with Permutation Importance
def explain_model(model, X_train):
    st.subheader('Model Explanation: Permutation Importance')
    
    # Preprocess data
    X_train_processed = preprocess_data(X_train)
    
    # Ensure consistent label types
    y_train = X_train_processed['Response'].map({'Yes': 1, 'No': 0})
    
    # Prepare feature data
    X_train_features = X_train_processed.drop(columns=['Response'])

    # Compute permutation importance
    results = permutation_importance(model, X_train_features, y_train, scoring=f2_scorer, random_state=42)
    
    # Get the importance values and standard deviations
    importance = results.importances_mean
    std = results.importances_std
    feature_names = X_train_features.columns
    
    # Sort features by importance
    sorted_idx = np.argsort(importance)[::-1]
    
    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feature_names[sorted_idx], importance[sorted_idx], capsize=5)
    ax.set_xlabel('Permutation Importance')
    ax.set_title('Feature Importance using Permutation Importance')
    ax.invert_yaxis()  # Most important features at the top
    
    # Display plot in Streamlit
    st.pyplot(fig)

    # Identify numeric and categorical features
    numeric_features = X_train_features.select_dtypes(include=[np.number]).columns
    categorical_features = X_train_features.select_dtypes(include=['object', 'category']).columns

    # Select top 3 numeric features for PDP
    top_numeric_features = [feature for feature in feature_names[sorted_idx] if feature in numeric_features][:3]

    st.subheader('Partial Dependence Plots for Top Numeric Features')
    if top_numeric_features:
        st.write('Visualizing the impact of key numeric features on model predictions.')
        # Create PDP plots for selected top numeric features
        fig, ax = plt.subplots(figsize=(12, 8))
        display = PartialDependenceDisplay.from_estimator(model, X_train_features, top_numeric_features, ax=ax)
        plt.suptitle('Partial Dependence Plots')
        plt.subplots_adjust(top=0.9)
        plt.show()
        
        # Display PDP plot in Streamlit
        st.pyplot(fig)
    else:
        st.write('No numeric features found for Partial Dependence Plots.')

    # Select top 3 categorical features
    top_categorical_features = [feature for feature in feature_names[sorted_idx] if feature in categorical_features][:3]

    st.subheader('Insights for Top Categorical Features')
    if top_categorical_features:
        st.write('Visualizing the impact of key categorical features on model predictions.')
        
        # Create subplots for top categorical features
        num_cats = len(top_categorical_features)
        fig, axes = plt.subplots(1, num_cats, figsize=(5 * num_cats, 6), sharey=True)
        
        for i, feature in enumerate(top_categorical_features):
            ax = axes[i] if num_cats > 1 else axes
            X_train_features[feature].value_counts().plot(kind='bar', ax=ax)
            ax.set_title(f'Distribution of {feature}')
            ax.set_xlabel('Category')
            ax.set_ylabel('Frequency')
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.write('No categorical features found for analysis.')

    # Interpretation
    st.subheader('Interpretation')
    st.markdown("""
    **Permutation Importance**
    **Key Feature Insights**:

    - **Renew Offer Type**: The most important feature, indicating that customers are highly sensitive to the type of renewal offers they receive. Offering attractive renewal options can greatly influence customer retention. The high frequency of "Offer1" suggests it is the most preferred, so optimizing and testing different renewal strategies similar to "Offer1" could further enhance retention.

    - **Sales Channel**: The second most influential feature, showing that the method of selling insurance has a significant impact on customer response. The predominance of agent-led sales suggests customers value personalized service. Investing in training for sales agents and optimizing branch operations could boost engagement and conversion rates.

    - **Education**: Indicates that educational background affects how customers respond to insurance offerings. Higher education levels may correlate with a better understanding of insurance products, suggesting a need for tailored communication strategies. Aligning messages to resonate with customers' education levels could improve engagement and conversion.

    - **Employment Status**: Affects customer behavior, with different employment statuses influencing response rates. Personalized marketing strategies based on employment status (e.g., employed, retired) could increase engagement and relevance of offers.

    - **Coverage**: The type of coverage selected is critical, indicating customers' varying needs and risk perceptions. Providing clear, detailed information on the benefits of different coverage options can help customers make informed choices, leading to higher conversion rates.

    - **Gender**: Gender differences in response indicate opportunities for gender-specific marketing strategies, catering to the unique needs and preferences of each demographic.

    - **Number of Policies**: The partial dependence plot shows that customers with more policies may respond less favorably, possibly due to increased complexity or cost. Simplifying offerings or providing bundled policy options could retain these customers.

    - **Vehicle Size**: The impact of vehicle size on customer response points to differing needs based on vehicle type. Tailoring insurance products to cater to specific vehicle types can enhance relevance and attractiveness.

    **Partial Dependence Plots**
    
    <br>**Insights for Top Categorical Features**:

    - **Renew Offer Type**: The distribution shows a preference for "Offer1" and "Offer2", highlighting the importance of these offers in retaining customers. Continuously innovating and optimizing these offers can help maintain customer loyalty.

    - **Sales Channel**: With agents being the most popular channel, focusing on agent training and performance can significantly impact customer acquisition and satisfaction. Exploring opportunities to enhance other channels like online and call centers could diversify and strengthen sales strategies.

    - **Education**: The distribution shows a concentration in Bachelor's and College-level education, indicating that these customers may value detailed, clear information. Crafting educational content and offers that resonate with these segments can improve engagement.
    
    **Insights for Top Numerical Features**:

    - **Number of Policies**: Customers with a higher number of policies show a decreasing response, suggesting that managing multiple policies could be a barrier. Streamlining policy management or offering discounts for multi-policy holders might improve satisfaction.

    - **Months Since Policy Inception**: Customers' responses vary with the duration of their policy, indicating key engagement moments. Utilizing these insights to schedule proactive communication can enhance customer retention.

    - **CLV_log (Customer Lifetime Value log)**: Variations in customer lifetime value impact response, suggesting that high-value customers may require different engagement strategies. Tailoring offers to match lifetime value can maximize customer loyalty and profitability.

    **Model Impact**:

    - **Predictive Accuracy**: The strong performance of features like Renew Offer Type and Sales Channel indicates that the model accurately predicts customer behavior based on these factors. This builds confidence in using the model for decision-making.

    - **Generalizability**: The insights into both numeric and categorical features provide a well-rounded understanding, helping the model adapt to different customer segments and data variations, enhancing its robustness.

    **Business Impact**:

    - **Customer Retention**: By focusing on top features like renew offer types and optimizing sales channels, businesses can improve customer retention and satisfaction. Implementing targeted renewal offers and analyzing channel performance can lead to strategic adjustments.

    - **Sales Optimization**: Insights from feature importance can guide strategic investments in high-performing sales channels and coverage types, improving marketing and sales efficiency.

    - **Targeted Marketing**: Understanding customer education, employment status, and gender allows for more targeted marketing efforts, increasing engagement and conversion rates.

    - **Product Development**: Insights into features like vehicle size and the number of policies can drive innovation in product offerings, catering to specific customer segments and enhancing market positioning.

    - **Managing Uncertainty**: The error bars in the permutation importance plot indicate some uncertainty. This variability should be considered, especially for features with larger error bars, and further analysis might be needed to refine these insights.

    - **Interaction Effects**: While these plots provide a global view, potential interactions between features may not be fully captured. Further analysis using advanced techniques like SHAP or 2D PDPs can explore these interactions.

    **Final Note**:
    - **Iterative Analysis**: This interpretation should be part of an iterative process, continuously refining insights based on new data and evolving business needs. Regularly updating the model and analysis will help maintain alignment with business goals and market dynamics.
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

    st.subheader('Make a Prediction')

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
        st.subheader('Prediction Results and Analysis')
        
        if st.session_state['input_df'] is not None and st.session_state['prediction_result'] is not None:
            prediction = st.session_state['prediction_result']
            st.write(f'The model predicts: {"Yes" if prediction == 1 else "No" if prediction == 0 else "Error"}')

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
        batch_data = batch_data.drop(columns=['Customer', 'Effective To Date'])
        
        # Mapping 'Yes'/'No' to 1/0 in the Response column (if applicable)
        if 'Response' in batch_data.columns:
            batch_data['Response'] = batch_data['Response'].map({'Yes': 1, 'No': 0}).astype(float)
        
        # Applying logarithmic transformation
        if 'Customer Lifetime Value' in batch_data.columns:
            batch_data['CLV_log'] = np.log1p(batch_data['Customer Lifetime Value'])
            batch_data = batch_data.drop(columns=['Customer Lifetime Value'])
        if 'Income' in batch_data.columns:
            batch_data['Income_Log'] = np.log1p(batch_data['Income'])
            batch_data = batch_data.drop(columns=['Income'])
        if 'Total Claim Amount' in batch_data.columns:
            batch_data['TCA_Log'] = np.log1p(batch_data['Total Claim Amount'])
            batch_data = batch_data.drop(columns=['Total Claim Amount'])
        
        # Ensure the columns are in the correct order by matching training data
        # Adjust based on actual model features used
        model_features = [col for col in batch_data.columns if col != 'Response']
        batch_data = batch_data[model_features]

        if st.button('Predict'):
            batch_predictions = treated_model.predict(batch_data)
            st.write("Batch Predictions:")
            st.write(batch_predictions)
    else:
        st.write("Please upload a CSV file for batch prediction.")

# Main Streamlit app
def main():
    # Navigation
    selected_page = option_menu(
        menu_title="Auto Insurance Response Prediction", 
        options=["Home", "Data Visualization", "Single Prediction", "Batch Prediction", "Model Explanation"], 
        icons=["house", "bar-chart", "robot", "archive", "info-circle"], 
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

        # Purpose of the Model
        st.subheader('Purpose of the Model')
        st.markdown("""
        The primary goal of this model is to predict customer responses to auto insurance offers. By understanding which factors influence customer decisions, the business can tailor its offerings to improve conversion rates and customer satisfaction.
        """)

        # Motivation and Benefits
        st.subheader('Motivation and Benefits')
        st.markdown("""
        - **Improved Customer Retention**: By identifying key factors that drive customer responses, we can design targeted strategies to retain more customers.
        - **Sales Optimization**: Understanding the most effective sales channels and offers enables the business to allocate resources more efficiently.
        - **Enhanced Customer Satisfaction**: Tailoring insurance products to match customer needs and preferences improves overall satisfaction and loyalty.
        """)

        # How the Model Works
        st.subheader('How the Model Works')
        st.markdown("""
        Our model uses machine learning techniques, specifically a pipeline model including feature transformations and classifiers. Key features such as 'Renew Offer Type', 'Sales Channel', and 'Education' are analyzed to predict customer responses. The model is trained on historical data and uses feature importance and partial dependence plots for interpretation.
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

        # Display the markdown in Streamlit
        st.subheader('Data and Features')
        st.markdown("""
        The model is built using data from customer demographics, policy details, and historical response behavior. Important features include:
        - **Renew Offer Type**: The type of renewal offer received by the customer.
        - **Sales Channel**: The channel through which the policy was sold.
        - **Education**: The educational background of the customer.
        These features help in understanding customer behavior and preferences.
        """)
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
        - **Data Visualization**: Explore various charts and graphs that provide insights into the dataset. This includes visualizations of key metrics, feature distributions, correlations between variables, and class imbalances. Use these tools to better understand the underlying patterns and trends in the data.
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
    elif st.session_state['page'] == 'Data Visualization':
        visualize_data(data)
    elif st.session_state['page'] == 'Single Prediction':
        single_prediction(data)
    elif st.session_state['page'] == 'Batch Prediction':
        batch_prediction()
    elif st.session_state['page'] == 'Model Explanation':
        explain_model(treated_model, new_obs)

if __name__ == '__main__':
    main()
