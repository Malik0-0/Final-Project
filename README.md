<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
# Auto Insurance Classification

### ***Overview***

Welcome to the Auto Insurance Classification Project, a comprehensive repository for data exploration and machine-learning classification models using Auto Insurance data. This project uses domain knowledge for insurance analysis data to gain insights. Data science is used to build robust models to predict policyholders who have the potential to renew insurance.

### ***Business Understanding***

In today's competitive landscape, businesses must effectively attract and retain customers to succeed, a task that is increasingly reliant on data science and machine learning. These technologies enable companies to create targeted marketing campaigns, optimizing efforts and reducing costs by focusing on customers most likely to engage. However, without a dedicated machine learning team, our company faces challenges in efficiently acquiring new customers and retaining existing ones. Our current approach, which involves manually reaching out to a broad audience, is less targeted and may lead to wasted resources. By leveraging data from 2012 onward, particularly focusing on existing insurance holders and potential new customers in the Western United States, we can enhance our marketing efforts. This strategy will help us boost the number of policyholders, ultimately driving long-term business success, including strong net income, earnings per share, and return on equity [The Progressive annual report 2023](https://s202.q4cdn.com/605347829/files/doc_financials/2023/q4/interactive/static/media/Progressive-2023-AR.f93a3e76939b58794122.pdf).

### ***Problem Statement***

The company is struggling to efficiently acquire new customers and retain existing ones due to the lack of a dedicated machine learning team, leading to a broad and less targeted marketing approach. As a result, the marketing department is forced to manually contact a wide audience, which has led to inefficient use of the marketing budget and unnecessary expenses. The primary focus is on existing insurance holders, potential new customers, and those likely to renew their policies in the Western United States, using data from 2012 onwards. This lack of precision in targeting is hindering the effectiveness of marketing efforts.

### ***Goal***

To overcome its current challenges, the company plans to implement a machine learning model to predict which customers are most likely to renew their insurance policies. By focusing on high-potential customers, the company can optimize marketing efforts, reduce costs, and make campaigns more targeted and efficient. This strategy is expected to decrease the marketing budget, improve customer acquisition and retention rates, and enhance revenue from both renewals and new customers.

### ***Project Stakeholder***

Project stakeholders are those with any interest in your project's outcome. Marketing Team, untuk memberikan pola pemegang polis yang akan melakukan renewal insurance serta wawasan mengenai hal-hal yang membuat pemegang polis bertahan sebagai loyal customer. 

### ***Metric Evaluation***

We decided to use evaluation metrics such as recall, precision, and accuracy. To minimize false negatives, the main evaluation metric used is recall, and to reduce the number of false positives, the metric used is precision. These two metrics will help to evaluate the model in order to increase the efficiency of marketing offers and potential policyholders. Apart from recall and precision, we also consider accuracy to ensure the model has good performance in identifying potential policyholders who will renew or not, calculated as the proportion of correct predictions.

### ***Data details***

This CSV file is a dataset that can be used for exploratory data analysis and modeling. This dataset has various attributes of auto insurance, which is very important to understand customer habits, patterns, and trends in the insurance sector.

### ***Exploratory Data Analysis (EDA_Auto Insurance_Ignite.ipynb)***

**Objective**: This notebook discusses exploratory data analysis (EDA) of auto insurance data, aimed at finding out the distribution, correlation, and patterns. In the business field, EDA aims to find patterns of policyholders who will renew insurance and insights into things that make policyholders stay loyal customers. This is the first step in making a machine-learning model.
**Key Features**: Define business problems, skimming data, data preparation, and visualization. This process is useful for analyzing data and getting insights and business recommendations.

### ***Modeling - Benchmark Models (Modeling_Auto Insurance_Ignite.ipynb)***

**Objective**: Focusing on benchmarking various machine learning classification models is the basis for determining the most effective model. The model that has been created can predict unseen data and create real products.
**Key Features**: Benchmarking with various machine learning classification models, evaluation metrics, and confusion matrix. This is needed to find out how good the model we have created is in the context of insurance data analysis.

### *Summary*

**Summary of Revenue**

- **Without model (TP + FP): $33610**
- **With model (TP + FN) : $54084**
- **Difference:** $54084 − $33610 = **$20474**
- **Percentage Revenue:** ($20474 / $33610) × 100 = **60.91%**

Implementing the model resulted in a significant revenue increase, with total earnings rising from $33,610 (without the model) to $54,084 (with the model). This represents a difference of $20,474, translating to a 60.91% increase in revenue, highlighting the model's effectiveness in boosting financial performance.

**Business Recommendations**

1. **Targeted Strategy for Policyholder Growth:**

- **Leverage the Model for Accurate Targeting:** Implement the KNN model to accurately identify and target customers who are likely to renew their policies. By focusing on these potential policyholders, the company can significantly increase the number of renewals, leading to substantial revenue growth.
- **Maximize Policyholder Retention:** The model's ability to correctly predict renewals allows the company to enhance its retention efforts, ensuring that more customers choose to extend their policies. This increase in retained customers directly contributes to higher revenue.

2. **Revenue Enhancement:**

- **Achieve Substantial Revenue Gains:** The implementation of the model has demonstrated a significant revenue increase, with total earnings rising by 60.91%. This growth reflects the model's effectiveness in accurately identifying potential renewals and converting them into actual revenue.
- **Focus on High-Impact Opportunities:** By concentrating efforts on customers identified by the model as likely to renew, the company can maximize revenue potential and achieve a higher return on investment.

3. **Ongoing Model Optimization:**

- **Continuous Model Refinement:** Regularly monitor and refine the model to ensure its ongoing accuracy in predicting policy renewals. As the model continues to evolve with new data, it will maintain its effectiveness in driving revenue growth.
- **Adapt to Customer Behavior:** Stay responsive to changes in customer behavior by incorporating new insights into the model. This adaptability will help sustain revenue increases over time by ensuring that the company continues to target the most promising policyholders.

### ***Acknowledgment***

[Auto Insurance dataset](https://www.kaggle.com/datasets/ranja7/vehicle-insurance-customer-data/data)

### Built With

This project was completely built using the tools:

* Python
* Datetime Library
* Time Library

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/Malik0-0/Travel-Journal-CRUD-.git
   ```

<!-- USAGE EXAMPLES -->

## Usage

The flow of using this application is very simple. When run, the program will display various main menus that can be selected in the application and to access additional features the user can select the fifth option, namely "Additional Features".

For an explanation of the use of each feature, it has been explained in "About the Project" and to see how the application workflow works, users can open the "Flowchart of Capstone Project.pdf" file.

Main Menu:

* Option 1:
  Users can select option 1 on the main menu if they want to add a new journal then input the title, content, date and location.
* Option 2:
  Users can select option 2 on the main menu if they want to display various journals that already exist in the database.
* Option 3:
  Users can select option 3 on the main menu if they want to edit one of the journals in the database.
* Option 4:
  Users can select option 4 on the main menu if they want to delete one of the journals in the database.
* Option 5:
  Users can select option 5 on the main menu if they want to access various additional features in the application.
* Option 6:
  Users can select option 6 on the main menu if they want to exit the application.

Additional Menu:

* Option 1:
  Users can select option 1 on the Additional menu if they want to search for journals based on keywords.
* Option 2:
  Users can select option 2 on the Additional menu if they want to sort journals based on date.
* Option 3:
  Users can select option 3 on the Additional menu if they want to filter journals based on a specific year.
* Option 4:
  Users can select option 4 on the Additional menu if they want to filter journals based on a specific year range.
* Option 5:
  Users can select option 5 on the Additional menu if they want to see a summary of their journey so far based on data from all journals in the database.
* Option 6:
  Users can select option 6 on the Additional menu if they want to Return to the Main Menu.

<!-- CONTACT -->

## Contact

Malik Alrasyid Basori - malikalrasyidbasori.1@gmail.com

Project Link: https://github.com/Malik0-0
