# Diabetes-Risk-Analysis

This is a Streamlit web application designed for exploratory data analysis (EDA) and risk prediction over a synthetic dataset of 15,000 patients. It explores patient attributes and trains a machine learning model to predict diabetes risk levels (Low, Moderate, or High).

## Live link 
[Diabetes Risk](https://baguirr1--diabetes-risk-dashboard-run.modal.run/)

## App Features
The dashboard is separated into three intuitive tabs:
* **📋 Data Overview**: Displays high-level dataset metrics (row and column counts), highlights missing values, and provides a numeric summary and sample rows for quick inspection.
* **📊 Exploratory Analysis**: Uses interactive Plotly visualizations to show risk distribution, age and BMI histograms, a feature correlation heatmap, and risk breakdowns by physical activity and family history.
* **🩺 Predict Risk**: Showcases a Random Forest classifier trained on an 80/20 split of the dataset. It displays the model's test accuracy, classification report, confusion matrix, and top feature importances, while allowing users to input custom patient attributes to generate real-time risk predictions.
<img width="1836" height="992" alt="image" src="https://github.com/user-attachments/assets/317ffb04-80f1-4996-97a3-4db95636a3aa" />
<img width="1844" height="1016" alt="image" src="https://github.com/user-attachments/assets/4934af24-433e-4417-934e-c895fe0437d5" />
<img width="1855" height="952" alt="image" src="https://github.com/user-attachments/assets/340edb06-a823-4a26-a6b5-194651917b1e" />




## Dependencies
To run this application, you must have Python installed along with the following libraries:
* `pandas`
* `plotly`
* `streamlit`
* `scikit-learn`

## Getting Started
1. Ensure the dataset file, exactly named `diabetes_risk.csv`, is located in the same directory as your Python script.
2. Install the necessary Python packages using pip:
   ```bash

   pip install pandas plotly streamlit scikit-learn

## Limitations

* **Not a Clinical Diagnostic Tool:** This application is strictly for educational, exploratory, and research purposes. It is a predictive model based on historical data patterns and should never replace professional medical diagnosis, advice, or clinical judgment.
* **Scope of Features:** While the dataset includes crucial clinical indicators (such as HbA1c and blood glucose levels) alongside demographic data, real-world diagnosis often relies on broader medical histories, lifestyle nuances, and longitudinal tracking not fully captured in a static dataset.
* **Potential Data Biases:** Predictive performance is inherently tied to the demographic and regional distribution of the patient population from which this data was collected. The model may not generalize perfectly to demographic groups underrepresented in the training set.
* **Static Model Training:** The Random Forest classifier operates on a fixed 80/20 train-test split of the existing data. It does not dynamically learn or retrain from new inputs submitted through the Streamlit user interface.
