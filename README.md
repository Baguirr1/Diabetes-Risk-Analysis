# Diabetes-Risk-Analysis

This is a Streamlit web application designed for exploratory data analysis (EDA) and risk prediction over a synthetic dataset of 15,000 patients. It explores patient attributes and trains a machine learning model to predict diabetes risk levels (Low, Moderate, or High).

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
