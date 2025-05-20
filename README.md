# Telecom-Churn-Dataset

This project aims to predict customer churn in a telecom company using various machine learning models. It follows an **object-oriented programming (OOP)** approach for modularity and scalability.

---

## 📁 Project Structure
├── data_loader.py # Handles dataset loading from URLs
├── preprocessing.py # Handles data cleaning, encoding, scaling, and SMOTE
├── model.py # Model wrapper class for training and prediction
├── evaluate.py # Accuracy and confusion matrix evaluation
├── requirements.txt # Required Python libraries
├── main.py # Pipeline driver script
├── Model.ipynb # Jupyter notebook version of the project


---

## 📊 Dataset

The dataset is taken from:
- [churn-bigml-80.csv](https://raw.githubusercontent.com/mukeshmagar543/Telecom-Churn-Dataset/refs/heads/main/churn-bigml-80.csv)
- [churn-bigml-20.csv](https://raw.githubusercontent.com/mukeshmagar543/Telecom-Churn-Dataset/refs/heads/main/churn-bigml-20.csv)

It contains information such as:
- State, account length, international plan, voice mail plan
- Usage metrics: minutes, calls, charges
- Target variable: **Churn**

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/telecom-churn-oop.git
cd telecom-churn-oop

2. Install Dependencies
pip install -r requirements.txt

3. Run the Project
python main.py


🧠 ML Models Used
Logistic Regression

Decision Tree

Random Forest

K-Nearest Neighbors (KNN)

Support Vector Machine (SVM)

AdaBoost

Gradient Boosting

XGBoost


📈 Evaluation Metrics
Accuracy Score

Cross-Validation Score

Confusion Matrix


📉 Feature Engineering
Label Encoding for categorical features

MinMax Scaling for numerical features

SMOTE for handling class imbalance

VIF for multicollinearity check (optional)

Train-test split with reproducibility


✅ Output
Model accuracy and cross-validation comparison plot

Export trained model as .pkl using joblib

📬 Contact
Maintained by: Mukesh Magar
Email: mukeshmagar543@gmail.com
GitHub: mukeshmagar543