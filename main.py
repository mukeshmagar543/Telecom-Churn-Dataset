from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.model import ModelBuilder
from src.evaluate import Evaluator
import xgboost
from xgboost import XGBClassifier

# Load and preprocess data
url1 = 'https://raw.githubusercontent.com/mukeshmagar543/Telecom-Churn-Dataset/refs/heads/main/churn-bigml-80.csv'
url2 = 'https://raw.githubusercontent.com/mukeshmagar543/Telecom-Churn-Dataset/refs/heads/main/churn-bigml-20.csv'

data_loader = DataLoader(url1, url2)
df = data_loader.load_data()

preprocessor = Preprocessor(df)
df_clean = preprocessor.transform()
X_train, X_test, y_train, y_test = preprocessor.split_and_scale()

# Train and predict
model = ModelBuilder(XGBClassifier())
trained_model = model.train(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
accuracy, cm = Evaluator.evaluate(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:\n", cm)
