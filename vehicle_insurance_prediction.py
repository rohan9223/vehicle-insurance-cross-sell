import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset from 'data.csv'
data = pd.read_csv('data.csv')

# 1. Check for missing values
print("Missing values in each column:")
print(data.isnull().sum())

# Handle missing values (if any):
# For simplicity, here we will drop rows with missing values, but you can impute if needed.
# Uncomment the line below if you wish to drop rows with missing values:
# data.dropna(inplace=True)

# 2. Creation of factor variables
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
data['Vehicle_Age'] = data['Vehicle_Age'].map({'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2})
data['Vehicle_Damage'] = data['Vehicle_Damage'].map({'No': 0, 'Yes': 1})

# 3. Normalize the 'Annual_Premium' feature using Min-Max Scaling
scaler = StandardScaler()
data['Annual_Premium'] = scaler.fit_transform(data[['Annual_Premium']])

# 4. Split the dataset into features (X) and target (y)
X = data.drop(columns=['id', 'Response'])
y = data['Response']

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verify preprocessing steps
print("Sample of preprocessed data:")
print(data.head())

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize Logistic Regression model
logreg = LogisticRegression()

# Fit the model
logreg.fit(X_train, y_train)

# Predictions
y_pred_logreg = logreg.predict(X_test)

# Confusion Matrix
cm_logreg = confusion_matrix(y_test, y_pred_logreg)
sns.heatmap(cm_logreg, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

# ROC Curve for Logistic Regression
fpr_logreg, tpr_logreg, _ = roc_curve(y_test, logreg.predict_proba(X_test)[:, 1])
roc_auc_logreg = auc(fpr_logreg, tpr_logreg)
plt.plot(fpr_logreg, tpr_logreg, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_logreg:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Logistic Regression')
plt.legend(loc='lower right')
plt.show()

# Sensitivity, Precision, Total Error Rate, and ROC
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_logreg))


from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

# Initialize Decision Tree model
dtree = DecisionTreeClassifier(max_depth=3,random_state=42)

# Fit the model
dtree.fit(X_train, y_train)

# Predictions
y_pred_dtree = dtree.predict(X_test)

# Confusion Matrix
cm_dtree = confusion_matrix(y_test, y_pred_dtree)
sns.heatmap(cm_dtree, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Decision Tree')
plt.show()

# ROC Curve for Decision Tree
fpr_dtree, tpr_dtree, _ = roc_curve(y_test, dtree.predict_proba(X_test)[:, 1])
roc_auc_dtree = auc(fpr_dtree, tpr_dtree)
plt.plot(fpr_dtree, tpr_dtree, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_dtree:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Decision Tree')
plt.legend(loc='lower right')
plt.show()

# Sensitivity, Precision, Total Error Rate, and ROC
print("Decision Tree Classification Report:")
print(classification_report(y_test, y_pred_dtree))

# Visualize Decision Tree
plt.figure(figsize=(12,8))
plot_tree(dtree, filled=True, feature_names=X.columns, class_names=['Not Interested', 'Interested'], rounded=True)
plt.title('Decision Tree Visualization')
plt.show()

# Total Error Rate Calculation
accuracy_logreg = logreg.score(X_test, y_test)
error_rate_logreg = 1 - accuracy_logreg

accuracy_dtree = dtree.score(X_test, y_test)
error_rate_dtree = 1 - accuracy_dtree

print(f"Logistic Regression Total Error Rate: {error_rate_logreg:.4f}")
print(f"Decision Tree Total Error Rate: {error_rate_dtree:.4f}")

if error_rate_logreg < error_rate_dtree:
    print("Logistic Regression is the better model based on Total Error Rate.")
else:
    print("Decision Tree is the better model based on Total Error Rate.")

