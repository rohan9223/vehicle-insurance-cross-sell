
# Vehicle Insurance Cross-Sell Prediction 🚗📈

This project applies **Machine Learning models** to predict the cross-sell potential of vehicle insurance to existing health insurance policyholders. Using **Decision Trees** and **Logistic Regression**, the model aims to optimize marketing strategies and improve cross-sell conversion rates.

---

## 📚 Features
- **Data Preprocessing:** Cleaning and transforming raw insurance data.
- **Model Building:** Logistic Regression and Decision Tree classification.
- **Evaluation Metrics:** ROC curves, AUC scores, and Confusion Matrices.
- **Imbalanced Data Handling:** Down-sampling to balance class distribution.

---

## 🛠️ Tech Stack
- **Language:** R  
- **Libraries:** `dplyr`, `caret`, `rpart`, `ROCR`, `partykit`, `corrplot`, `pROC`, `rattle`  

---

## 🏗️ How to Run Locally

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/rohan9223/vehicle-insurance-cross-sell.git
   cd vehicle-insurance-cross-sell
2. **Open the R Script:**
    Load vehicle_insurance_prediction.R in RStudio.

3. **Install Required Libraries:**

install.packages(c("dplyr", "caret", "rpart", "ROCR", "partykit", "corrplot", "pROC", "rattle"))

4. **Run the Code:** 
Execute the script to preprocess data, build models, and evaluate performance.


## 🎨 Results
Logistic Regression AUC: 83.67%
Decision Tree AUC: 83.44%
Recommended Model: Decision Tree due to lower error rate
