# Vehicle Insurance Cross-Sell Prediction 🚗📈

This project applies **Machine Learning models** to predict the cross-sell potential of vehicle insurance to existing health insurance policyholders. Using **Decision Trees** and **Logistic Regression**, the model aims to optimize marketing strategies and improve cross-sell conversion rates.

---

## 📚 Features
- **Data Preprocessing:** Cleaning and transforming raw insurance data.
- **Model Building:** Logistic Regression and Decision Tree classification.
- **Evaluation Metrics:** ROC curves, AUC scores, Confusion Matrices, Sensitivity, Precision, and Total Error Rate.
- **Visualizations:** Exploratory Data Analysis with bar graphs and decision tree visualization.

---

## 🛠️ Tech Stack
- **Language:** Python  
- **Libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

---

## 🏗️ How to Run Locally

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/rohan9223/vehicle-insurance-cross-sell.git
   cd vehicle-insurance-cross-sell
   ```

2. **Open the Python Script:**
   Load `vehicle_insurance_prediction.py` in Visual Studio Code or your preferred IDE.

3. **Install Required Libraries:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

4. **Run the Code:** 
   Execute the script to preprocess data, build models, and evaluate performance:
   ```bash
   python vehicle_insurance_prediction.py
   ```

---

## 🎨 Results

### Logistic Regression:
- **AUC:** 0.8210
- **Precision (Interested):** 0.15
- **Recall (Interested):** 0.00
- **Total Error Rate:** 0.0921

### Decision Tree:
- **AUC:** 0.8156
- **Precision (Interested):** 0.00
- **Recall (Interested):** 0.00
- **Total Error Rate:** 0.0919

**Recommended Model:** Decision Tree due to slightly lower Total Error Rate.

---

## 📝 Observations
- Both models show high accuracy for predicting customers *not interested* in vehicle insurance but struggle with predicting *interested* customers.
- The data imbalance significantly affects model performance, indicating the need for advanced techniques like SMOTE or cost-sensitive learning.

---

## 📂 Dataset Information

- **train.csv & test.csv** files contain customer data.
- **Key Columns:**
  - `Response`: Target variable (1 = Interested, 0 = Not Interested)
  - `Gender`, `Age`, `Driving_License`, `Vehicle_Age`, `Vehicle_Damage`, `Annual_Premium`, etc.

---



---



