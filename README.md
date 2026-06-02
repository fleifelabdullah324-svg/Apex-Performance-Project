# Apex Performance Analytics: Automotive Tuning & ROI Optimization 🏎️📊

## 📌 Project Overview
The aftermarket automotive tuning industry has historically operated on trial-and-error. This project bridges the gap between mechanical engineering and data science by introducing a Business Intelligence (BI) and Machine Learning (ML) framework.

## 👥 Authors
* **Abdullah Fleifel**
* 
* **Supervised by:** [Ayman M. Mansour]

## 🗄️ Dataset
* **Size:** 25,000 independent vehicle records.
* **Features:** 94 distinct mechanical and financial columns (e.g., Target_Boost_Bar, Internals_Specs, Fuel_Type, Total_Build_Cost_USD).
* **Cost Range:** $2,541 - $50,433.
* **Preprocessing:** Mean Imputation for missing values, Label Encoding, and Min-Max Normalization using Python (Pandas).

## 🧠 Machine Learning Model
* **Algorithm:** Random Forest Classifier (Scikit-Learn).
* **Performance on 5,000 Test Samples:** * Accuracy: 99.76%
  * Precision/Recall/F1-Score: 0.99 - 1.00
  * ROC AUC Score: 0.999 (Demonstrating near-perfect classification for Critical Failure).
* **Engineering Context:** The model successfully mapped the deterministic physical bounds of internal combustion, identifying critical failure thresholds automatically based on boost, fuel, and internal components.

## 📈 Power BI Dashboard
* Features a custom dark-theme UI.
* Utilizes advanced **DAX functions** to prevent summary inflation and calculate true average build costs.
