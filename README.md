# Diabetes Risk Classification (KNN + Streamlit)

An interactive machine learning web application built with **Streamlit** and **scikit-learn** using **K-Nearest Neighbors (KNN)** to classify diabetes risk and display class probabilities.

## Dataset & Schema
The app uses a 500-row synthetic patient dataset with the following schema:
- `PatientID`: Patient identifier (dropped before model training)
- `Glucose`: Plasma glucose concentration (`mg/dL`)
- `BloodPressure`: Diastolic blood pressure (`mm Hg`)
- `BMI`: Body mass index (`kg/m²`)
- `Insulin`: Serum insulin (`μU/mL`)
- `Age`: Age in years
- `DiabetesStatus`: Target label (`0` = Non-Diabetic, `1` = Diabetic)

## ML Pipeline
1. **Data Cleaning & Imputation:** Biologically invalid `0` values in clinical features are replaced with `NaN` and imputed using column medians.
2. **Feature Selection & Scaling:** `PatientID` is excluded from features, and remaining numerical features are standardized using `StandardScaler`.
3. **Train/Test Split:** Stratified `80/20` split to preserve class balance.
4. **Model Tuning:** Evaluates `KNeighborsClassifier` across `k = [3, 5, 7, 9]` (selecting `k = 9`) and reports model performance metrics and confusion matrix.
5. **Interactive Inference:** Accepts patient parameters in the UI, scales inputs, and returns risk classification along with probability scores.

## Tech Stack
- Python
- Streamlit
- Pandas & NumPy
- Scikit-learn

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
