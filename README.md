# CSC 4110 Machine Learning Group Project - Group G7

## 📌 Project Overview
This repository contains the complete machine learning pipeline and web application for **Group G7**. The project uses the "Rain in Australia" dataset (approx. 145,000 daily weather records from 49 stations) to solve two real-world problems for **Mvua Alert**, a fictional weather-alert start-up in Kisumu:
1. **Regression:** Estimate the maximum temperature (`MaxTemp`) using morning (9 a.m.) readings.
2. **Classification:** Predict whether it will rain tomorrow (`RainTomorrow`)

## How to Run the Notebooks
1. Open the `notebooks` folder.
2. Upload the desired notebook to Google Colab.
3. Run the first cell to mount your Google Drive.
4. Ensure the dataset `weatherAUS.csv` is in `CSC4110_G07/data/`.
5. Click `Runtime` -> `Run all`.

## How to Run the App
1. Navigate to the `app` folder.
2. Run the app locally using: `streamlit run app.py`

## Task Ownership
| Member | Task 1 | Task 2 |
| :--- | :--- | :--- |
| Lawrence Simiyu | Task 2.1 (Data Cleaning) | Task 7 (Unsupervised & RL) |
| Felix Aloo | Task 2.2 (EDA & Split) | Task 8 (Neural Networks) |
| Martin Rotich | Task 3.1 (Linear Regression) | Task 5 (Evaluation) |
| Winfred Mutuku | Task 3.2 (Regularisation) | Task 9.1 (Streamlit App) |
| Swynneystone Ongoto | Task 4 (Classification) | Task 6 (Trees & Ensembles) |

## For the Fisher (Plain Words)
When you receive an SMS from Mvua Alert, it uses today's weather to guess if it will rain tomorrow. If the message says rain is coming, it is best to stay close to the shore and secure your nets. The app is not a replacement for official warnings, but it helps you plan your day safely.

##🛠️ Technologies Used
1. **Python** (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
2. **Google Colab** (Model training and evaluation)
3. **Streamlit** (Web application deployment)
4. **Joblib** (Model packaging and serialization)
