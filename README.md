# aai_540
AAI_540 Final Project Repo

Contributors: Ryan Laxamana, Dina Shalaby, Lishi Wang

Project Background: Optimizing Patient Care with Advanced Predictive Modeling
In this project, the primary objective is to optimize hospital operations by accurately predicting the length of patient stays. This prediction is essential for improving resource allocation, including staff scheduling, bed management, and overall hospital capacity. Accurately forecasting patient stays will enhance the efficiency of care delivery, minimize wait times, and potentially reduce costs. By leveraging advanced machine learning techniques, the goal is to create a robust model that can predict patient stays based on various patient and hospital-related features.

The problem being solved is a supervised machine learning problem, specifically a classification task. The "Stay" column in the dataset has been label-encoded, meaning the goal is to classify patients into different stay duration categories. The machine learning model will use historical data to learn patterns and predict outcomes based on input features, helping healthcare administrators better manage hospital operations.

Technical Background:
To evaluate the model, a combination of accuracy and performance metrics like precision, recall, and F1-score will be used. These metrics will help determine how well the model can predict patient stays and guide improvements in hospital resource allocation. The business use case is to help hospitals streamline operations, reduce bottlenecks, and improve the overall patient care experience.

The data for this project comes from a CSV file containing relevant patient and hospital stay information. It includes both categorical and numerical features. Key steps for data preparation include handling missing values in columns like "Bed Grade" and "City_Code_Patient" using mode imputation. Additionally, categorical columns such as "Stay" have been encoded using Label Encoding to make them suitable for machine learning models. Data exploration will involve checking for outliers, distribution of features, and relationships between variables.

Main features likely to impact the prediction include patient demographics, hospital infrastructure (e.g., bed grade), and the patient's prior medical history. The model selected is LightGBM, a gradient boosting framework that offers fast training and strong performance for classification tasks, especially on structured datasets.

Goals vs Non-Goals:
Goals:

Develop a machine learning model that can accurately predict patient length of stay based on historical data.
Improve hospital resource management by providing actionable insights from the model's predictions.
Ensure the model can be refined for potential real-time deployment in healthcare systems.
Analyze and present model performance using key evaluation metrics (accuracy, F1 score, etc.).
Explore the impact of different features on the model's predictions.
Non-Goals:

The model will not focus on predicting specific health outcomes for patients (e.g., recovery or complications).
The project does not aim to develop a system for clinical decision-making regarding patient treatment.
Real-time deployment of the model is not the current focus, though it may be explored in future iterations.
No integration with existing hospital management software systems during the initial project phase.
The project will not attempt to address broader healthcare policy or financial decisions related to hospital management.

![image](https://github.com/user-attachments/assets/c12e3d5b-493f-478f-ad97-0573a1f51bed)

Repo Structure:
Final_Project/AAI-540_Group-8_notebook.ipynb -> Notebook with all implementation <br>
Final_Project/code -> Generated pipeline scripts <br>
Final_Project/dataset -> Dataset used.

Developement done primarily on AWS. Any data used or generated was stored in S3.
