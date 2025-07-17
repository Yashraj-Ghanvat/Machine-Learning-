# Heart Disease Prediction using Machine Learning  

## Overview  
This project is a *machine learning classification model* that predicts whether a person has *heart disease* based on medical attributes. The model is integrated into a *Streamlit web application* for user-friendly interaction.  

## Dataset  
The dataset used in this project includes various medical features such as:  
- Age  
- Sex  
- Chest Pain Type  
- Resting Blood Pressure
- Serum Cholesterol
- Fasting Blood Sugar 
- Resting Electrocardiographic Results
- Maximum Heart Rate Achieved 
- Exercise-Induced Angina
- ST Depression Induced by Exercise 
- Slope of the Peak Exercise ST Segment  
- Number of Major Vessels Colored by Fluoroscopy 
- Thalassemia
- target
## Technologies Used  
- *Python*  
- *Pandas, NumPy* (Data Preprocessing)  
- *Scikit-Learn* (Machine Learning)  
- *Matplotlib, Seaborn* (Visualization)  
- *Streamlit* (Web App)  

## Model Details  
The classification model is trained using different algorithms, including:  
- Logistic Regression  
- Random Forest Classifier  
- Support Vector Machine (SVM)  
- Decision Tree  
- K-Nearest Neighbors (KNN)  

The best-performing model is selected based on evaluation metrics like *accuracy, precision, recall, and F1-score*.  

## Installation  
To run this project locally, follow these steps:  

1. Clone the repository:  
   ```bash
   git clone https://github.com/YashrajGhanvat1475/Machine-Learning-/new/main/Heart%20project
2. Install dependencies:
   pip install -r requirements.txt
3. Run the Streamlit app:
   streamlit run app.py

## Usage : 
  Open the Streamlit web app.
  Enter the required medical details in the input fields.
  Click Predict to get the result.
  The app will display whether the person is predicted to have heart disease or not.
## Results & Performance : 
  The model achieved an accuracy of (based on evaluation).
  Confusion matrix, ROC curve, and other performance metrics are available for analysis.
## Future Improvements : 
  Enhance feature selection for better accuracy.
  Integrate deep learning models.
  Deploy the app on cloud services like Heroku or AWS
## Contributors
  Contributions are welcome!
## License
  This project is licensed under the MIT License.
