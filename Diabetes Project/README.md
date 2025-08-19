Diabetes Prediction Web App
This project is a simple web application built with Streamlit that predicts the likelihood of a person having diabetes based on several medical features. It uses a pre-trained machine learning model to make real-time predictions through a user-friendly interface.

🚀 Features
Interactive UI: A clean and simple user interface for data input.

Real-time Predictions: Get instant diabetes predictions based on the input data.

Easy to Use: Simply enter the patient's details in the sidebar and click "Predict".

Scikit-learn Integration: Utilizes a pickled scikit-learn model and scaler for prediction.

📋 Project Structure
For the application to work correctly, your project folder should be organized as follows. Make sure the model and scaler files are in the same directory as your app.py script.

your-project-folder/
├── app.py                  # Main Streamlit application script
├── diabetes_model.pickle   # Pre-trained classification model
├── scaler.pkl              # Pre-trained scaler
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

⚙️ Installation & Setup
Follow these steps to set up and run the project on your local machine.

1. Clone the Repository
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name

2. Create a Virtual Environment (Recommended)
It's a good practice to create a virtual environment to manage project dependencies.

# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

3. Install Dependencies
Create a file named requirements.txt in your project directory and add the following lines:

streamlit
numpy
scikit-learn==1.3.0 # Specify a version for consistency

Now, install these packages using pip:

pip install -r requirements.txt

⚠️ 4. IMPORTANT: Update File Paths in app.py
The current app.py script uses absolute file paths, which will cause errors on any other computer. You must change them to relative paths.

Open app.py and modify these lines:

Change this:

model_path = r'C:\Users\Admin\Desktop\Yashraj\ML\Diabetes Project\diabetes_model.pickle'
scaler_path = r'C:\Users\Admin\Desktop\Yashraj\ML\Diabetes Project\scaler (1).pkl'

# and the paths inside the infer_diabetes function
with open(r'C:\Users\Admin\Desktop\Yashraj\ML\Diabetes Project\diabetes_model.pickle', 'rb') as model_file:
    # ...
with open(r'C:\Users\Admin\Desktop\Yashraj\ML\Diabetes Project\scaler (1).pkl', 'rb') as scaler_file:
    # ...

To this:

# Use relative paths for the model and scaler
model_path = 'diabetes_model.pickle'
scaler_path = 'scaler.pkl' # Make sure your scaler file is named this way

# ... inside the infer_diabetes function
try:
    # Load the trained model and scaler using the relative paths
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
# ... rest of the code

▶️ How to Run the App
Once you have completed the setup and corrected the file paths, run the following command in your terminal:

streamlit run app.py

This command will start the application, and it should automatically open in a new tab in your web browser.

Usage
The application will display a sidebar with input fields for various medical attributes.

Enter the values for each feature:

Pregnancies: Number of times pregnant

Glucose: Plasma glucose concentration

Blood Pressure: Diastolic blood pressure (mm Hg)

Skin Thickness: Triceps skin fold thickness (mm)

Insulin: 2-Hour serum insulin (mu U/ml)

BMI: Body mass index

Diabetes Pedigree Function: A function that scores likelihood of diabetes based on family history

Age: Age in years

Click the "Predict Diabetes" button.

The prediction result ("Diabetes Detected" or "No Diabetes") will be displayed on the main page.
