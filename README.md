# 🧠 Software Defect Prediction Using Machine Learning

This project predicts software defects using various Machine Learning algorithms.  
It helps improve software quality by identifying defective modules early in the development lifecycle.

---

## 🚀 Project Overview

Software defect prediction (SDP) uses historical software metrics to automatically classify modules as **defective** or **non-defective**.  
This project includes:

- Data pre-processing  
- Feature engineering  
- Training multiple ML models  
- Evaluation & comparison  
- Django-based Web Interface for predictions  

---

## 📁 Project Structure

project/
│── code/
│ ├── manage.py
│ ├── admins/
│ ├── users/
│ ├── templates/
│ ├── static/
│ ├── models/
│ ├── views.py
│ └── ML_model.pkl
│
│── Base Paper/
│── Abstract/
│── Project Document.pdf
│── README.md

yaml
Copy code

---

## 🧪 Machine Learning Models Used

- Logistic Regression  
- Random Forest  
- Decision Tree  
- Support Vector Machine  
- K-Nearest Neighbors  

The best performing model is saved as:

ML_model.pkl

yaml
Copy code

---

## ⚙️ Requirements

Install the required packages:

pip install -r requirements.txt

yaml
Copy code

### Main libraries:
- Python 3.9+
- Django
- NumPy
- Pandas
- Scikit-learn
- Joblib

---

## ▶️ How to Run the Project

1. **Clone the repository**
   ```bash
   git clone https://github.com/likhithamuddala/software-defect-prediction.git
Navigate into folder

bash
Copy code
cd software-defect-prediction/code
Migrate database

bash
Copy code
python manage.py migrate
Run server

bash
Copy code
python manage.py runserver
Open in browser:

cpp
Copy code
http://127.0.0.1:8000/
📊 Features
Admin & User login

Upload software metrics

Predict defect / non-defect

View prediction history

Model accuracy report

Clean UI

📈 Model Evaluation
Model	Accuracy
Logistic Regression	93%
Random Forest	95%
SVM	92%
Decision Tree	89%

(Random Forest performed the best)

🖼️ Screenshots
(Add your UI screenshots here)

👩‍💻 Author
Likhitha Muddala

GitHub: https://github.com/likhithamuddala

⭐ Contributing
Pull requests are welcome!

📜 License
This project is licensed under the MIT License.
