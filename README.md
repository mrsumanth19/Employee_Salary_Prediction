# 👨‍💼 Employee Salary Prediction using Machine Learning

This project predicts employee salaries based on features like experience, education level, job role, and more. Built using Python, Streamlit, and machine learning models from scikit-learn.

## 🔍 Overview

This app helps HR departments and companies estimate salary ranges for employees or new candidates using a machine learning model trained on employee data.

- 🔧 Built with: **Python**, **scikit-learn**, **Streamlit**
- 📊 Model: Trained with **Linear Regression**
- 🧠 Features used: Experience, Test Score, Interview Score, etc.
- 📈 Scales features using MinMaxScaler
- 🧮 Encodes categorical data using LabelEncoder

## 🚀 Live App 
 [Click here to use the app](https://employeesalaryprediction-8.streamlit.app/)

---


## 🖥️ Screenshots

<img width="1918" height="869" alt="image" src="https://github.com/user-attachments/assets/f3bfcc22-7cc1-4eb8-ac4b-c1792999dbbf" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/304be2a6-4e83-4ece-8676-cf460b86ba59" />
<img width="941" height="786" alt="image" src="https://github.com/user-attachments/assets/5686d76b-bfc7-47ce-97d6-850904917d4c" />
<img width="941" height="786" alt="image" src="https://github.com/user-attachments/assets/9c762fe9-3a32-4137-9305-046445a00578" />
<img width="938" height="756" alt="image" src="https://github.com/user-attachments/assets/31a3266d-a73a-488c-87c3-dc8da773a9b6" />






## 📁 Project Structure

```

Employee\_Salary\_Prediction/
│
├── app.py                  # Streamlit app script
├── model.pkl               # Trained ML model
├── data.csv                # Dataset used for training
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

````

## ⚙️ How to Run Locally

1. **Clone the repository**

```bash
git clone https://github.com/mrsumanth19/Employee_Salary_Prediction.git
cd Employee_Salary_Prediction
````

2. **Create a virtual environment**

```bash
python -m venv .venv
source .venv/Scripts/activate    # Windows
# OR
source .venv/bin/activate        # Mac/Linux
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the app**

```bash
streamlit run app.py
```

## 🧠 Model Training

The model is trained on the dataset (`data.csv`) using Linear Regression. It uses encoded and scaled input features to predict salaries and is saved as `model.pkl` using `joblib`.

## 🛠️ Tools & Technologies

* Python
* Streamlit
* Pandas
* scikit-learn
* NumPy
* Matplotlib / Seaborn (optional for visualizations)

## 📦 Deployment

You can deploy this app on [Streamlit Cloud](https://streamlit.io/cloud) by connecting your GitHub repo and setting the main file to `app.py`.

---

## 🤝 Contributing

Feel free to fork the project, open issues, or submit pull requests to improve this app.

## 📄 License

This project is licensed under the MIT License.

---

## 🙋‍♂️ Author

Made with ❤️ by [Sumanth](https://github.com/mrsumanth19)

````

---

