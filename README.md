# 🛡️ Fake News Detection System

## 📌 Overview
This project is an AI-powered Fake News Detection system that classifies news articles as REAL or FAKE using Natural Language Processing (NLP) and Machine Learning techniques. It provides real-time predictions through a Streamlit web interface.

---

## 🚀 Features
- Real-time news classification  
- Confidence score for predictions  
- Interactive Streamlit UI  
- Fast and lightweight ML model  
- Modular project structure using `src/`  

---

## 🛠️ Tech Stack
- Python  
- Scikit-learn  
- Streamlit  
- NLP (TF-IDF Vectorization)  
- pandas, numpy  

---
## 📁 Dataset

The dataset used in this project is not included in the repository due to size limitations.

You can download the dataset from the following source:

👉 https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

### 📌 Instructions
1. Download the dataset from the above link  
2. Extract the files  
3. Place them inside a `data/` folder in the project directory  

Expected structure:


## 📂 Project Structure
```
fake_news_detector/
│
├── app.py
├── models/
│   ├── model.pkl
│   ├── vectorizer.pkl
│
├── src/
│   ├── __init__.py
│   ├── train.py
│   ├── predict.py
│   ├── preprocess.py
│
├
├── requirements.txt
└── README.md
```

---

## ⚙️ How It Works
1. Text is cleaned and preprocessed  
2. TF-IDF converts text into numerical features  
3. Logistic Regression model classifies news  
4. Streamlit displays prediction with confidence  

---

## ▶️ How to Run

### 1. Clone the repository
```
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector
```

### 2. Create virtual environment
```
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Train the model
```
python src/train.py
```

### 5. Run the app
```
streamlit run app.py
```

---

## 📊 Example Output
- FAKE NEWS (Confidence: 0.55)  
- REAL NEWS (Confidence: 0.87)  

---

## 🎯 Future Improvements
- Upgrade to BERT / Transformers  
- Add explainability (LIME/SHAP)  
- Deploy on cloud  
- URL-based detection  

---

## 👨‍💻 Author
Mehar Arora  

---

## ⭐ Note
This project is for educational purposes and may not always provide accurate real-world predictions.
