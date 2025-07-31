# 🧠 Sentiment Analysis with Naive Bayes & Streamlit

A complete end-to-end sentiment analysis project built using:

- ✅ Python
- ✅ TfidfVectorizer from `sklearn`
- ✅ Multinomial Naive Bayes
- ✅ Jupyter Notebook for training and evaluation
- ✅ Streamlit web app for deployment

---

## 📦 Features

- Trains a sentiment classifier from scratch using labeled text data (`__label__1`, `__label__2` format)
- Evaluates model performance (accuracy, confusion matrix, classification report)
- Provides a clean and interactive web interface via Streamlit
- Automatically trains and saves the model if not found on disk

---

### 📁 Project Structure

- 📄 **`sentiment_data.txt`** — Raw labeled data  
- 🧠 **`sentiment_model.py`** — Training & model logic  
- 🌐 **`streamlit_app.py`** — Web interface  
- 📒 **`notebook.ipynb`** — Jupyter workflow  
- 🧾 **`nb_model.pkl`** — Trained model (saved)  
- 📄 **`vectorizer.pkl`** — Tfidf Vectorizer (saved)  
- 📦 **`requirements.txt`** — Python deps  
- 📘 **`README.md`** — Project doc
""")         → Project documentation (this file)


---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/sentiment-analysis-naive-bayes-streamlit.git
cd sentiment-analysis-naive-bayes-streamlit

2️⃣ Install Dependencies
pip install -r requirements.txt

📊 Training the Model
You can train and test the model using the included notebook.ipynb or via the terminal:

▶️ Using Jupyter Notebook
Open notebook.ipynb
Run all cells to train, evaluate, and save:
nb_model.pkl
vectorizer.pkl

🌐 Running the Web App
streamlit run streamlit_app.py

✨ App Features
Paste any review or product text
Classifies the sentiment as positive or negative
Displays prediction confidence
Auto-trains model if .pkl files are missing

✅ Requirements
scikit-learn
streamlit
joblib
pandas

Install using:
Edit
pip install -r requirements.txt

📬 Contact
Created by Sunidhi Shukla
📧 ssunidhi136@gmail.com
