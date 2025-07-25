
# 💬 Toxic Comment Classification System

This project detects multiple types of toxicity (toxic, severe_toxic, obscene, threat, insult, identity_hate) in user comments using a deep learning model built with **Bidirectional GRU** and **GloVe word embeddings**.

---

## 🧠 Model

- **Embedding**: GloVe (100D, 6B tokens)
- **Architecture**: Bidirectional GRU → Dense layers
- **Task**: Multi-label classification (6 toxicity categories)
- **Dataset**: Jigsaw Toxic Comment Classification Challenge

---

## 📥 Download Dataset

🔗 [Jigsaw Toxic Comment Classification Dataset on Kaggle](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data)

After downloading:

1. Place the following files inside a `data/` directory:
   - `train.csv`
   - `test.csv`
   - `test_labels.csv`

Your structure should look like:

```
data/
├── train.csv
├── test.csv
└── test\_labels.csv
```

> ⚠️ These files are not included in the repository because of GitHub’s 100MB limit.

---

## 📁 Project Structure

```
toxic-comment-classification-system/
├── app.py                  # Streamlit app
├── models/
│   ├── final\_model/        # Trained SavedModel directory
│   └── final\_tokenizer.pkl
├── data/
│   ├── train.csv           # Place manually from Kaggle
│   ├── test.csv
│   └── test\_labels.csv
├── requirements.txt
└── README.md
````

---

## ⚙️ Setup Instructions

1. Clone the repo  
```bash
git clone https://github.com/yourusername/toxic-comment-classifier.git
cd toxic-comment-classifier
````

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app

```bash
streamlit run app.py
```

---

## ✨ Features

*  Multi-label toxicity classification
*  Threshold slider for predictions
*  Color-coded UI with emoji indicators
*  GloVe 100D embedding integration
*  GRU-based model with tokenizer

---

## 🛠️ Future Enhancements

*  Upload CSV batch prediction
*  SHAP/attention-based explainability
*  Deploy to Hugging Face or Streamlit Cloud

---

## 📚 Socials

© 2025 Lakshay Jain
https://www.linkedin.com/in/lakshay-jain-a48979289/

