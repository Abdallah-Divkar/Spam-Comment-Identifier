# YouTube Comment Spam Detection

This repository contains a Python-based natural language processing project to classify YouTube comments as spam or non-spam. The workflow involves data preprocessing, feature extraction, model training, and evaluation using a Multinomial Naive Bayes classifier.

---

## **Features**
- **Data Cleaning:** Preprocesses text by removing stop words, contractions, and escape characters.
- **Feature Extraction:** Uses CountVectorizer and TF-IDF transformation for text representation.
- **Model Training:** Employs Multinomial Naive Bayes for classification.
- **Evaluation:** Includes confusion matrix, accuracy score, and 5-fold cross-validation.
- **Prediction:** Tests the model on new comments to determine spam or non-spam.

---

## **Getting Started**

### **Prerequisites**
- Python 3.7 or higher
- Required Python libraries: `pandas`, `scikit-learn`, `pathlib`, `re`

### **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/spam-detection.git
