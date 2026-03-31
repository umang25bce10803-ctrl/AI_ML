# 📥 Self-Contained Spam Detection System

A lightweight, zero-dependency data file Python script to detect spam messages using Natural Language Processing (NLP) and Machine Learning. The project includes a built-in dataset and automatic visual reporting.

## 📝 Description
This Python script builds a complete **spam detection system** using a self-contained dataset of 40 varied examples. It uses **TfidfVectorizer** to convert text and **Multinomial Naive Bayes** to classify it as 'Ham' or 'Spam'. Finally, it auto-generates a dataset pie chart and a model confusion matrix using **Matplotlib**.

## ✨ Key Features
- **Zero File Uploads:** Works entirely out of the box with 40 hardcoded training examples (20 Ham / 20 Spam).
- **Machine Learning Pipeline:** Utilizes TF-IDF vectorization and the Naive Bayes algorithm for high-speed text classification.
- **Visual Analytics:** Generates a pie chart to visualize class distributions and a confusion matrix to evaluate model accuracy.
- **Custom Prediction Hook:** Includes a reusable function to test your own custom strings.

## 🚀 Getting Started

### Prerequisites
Make sure you have Python 3.x installed. You will also need to install the required data science libraries:

```bash
pip install pandas scikit-learn matplotlib
