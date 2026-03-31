import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --- Embedded Data ---
data = {'text': ["Sample ham..."] * 20 + ["Sample spam..."] * 20, # Placeholder for brevity
        'label': [0]*20 + [1]*20}
df = pd.DataFrame(data)

# --- Train Model ---
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.25)
vectorizer = TfidfVectorizer(stop_words='english')
model = MultinomialNB().fit(vectorizer.fit_transform(X_train), y_train)
y_pred = model.predict(vectorizer.transform(X_test))

# --- Matplotlib Visualizations ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 1. Pie Chart
df['label'].value_counts().plot.pie(ax=axes[0], autopct='%1.1f%%', colors=['#a8e6cf', '#ff8b94'])
axes[0].set_title("Dataset Distribution (Ham vs Spam)")

# 2. Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=['Ham', 'Spam'])
disp.plot(ax=axes[1], cmap='Blues')
axes[1].set_title("Model Confusion Matrix")

plt.show()
