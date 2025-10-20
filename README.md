# 📄 Task 5 — Text Classification on Consumer Complaint Dataset

**Project:** Binary Text Classification (Credit reporting, repair, or 
other vs Debt collection vs Mortgage vs Consumer Loan  )

**Author:** Srujana

---

## 📘 Overview

This project demonstrates **text classification** on the **Consumer Complaint Database**, a public dataset from the U.S. government. The goal is to predict the type of complaint based on the customer’s written text.

In the dataset used, there are **18 columns**, but only two product categories are available:

* Debt collection
* Mortgage

Thus, the project performs **binary classification**.

The notebook (`Task5.ipynb`) completes the task in **10 major steps**.

---

## 🚀 Step-by-Step Workflow

### **Step 1: Importing Required Libraries** 🧩

All necessary Python libraries for data analysis, visualization, text preprocessing, and model training are imported. These include:

* `pandas`, `numpy` for data manipulation
* `matplotlib`, `seaborn` for visualization
* `nltk` for text cleaning and stopwords
* `sklearn` for vectorization, model building, and evaluation

### **Step 2: Load the Dataset** 📥

The Consumer Complaint CSV file is loaded into a pandas DataFrame using `pd.read_csv()`. Basic info like number of rows, columns, and sample entries are displayed using `.info()` and `.head()`.

### **Step 3: Exploratory Data Analysis (EDA)** 🔍

Performed EDA to understand the data distribution and check for missing values:


<img width="2383" height="1106" alt="Screenshot 2025-10-20 170633" src="https://github.com/user-attachments/assets/73a350ce-7397-49a0-bdb6-9fe971e6a199" />

* Displayed count of records per product category.
* Visualized category counts using a bar chart.
* Identified that the available categories are **Debt collection** and **Mortgage**.

### **Step 4: Filtering and Label Encoding** 🎯

Filtered the dataset to include only the two target categories. Then, product names were mapped to numeric labels:

```
Debt collection → 0
Mortgage → 1
```

A new column `label` was created for the model.

### **Step 5: Text Preprocessing** ✂️🧹

The complaint text column (`Consumer complaint narrative`) was cleaned to remove noise:

* Converted text to lowercase
* Removed numbers and punctuation
* Removed English stopwords using `nltk`

This ensures the model learns meaningful patterns from clean text.



### **Step 6: Train-Test Split** 🧪

The data was split into **training** and **testing** sets (typically 80% train, 20% test) using `train_test_split()` from sklearn.
This helps evaluate model performance on unseen data.

### **Step 7: Model Selection and Training** 🤖

Three machine learning models were trained for comparison:

* Logistic Regression
* Naive Bayes (MultinomialNB)
* Random Forest

Each model was trained using the training data, and predictions were made on the test data.

### **Step 8: Model Evaluation** 📊

<img width="2468" height="1202" alt="Screenshot 2025-10-20 170556" src="https://github.com/user-attachments/assets/7f02fc7c-38a0-44c6-a9cb-7562250ed1b3" />

Models were evaluated using standard classification metrics:

* Accuracy Score
* Precision, Recall, F1-score (`classification_report`)
* Confusion Matrix

A bar plot compared the accuracy of all three models, identifying the best performer (usually **Logistic Regression** for text data).

### **Step 9: Prediction Function** 🔮

A custom function `predict_category(text)` was implemented to predict the complaint category for any new text input.
Steps inside the function:

1. Clean the input text
2. Transform it with the trained TF-IDF vectorizer
3. Predict the label using the trained model
4. Return the corresponding category name

**PREDICTED OUTPUT : 'Mortgage'**

Example:

```python
sample_text = "My mortgage payment was incorrectly reported to the credit bureau."
print(predict_category(sample_text))  # Output: 'Mortgage'
```

---

## 🧰 Tools and Libraries Used

* Python 🐍
* pandas, numpy
* matplotlib, seaborn
* scikit-learn
* nltk

To install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk
```

---

## 📈 Results Summary

| Model               | Accuracy                |
| ------------------- | ----------------------- |
| Logistic Regression | ~High Accuracy          |
| Naive Bayes         | Moderate                |
| Random Forest       | Lower (text-heavy data) |



The **Logistic Regression** model gave the best overall accuracy and F1-score for this binary text classification task.

---

## 📚 Project Summary

✅ Loaded dataset (18 columns)
✅ Performed EDA and visualization
✅ Cleaned and preprocessed complaint text
✅ Converted text to TF-IDF numeric form
✅ Trained multiple models for comparison
✅ Evaluated using accuracy, precision, recall, F1
✅ Implemented final prediction function

---

## 💡 Conclusion

The notebook demonstrates an end-to-end workflow of a **text classification pipeline** for consumer complaints. Even users with no data science background can follow it easily to reproduce the results.

This project can be extended to multi-class classification if the dataset includes more product types (like Credit Reporting and Consumer Loans).

---

Thanks for checking out this project! ✨
