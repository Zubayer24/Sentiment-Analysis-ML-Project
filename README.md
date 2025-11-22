# 🔹Sentiment Analysis on Amazon Product Reviews

## 1. Project Title
**Sentiment Analysis on Amazon Product Reviews Using Machine Learning**

## 2. Problem Summary
Understanding customer sentiment is crucial for e-commerce businesses to improve products, services, and overall customer satisfaction. By automatically classifying Amazon product reviews as positive, neutral, or negative, companies can make data-driven decisions, detect potential product issues early, and optimize marketing strategies. This project demonstrates how machine learning can be applied to analyze text reviews efficiently and accurately.

## 3. What You Did (Your Approach)
- **Data Cleaning:** Removed duplicates, missing values, and irrelevant characters.  
- **Text Preprocessing:** Lowercasing, stopword removal, tokenization, and lemmatization.  
- **Feature Engineering:** Applied TF-IDF vectorization to convert text into numerical features.  
- **Model Selection & Training:** Tested Logistic Regression, Random Forest, and XGBoost models.  
- **Evaluation:** Measured performance using accuracy, precision, recall, F1-score, and confusion matrices.

## 4. Key Results (Visuals / Numbers)

**Multinomial Naive Bayes Model Performance:**  
- Accuracy: 0.88

**Classification Report:**

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.71      | 0.81   | 0.76     | 1430    |
| 1     | 0.94     | 0.90   | 0.92     | 4570     |

**Visuals:**  
- Confusion Matrix  
- ROC Curve  
- Accuracy / Precision / Recall Table  

*Visuals can be embedded here for a more professional look.*

## 5. Live Notebook Viewer Link
- View Notebook on **NBViewer** → [NBViewer Link](https://nbviewer.org/github/Zubayer24/Sentiment-Analysis-ML-Project/blob/main/Sentiment%20Analysis%20on%20Amazon%20Product%20Reviews.ipynb)  

## 6. GitHub Repository Link
- Access the full project code → [GitHub Repository](https://github.com/Zubayer24/Sentiment-Analysis-ML-Project)

---

## 7. How to Run
1. Clone the repository:  
   ```bash
   git clone https://github.com/Zubayer24/Sentiment-Analysis-ML-Project.git
