# **Sentiment Analysis on Drug Reviews**  

## **Overview**  
This project applies **machine learning** and **natural language processing (NLP)** techniques to analyze sentiment in patient drug reviews. By leveraging structured and unstructured data, the model classifies patient sentiment based on drug effectiveness, side effects, and overall experience. The findings help identify patterns in patient satisfaction and assess the impact of side effects on drug perception.

## **Key Features**  
✔ **Sentiment Classification** – Categorizes drug reviews as positive or negative using machine learning models.  
✔ **Side Effects Analysis** – Examines the relationship between side effect severity and patient satisfaction.  
✔ **Text Processing with TF-IDF** – Converts raw text into structured numerical features for analysis.  
✔ **Machine Learning Models** – Implements **Logistic Regression, Random Forest, and Support Vector Machine (SVM)** for sentiment prediction.  
✔ **Performance Evaluation** – Compares models using **AUC (ROC Curve) and accuracy metrics** to determine effectiveness.  

## **Technical Stack**  
- **Programming Language**: Python  
- **Data Processing**: Pandas, NumPy  
- **Text Vectorization**: TF-IDF (TfidfVectorizer from Scikit-learn)  
- **Feature Selection**: Chi-Square Test  
- **Machine Learning Models**: Logistic Regression, Random Forest, Support Vector Machine (SVM)  
- **Model Evaluation**: ROC Curve, Accuracy, AUC Score  
- **Data Visualization**: Matplotlib, Seaborn  

## **Usage**  
1. **Data Preparation** – Load and clean patient drug reviews, ensuring proper sentiment labeling.  
2. **Feature Engineering** – Apply **TF-IDF** for text vectorization and select key features for classification.  
3. **Model Training & Evaluation** – Train **Logistic Regression, Random Forest, and SVM**, and compare their performance using **AUC and accuracy metrics**.  
4. **Side Effects Analysis** – Investigate correlations between side effects and patient satisfaction using **Spearman correlation and visualization techniques**.  

## **Results & Insights**  
📊 **Best Performing Model** – **SVM achieved the highest accuracy (82.1%) and AUC score (0.82), outperforming other models.**  
📉 **Impact of Side Effects** – A strong negative correlation (-0.63) was found between side effect severity and drug ratings.  
💡 **Future Improvements** – Implementing **deep learning models (e.g., BERT)** and **Patient-Reported Outcome Measures (PROMs)** can enhance sentiment classification and patient insights.  

## **Future Work**  
🔹 Experiment with deep learning techniques such as **BERT-based sentiment analysis**.  
🔹 Integrate **longitudinal tracking** of patient sentiment for real-time drug performance monitoring.  
🔹 Explore **unsupervised clustering** to detect emerging trends in patient satisfaction.  
