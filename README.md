<video src="https://github.com/user-attachments/assets/5680d975-0023-4be2-9f11-4e706882aed7" width="100%" controls></video>



# Medical-Nlp-Analyzer

> NLP pipeline for drug review sentiment analysis and named entity recognition.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Accuracy](https://img.shields.io/badge/Accuracy-84%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Dataset](https://img.shields.io/badge/Dataset-UCI-orange)

---

## Overview

MedSense analyzes patient drug reviews using NLP to classify sentiment and extract medical entities. Built on 161,297 real patient reviews from the UCI Drug Review Dataset.

---

## Demo

```
Input:  "I have been taking Metformin for 3 months and my blood
         sugar levels have significantly improved with no side effects."

Output: Sentiment  →  POSITIVE
        Entities   →  Metformin
```

---

## Features

- Sentiment Classification — Positive / Negative / Neutral
- Named Entity Recognition — Extracts drug and medical terms
- 84% Model Accuracy — TF-IDF + Logistic Regression
- Interactive UI — Built with Streamlit
- Clean Prediction API — predict.py for easy integration

---

## Dataset

**UCI Drug Review Dataset**
- Source: UCI Machine Learning Repository (ID: 462)
- Size: 161,297 patient reviews
- License: CC BY 4.0

| Column | Description |
|--------|-------------|
| drugName | Name of the drug |
| condition | Medical condition |
| review | Patient review text |
| rating | Score 1-10 |

---

## Project Structure

```
medical-nlp-analyzer/
├── data/                        
├── notebooks/
│   ├── EDA.ipynb                
│   ├── preprocessing.ipynb      
│   ├── sentiment_model.ipynb    
│   └── ner_pipeline.ipynb       
├── src/
│   └── predict.py               
├── app/
│   └── app.py                   
├── models/                      
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Model Performance

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Positive | 0.86 | 0.96 | 0.91 |
| Negative | 0.78 | 0.77 | 0.78 |
| Neutral | 0.61 | 0.12 | 0.20 |
| **Accuracy** | | | **84%** |

---

## Installation

```bash
git clone https://github.com/hassan-ali786/medical-nlp-analyzer.git
cd medical-nlp-analyzer
pip install -r requirements.txt
python download_data.py
streamlit run app/app.py
```

---

## Usage

```python
from src.predict import predict_sentiment, extract_entities

predict_sentiment("Aspirin worked great, no side effects!")
# Output: 'positive'

extract_entities("I took Metformin for my diabetes.")
# Output: [('Metformin', 'PERSON')]
```

---

## Test Reviews

| Review | Expected Output |
|--------|----------------|
| "Metformin has been a lifesaver. My sugar levels are finally under control." | Positive |
| "Lisinopril made me feel dizzy and extremely tired. I stopped after two weeks." | Negative |
| "I have been on Sertraline for a month. Some days better, some not." | Neutral |
| "Atorvastatin brought my cholesterol down significantly. Doctor is very pleased." | Positive |
| "Amoxicillin gave me severe stomach pain and rashes. Would not recommend." | Negative |

---

## Author

**Hassan Ali** — Data Scientist & ML Engineer

[![GitHub](https://img.shields.io/badge/GitHub-hassan--ali786-black)](https://github.com/hassan-ali786)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-hassan--ali--datascientist-blue)](https://linkedin.com/in/hassan-ali-datascientist)
[![Kaggle](https://img.shields.io/badge/Kaggle-hassanali789-20BEFF)](https://kaggle.com/hassanali789)

---

---

## License

MIT License — Dataset: CC BY 4.0
