

# CEFR Text Difficulty Analyzer

## 📖 Project Overview
The **CEFR Text Difficulty Analyzer** is a tool designed to automatically evaluate the reading difficulty of English texts based on the CEFR scale, ranging from A1 to C2. 

Available as both an **application** and a **browser extension**, this project aims to help English language learners easily find reading materials perfectly tailored to their current proficiency level.

### 🎯 Who is this for?
This tool is created for **English language learners** who struggle to find appropriate articles, blogs, or books to read. It solves the frustrating experience of starting an article only to realize halfway through the first paragraph that the vocabulary is too advanced. By checking the text beforehand, learners can ensure the material matches their skills, keeping them motivated and making the learning process much smoother.

### 🚀 Key Features (_note_: this section is still under development)
* **Multiple Input Methods:** Users can paste plain text directly into the application, submit a URL for analysis, upload text file, or use the browser extension to instantly evaluate the text on their current webpage.
* **Linguistic Analysis:** The algorithm evaluates the text based on:
  * Vocabulary difficulty (word frequency and rarity)
  * Grammatical complexity
  * Average sentence length and structure
* **Detailed Grading System:** The tool provides:
  * An overall CEFR score for the text.
  * A detailed percentage breakdown of the text's composition (e.g., *A1: 45%, A2: 30%, B1: 15%, B2: 10%*), giving users a granular look at what to expect before they start reading.
 
# 📁 Repository Structure


```text
├── data/
│   ├── raw/               <- Immutable raw data source files
│   └── processed/         <- Cleaned, transformed data ready for modeling
├── models/                <- Serialized model pipelines and artifacts
├── notebooks/             <- Jupyter notebooks for EDA and experimentation
├── src/
│   ├── __init__.py
│   └── train.py           <- Automated training and evaluation pipeline
├── app/
│   └── app.py             <- Streamlit web dashboard for live inference
├── .gitignore             <- Pre-configured Git exclusion rules
├── requirements.txt       <- Python dependencies
└── README.md
```


