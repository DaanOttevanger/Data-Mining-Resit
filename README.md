# Fake Review Detection

This project focuses on detecting fake reviews using machine learning. The project follows the CRISP-DM methodology and includes data preparation, modelling, evaluation and a small Streamlit prototype.

## Project structure

- fake_review_combined_dataset.ipynb  
  Data preparation notebook. In this notebook, six datasets are cleaned, standardised and combined into one dataset.

- 03_modelling_fake_reviews.ipynb  
  Modelling notebook. In this notebook, several machine learning models are compared using TF-IDF features.

- app.py  
  Streamlit prototype. This app uses the saved best model to predict whether a new review is fake or real.

- models/best_fake_review_pipeline.pkl  
  Saved best-performing model pipeline.

- models/best_model_info.json  
  Information about the selected best model.

- data/processed/label_mapping_overview.csv  
  Overview of the label mappings used during data preparation.

## Important note about datasets

The full combined dataset is not included in this GitHub repository because it is larger than GitHub's file size limit of 100 MB. The notebooks show how the data was prepared and used.

## How to install requirements

Run:

pip install -r requirements.txt

## How to run the Streamlit app

Run:

python -m streamlit run app.py

## CRISP-DM structure

The project follows these CRISP-DM phases:

1. Business understanding  
2. Data understanding  
3. Data preparation  
4. Modelling  
5. Evaluation  
6. Deployment
