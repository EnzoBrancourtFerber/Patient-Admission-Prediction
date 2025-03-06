# Patient Admissions Prediction

This project aims to develop a machine learning model to predict patient admissions in hospitals. By analyzing historical data, the model will help healthcare providers anticipate patient inflow, optimize resource allocation, and improve patient care.

## Project Structure

- **data/**
  - **raw/**: Contains the raw data files used for training the machine learning model.
  - **processed/**: Contains the processed data files that have been cleaned and transformed for use in model training.
  
- **notebooks/**
  - **exploratory_analysis.ipynb**: Jupyter notebook for exploratory data analysis (EDA) to understand the dataset, visualize trends, and identify patterns related to patient admissions.
  
- **src/**
  - **data_preprocessing.py**: Functions for data cleaning and preprocessing, including handling missing values, encoding categorical variables, and normalizing numerical features.
  - **model_training.py**: Implementation of the machine learning model training process, defining functions to train the model using the processed data and save the trained model.
  - **model_evaluation.py**: Functions for evaluating the performance of the trained model, including metrics such as accuracy, precision, recall, and F1 score.
  - **utils.py**: Utility functions used across the project, such as loading data, saving models, and other helper functions.
  
- **models/**
  - **model.pkl**: Serialized version of the trained machine learning model, saved for later use in predictions.
  
- **requirements.txt**: Lists the Python dependencies required for the project, including libraries such as pandas, scikit-learn, and any other necessary packages.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd patient-admissions-prediction
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Prepare the data:
   - Place raw data files in the `data/raw/` directory.
   - Run the data preprocessing script to clean and transform the data.

4. Train the model:
   - Execute the model training script to train the machine learning model and save it to `models/model.pkl`.

5. Evaluate the model:
   - Use the model evaluation script to assess the performance of the trained model.

## Usage Examples

- To run exploratory data analysis, open the Jupyter notebook located in the `notebooks/` directory.
- Use the functions in `src/model_training.py` to train the model with your processed data.
- Evaluate the model's performance using the functions in `src/model_evaluation.py`.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or suggestions.