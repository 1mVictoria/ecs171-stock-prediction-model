# ECS-171-Group18 Final Project - Stock Action Prediction

## Project Structure

### `data_Cleaning/`
Contains cleaned and preprocessed datasets, including:
- `prices_cleaned.csv`
- `fundamentals_cleaned.csv`
- `esgRisk_cleaned.csv`

These files serve as input features for training the models.


### `src/`
Hold the programming code for training and evaluating models:
- Includes implementation for Linear Regression, Logistic Regression, and Random Forest
- Handles feature engineering, model training and optimization

  ## Training the Random Forest Model

All code lives in `src/train_random_forest.py`.  This script implements:

1. Loading and merging cleaned price, fundamentals, and ESG CSVs  
2. Engineering 9 technical + 3 fundamental + 4 ESG features  
3. Splitting the data, fitting a RandomForestClassifier (with optional Balanced RF or RandomizedSearchCV)  
4. Reporting CV balanced-accuracy, top-10 feature importances, classification report, and confusion matrix  
5. Saving the trained model to `outputs/RF_model_binary.pkl` and figures under `outputs/figures/`

### Prerequisites

- Python 3.8+  
- pandas 1.5.3  
- numpy 1.24.3  
- scikit-learn 1.4.0  
- imbalanced-learn 0.10.1  
- matplotlib, seaborn  


### Usage

```bash
cd src
python train_random_forest.py \
    --data_dir ../data_Cleaning/cleaned \
    --output_dir ../outputs \
    --n_estimators 200 \
    --max_depth None \
    --min_samples_leaf 3 \
    --random_search   # add this flag to run RandomizedSearchCV




## Key Points of the Project

- Binary classification based on 21-day (around 1 month) price return (> ±1%)
- Random Forest achieved **88% accuracy**








