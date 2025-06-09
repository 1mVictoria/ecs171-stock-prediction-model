# ECS-171-Group18 Final Project - Stock Action Prediction

## Project Structure

### `data_Cleaning/`
Contains cleaned and preprocessed datasets, including:
- `prices_cleaned.csv`
- `fundamentals_cleaned.csv`
- `esgRisk_cleaned.csv`

These files serve as input features for training the models.


### `src/`

This directory contains the training and evaluation scripts for all three models. Each script:

- Loads the cleaned CSVs (`prices_cleaned.csv`, `fundamentals_cleaned.csv`, `esgRisk_cleaned.csv`)  
- Engineers features  
- Splits into train/test  
- Fits the model (with optional hyper-parameter search)  
- Outputs performance metrics, plots, and a serialized model



## Linear Regression

```bash
python src/linear_model.py 
```
## Logistic Regression
```bash
python src/logistic_model.py 
```

## Random Forest
```bash
cd src
python train_random_forest.py 
```
---



## Key Points of the Project

- Binary classification based on 21-day (around 1 month) price return (> ±1%)
- Random Forest achieved **88% accuracy**








