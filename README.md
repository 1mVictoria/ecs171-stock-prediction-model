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



## Key Points of the Project

- Binary classification based on 21-day (around 1 month) price return (> ±1%)
- Random Forest achieved **88% accuracy**








