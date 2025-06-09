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
- Handles feature engineering, class balancing, and model optimization



### `outputs/`
Stores visual results of model training:
- Evaluation reports, confusion matrices, and performance visualizations


## Highlights

- Binary classification based on 21-day price return (> ±1%)
- Total of 16 features: technicals & fundamentals & ESG
- Random Forest achieved **88% accuracy**

## Team Members

Victoria Zhang, Jennifer Liu, Rafi Biswas, Jashan Singh, Stephanie Espinoza







