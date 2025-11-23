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



#### Linear Regression

```bash
cd src
python linear_model.py 
```
#### Logistic Regression
```bash
python src/logistic_model.py 
```

#### Random Forest
```bash
cd src
python train_random_forest.py 
```
---



## Notes

- Binary classification based on 21-day (around 1 month) price return (> ±1%)
- Best model: Random Forest (achieved **88% accuracy**)


### `web-interface/`
A simple Flask app that lets you enter a stock ticker, displays 16 key features (technical, fundamental, ESG), and uses a pre-trained Random Forest to recommend **Hold** or **Action (Buy/Sell)**.

#### Prerequisites

- Python 3.8+  
- Git (with LFS enabled, if you’re cloning rather than downloading a ZIP)  
- Virtual environment tool: `venv`, `virtualenv`, or `conda`

### Installation

1. Clone the repo** (or download & unzip)  
   ```bash
   git clone https://github.com/your_org/your_repo.git
   cd your_repo/web-interface
   ```
2. Create & activate a virtual environment
   ```bash
   python3 -m venv venv
   source venv/bin/activate    # macOS/Linux
   venv\Scripts\activate       # Windows
   ```
3. Install Python dependencies
   ```bash
   pip install -r requirements.txt
   ```
4. Fetch LFS-tracked files (if you cloned with Git LFS)
   ```bash
   git lfs install
   git lfs pull

   ```
5. Verify that you have
   ```bash
   outputs/RF_model_binary.pkl  #run train_random_forest will output this file, and you need to manually put the .pkl file under web-interface/outputs
   static/data/feature_table.pkl
   ```
### Running the Flask App
1. Start the server
```bash
  export FLASK_APP=app.py
  export FLASK_ENV=development
  flask run
```
2. Visit http://127.0.0.1:5000/ in your browser.


  
## Demo Video Link

https://drive.google.com/file/d/1bxdC3QI0MxXf20LmhCrOp52fu7POmMmi/view?usp=sharing

## Final Report
- [ECS171 Final Report (PDF)](./ECS171_Stock_prediction_Group18.pdf)




