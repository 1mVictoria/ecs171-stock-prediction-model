import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Folder where cleaned files live
clean_folder = "cleaned"
os.makedirs(clean_folder, exist_ok=True)

# Data sources: (what it is, filename)
data_files = [
    ("ESG Risk", "esgRisk_cleaned.csv"),
    ("Fundamentals", "fundamentals_cleaned.csv"),
    ("Prices", "prices_cleaned.csv"),
]

# Loop through each dataset and plot the first numeric column
for label, fname in data_files:
    file_path = os.path.join(clean_folder, fname)

    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Couldn't find the file: {file_path}")
        continue

    # Find numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        print(f"No numeric columns found in {fname}")
        continue

    col_to_plot = numeric_cols[0]  # Just pick the first one for a quick look

    # Plot histogram
    plt.figure()
    plt.hist(data[col_to_plot].dropna(), bins=50)
    plt.title(f"{label} – Histogram of {col_to_plot}")
    plt.xlabel(col_to_plot)
    plt.ylabel("Frequency")
    plt.tight_layout()
    hist_file = os.path.join(clean_folder, f"{label.lower().replace(' ', '_')}_{col_to_plot}_hist.png")
    plt.savefig(hist_file)
    plt.close()
    print(f"Saved histogram: {hist_file}")

    # Plot boxplot
    plt.figure()
    plt.boxplot(data[col_to_plot].dropna(), vert=False)
    plt.title(f"{label} – Boxplot of {col_to_plot}")
    plt.xlabel(col_to_plot)
    plt.tight_layout()
    box_file = os.path.join(clean_folder, f"{label.lower().replace(' ', '_')}_{col_to_plot}_box.png")
    plt.savefig(box_file)
    plt.close()
    print(f" Saved boxplot: {box_file}")

print("All plots generated and saved in the 'cleaned/' folder.")
