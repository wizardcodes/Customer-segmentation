

# -----------------------------------------------
# Customer Segmentation using K-Means and RFM
# Based on the provided transaction data
# -----------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import datetime as dt
import warnings

warnings.filterwarnings('ignore', category=FutureWarning) 
warnings.filterwarnings('ignore', category=UserWarning) 

print("--- Customer Segmentation Program (RFM Analysis) ---")

# --- 1. Load Data ---
file_path = "D:\DEVELOPER\Customer-Segmentation-AIML-6th-Semester\Retail_dataset.csv"
print(f"\n[1] Loading Data from {file_path}...")

try:
    # Read the CSV, handling potential encoding issues (like BOM) and parsing dates
    # Specify the date format explicitly if pandas struggles to infer it
    data = pd.read_csv(file_path, encoding='utf-8-sig')
    # Try parsing the date column - adjust format if necessary
    try:
        data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], format='%d-%m-%Y %H:%M')
    except ValueError:
        print("Warning: Could not parse 'InvoiceDate' with format '%d-%m-%Y %H:%M'. Attempting automatic parsing.")
        data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

    print("Data loaded successfully.")
    print("\nOriginal Data Info:")
    data.info()
    print("\nSample Data Head:")
    print(data.head())

except FileNotFoundError:
    print(f"Error: File not found at {file_path}. Please check the path and filename.")
    exit() # Stop execution if file not found
except Exception as e:
    print(f"An error occurred during data loading: {e}")
    exit()

# --- 2. Data Cleaning & Preparation ---
print("\n[2] Cleaning and Preparing Data...")

# Check for missing values
print("\nMissing values before cleaning:")
print(data.isnull().sum())

# Handle missing CustomerID - Segmentation requires customer identification
initial_rows = data.shape[0]
data.dropna(subset=['CustomerID'], inplace=True)
print(f"\nRemoved {initial_rows - data.shape[0]} rows with missing CustomerID.")

# Convert CustomerID to integer
data['CustomerID'] = data['CustomerID'].astype(int)

# Remove cancelled orders (InvoiceNo starts with 'C') or negative quantities
# Also remove transactions with zero or negative unit price as they don't contribute to monetary value meaningfully
original_rows = data.shape[0]
data = data[~data['InvoiceNo'].astype(str).str.startswith('C')]
data = data[data['Quantity'] > 0]
data = data[data['UnitPrice'] > 0]
print(f"Removed {original_rows - data.shape[0]} rows corresponding to cancellations, negative/zero quantity or price.")

# Calculate Total Price for each item
data['TotalPrice'] = data['Quantity'] * data['UnitPrice']

print("\nData after initial cleaning:")
print(data.head())
data.info() # Check data types again

# --- 3. Calculate RFM Features ---
print("\n[3] Calculating RFM Features...")

# Find the most recent date in the dataset to use as a reference point (snapshot date)
# Add one day to make sure even the latest transaction has a recency > 0
snapshot_date = data['InvoiceDate'].max() + dt.timedelta(days=1)
print(f"Snapshot date for Recency calculation: {snapshot_date.strftime('%Y-%m-%d')}")

# Aggregate data by CustomerID
rfm_df = data.groupby('CustomerID').agg({
    'InvoiceDate': lambda date: (snapshot_date - date.max()).days, # Recency: Days since last purchase
    'InvoiceNo': 'nunique',                                     # Frequency: Count of unique invoices
    'TotalPrice': 'sum'                                         # Monetary: Sum of total purchases
})

# Rename columns for clarity
rfm_df.rename(columns={'InvoiceDate': 'Recency',
                       'InvoiceNo': 'Frequency',
                       'TotalPrice': 'MonetaryValue'}, inplace=True)

print("\nRFM Dataframe Head:")
print(rfm_df.head())

print("\nDescriptive Statistics for RFM:")
print(rfm_df.describe())

# --- Handle Skewness ---
# RFM features often have skewed distributions. Log transformation can help.
# Add 1 to avoid log(0) issues.
print("\nApplying log transformation to handle skewness in F and M (and potentially R)...")
rfm_log = rfm_df.copy()
# Check for non-positive values before log transform (should be okay after cleaning, but good practice)
if (rfm_log['Recency'] <= 0).any() or (rfm_log['Frequency'] <= 0).any() or (rfm_log['MonetaryValue'] <= 0).any():
    print("Warning: Found non-positive values before log transform. This might indicate data issues.")
    # Option: Add a small constant or investigate further. For now, we proceed assuming positive values after cleaning.
    rfm_log['Recency'] = rfm_log['Recency'].apply(lambda x: max(x, 1)) # Ensure Recency >= 1
    rfm_log['Frequency'] = rfm_log['Frequency'].apply(lambda x: max(x, 1)) # Ensure Frequency >= 1
    rfm_log['MonetaryValue'] = rfm_log['MonetaryValue'].apply(lambda x: max(x, 0.01)) 

rfm_log['Recency'] = np.log1p(rfm_log['Recency']) # log1p calculates log(1 + x)
rfm_log['Frequency'] = np.log1p(rfm_log['Frequency'])
rfm_log['MonetaryValue'] = np.log1p(rfm_log['MonetaryValue'])

print("\nRFM Dataframe Head (Log Transformed):")
print(rfm_log.head())

# Visualize distributions after log transform (optional but recommended)
plt.figure(figsize=(15, 5))
for i, feature in enumerate(['Recency', 'Frequency', 'MonetaryValue']):
    plt.subplot(1, 3, i + 1)
    sns.histplot(rfm_log[feature], kde=True) # type: ignore
    plt.title(f'Distribution of Log({feature})')
plt.tight_layout()
plt.suptitle('RFM Feature Distributions (Log Transformed)', y=1.02)
plt.show()


# --- 4. Preprocessing RFM Data ---
print("\n[4] Preprocessing RFM Data (Scaling)...")
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log) # Scale the log-transformed data

print("RFM data scaled using StandardScaler.")
# print("\nScaled RFM Data Sample:\n", rfm_scaled[:5]) # Optional

# --- 5. Determine Optimal Number of Clusters (K) ---
print("\n[5] Determining Optimal Number of Clusters (K)...")

wcss = [] # Within-Cluster Sum of Squares (Inertia)
silhouette_scores = []
k_range = range(2, 11) # Test K from 2 to 10

for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(rfm_scaled)
    wcss.append(kmeans.inertia_)
    score = silhouette_score(rfm_scaled, kmeans.labels_)
    silhouette_scores.append(score)
    print(f" K={k}, WCSS={kmeans.inertia_:.2f}, Silhouette Score={score:.3f}")

# Plotting the results
plt.figure(figsize=(12, 5))

# Elbow Method Plot
plt.subplot(1, 2, 1)
plt.plot(k_range, wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Inertia)')
plt.grid(True)

# Silhouette Score Plot
plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, marker='o', linestyle='--')
plt.title('Silhouette Scores for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.grid(True)

plt.tight_layout()
plt.show()

# --- Choosing K ---
# Analyze the plots. Look for the "elbow" in WCSS and the peak in Silhouette Score.
# For RFM data, 3-5 clusters are often meaningful. Let's start with 4, but adjust based on your plots.
optimal_k = 4 # <<< ADJUST THIS BASED ON YOUR PLOTS <<<
print(f"\nBased on Elbow Method and Silhouette Score analysis, choosing K = {optimal_k}")


# --- 6. Apply K-Means Clustering ---
print(f"\n[6] Applying K-Means with K = {optimal_k}...")
kmeans_final = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=42)
kmeans_final.fit(rfm_scaled)

# Add the cluster labels back to the ORIGINAL RFM dataframe (before log/scaling) for interpretation
rfm_df['Cluster'] = kmeans_final.labels_
print("Cluster labels assigned to RFM data.")
print("\nRFM Data Head with Cluster Labels:")
print(rfm_df.head())

# --- 7. Analyze and Visualize Clusters ---
print("\n[7] Analyzing and Visualizing Clusters...")

# Calculate average RFM values for each cluster (using the original, non-logged values)
cluster_summary = rfm_df.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'MonetaryValue': 'mean',
    'Cluster': 'size' # Get cluster size
}).rename(columns={'Cluster': 'Cluster Size'})

print("\nCluster Summary (Mean RFM Values & Size):")
print(cluster_summary.round(2)) # Round for readability

# --- Visualizations (using original RFM values for interpretability) ---

# Scatter plot: Recency vs MonetaryValue
plt.figure(figsize=(10, 7))
sns.scatterplot(data=rfm_df, x='Recency', y='MonetaryValue', hue='Cluster', palette='viridis', s=50, alpha=0.7)
plt.title(f'Customer Segments by Recency vs Monetary Value (K={optimal_k})')
plt.xlabel('Recency (Days since last purchase)')
plt.ylabel('Monetary Value (Total Spent)')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# Scatter plot: Frequency vs MonetaryValue
plt.figure(figsize=(10, 7))
sns.scatterplot(data=rfm_df, x='Frequency', y='MonetaryValue', hue='Cluster', palette='viridis', s=50, alpha=0.7)
plt.title(f'Customer Segments by Frequency vs Monetary Value (K={optimal_k})')
plt.xlabel('Frequency (Number of Purchases)')
plt.ylabel('Monetary Value (Total Spent)')
plt.legend(title='Cluster')
# Use log scale for axes if distribution is very skewed, even after clustering
# plt.xscale('log')
# plt.yscale('log')
plt.grid(True)
plt.show()

# Scatter plot: Recency vs Frequency
plt.figure(figsize=(10, 7))
sns.scatterplot(data=rfm_df, x='Recency', y='Frequency', hue='Cluster', palette='viridis', s=50, alpha=0.7)
plt.title(f'Customer Segments by Recency vs Frequency (K={optimal_k})')
plt.xlabel('Recency (Days since last purchase)')
plt.ylabel('Frequency (Number of Purchases)')
plt.legend(title='Cluster')
# plt.yscale('log') # Frequency might benefit from log scale for visualization
plt.grid(True)
plt.show()


# Box plots to see RFM distributions per cluster
for feature in ['Recency', 'Frequency', 'MonetaryValue']:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=rfm_df, x='Cluster', y=feature, palette='viridis', showfliers=False) # Hide outliers for clarity
    plt.title(f'{feature} Distribution per Cluster (K={optimal_k})')
    # if feature in ['Frequency', 'MonetaryValue']: # Use log scale if needed for visualization
    #     plt.yscale('log')
    plt.show()


# --- 8. Interpretation ---
print("\n--- [8] Interpretation Guide ---")
print("Analyze the 'Cluster Summary (Mean RFM Values & Size)' and the plots.")
print("Assign descriptive names to each cluster based on their RFM characteristics:")
print(" - Low Recency = Recent customer")
print(" - High Frequency = Buys often")
print(" - High MonetaryValue = Spends a lot")
print("\nExample Interpretations (modify based on YOUR results):")
print(" * Cluster with Low R, High F, High M: Best Customers / Champions")
print(" * Cluster with High R, Low F, Low M: Lost Customers / At Risk (Churned)")
print(" * Cluster with Low R, High F, Mid/Low M: Loyal Customers (Frequent but lower spend)")
print(" * Cluster with Low R, Low F, Low M: New Customers")
print(" * Cluster with High R, High F, High M: Can't Lose Them (High value, but becoming inactive)")
print(" * Cluster with Mid R, Mid F, Mid M: Potential Loyalists / Needs Attention")
print("\nLook at your specific 'Cluster Summary' table to name your segments accurately.")

# --- Saving Results (Optional) ---
output_filename = 'customer_segments_rfm.csv'
# Save the RFM dataframe with cluster labels
try:
    rfm_df.reset_index().to_csv(output_filename, index=False) # Reset index to include CustomerID as a column
    print(f"\nSegmented RFM data saved to {output_filename}")
except Exception as e:
    print(f"\nError saving results to CSV: {e}")

print("\n--- Program End ---")