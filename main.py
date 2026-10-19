
# Hospital Readmission Analysis Using K-Means

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Excess Readmission Ratio:
# Below 1.00 = fewer readmissions than expected
# Around 1.00 = readmissions as expected
# Above 1.00 = more readmissions than expected

# 1. Load the dataset
data = pd.read_csv("FY_2026_Hospital_Readmissions_Reduction_Program_Hospital.csv")
data.head(5)
data.info()
data.columns = data.columns.str.strip()

# 2. Clean the readmission ratio
data["Excess Readmission Ratio"] = pd.to_numeric(
    data["Excess Readmission Ratio"], errors="coerce"
)

data = data.dropna(
    subset=["Facility ID", "Facility Name", "State"]
)
ratio_col = "Excess Readmission Ratio"

# Report missing ratios before excluding them
missing_count = data[ratio_col].isna().sum()
missing_percent = data[ratio_col].isna().mean() * 100

print("Missing readmission ratios:", missing_count)
print(f"Percentage missing: {missing_percent:.2f}%")

# Keep valid positive ratios for the analysis
data = data[
    data[ratio_col].notna() &
    (data[ratio_col] > 0)
].copy()

print("Records remaining after cleaning:", len(data))

data = data[data["Excess Readmission Ratio"] > 0].copy()

# 3. Calculate the average ratio for each hospital
hospital_data = data.groupby(
    ["Facility ID", "Facility Name", "State"],
    as_index=False
).agg(
    Average_Ratio=("Excess Readmission Ratio", "mean"),
    Number_of_Measures=("Measure Name", "nunique")
)

# 4. Prepare the data for clustering
X = hospital_data[["Average_Ratio"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Create three hospital clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
hospital_data["Cluster"] = kmeans.fit_predict(X_scaled)

# Name clusters from lowest to highest ratio
cluster_order = hospital_data.groupby("Cluster")[
    "Average_Ratio"
].mean().sort_values().index.tolist()

cluster_names = {
    cluster_order[0]: "Lower Ratio",
    cluster_order[1]: "Middle Ratio",
    cluster_order[2]: "Higher Ratio"
}

hospital_data["Cluster Name"] = hospital_data["Cluster"].map(
    cluster_names
)

# 6. Display the cluster summary
cluster_summary = hospital_data.groupby("Cluster Name").agg(
    Number_of_Hospitals=("Facility ID", "nunique"),
    Average_Ratio=("Average_Ratio", "mean"),
    Median_Ratio=("Average_Ratio", "median")
).reindex(
    ["Lower Ratio", "Middle Ratio", "Higher Ratio"]
).round(3)

print("\nHOSPITAL CLUSTER SUMMARY")
print(cluster_summary)

# Set the graph style
sns.set_style("whitegrid")

cluster_order_names = [
    "Lower Ratio", "Middle Ratio", "Higher Ratio"
]

cluster_colors = {
    "Lower Ratio": "green",
    "Middle Ratio": "orange",
    "Higher Ratio": "red"
}

# 7. Number of hospitals in each cluster
cluster_counts = hospital_data["Cluster Name"].value_counts().reindex(
    cluster_order_names, fill_value=0
)

plt.figure(figsize=(9, 5))

cluster_counts.plot(
    kind="bar",
    color="steelblue",
    edgecolor="black"
)

plt.title("Number of Hospitals in Each Cluster", fontsize=14)
plt.xlabel("Hospital Cluster")
plt.ylabel("Number of Hospitals")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("Hospitals_in_Each_Cluster.png", dpi=300)
plt.show()

# 8. Average readmission ratio by cluster
cluster_averages = hospital_data.groupby(
    "Cluster Name"
)["Average_Ratio"].mean().reindex(cluster_order_names)

plt.figure(figsize=(9, 5))

cluster_averages.plot(
    kind="bar",
    color="cornflowerblue",
    edgecolor="black"
)

plt.axhline(
    y=1.0,
    color="red",
    linestyle="--",
    label="Expected Ratio (1.0)"
)

plt.title("Average Readmission Ratio by Cluster", fontsize=14)
plt.xlabel("Hospital Cluster")
plt.ylabel("Average Excess Readmission Ratio")
plt.xticks(rotation=0)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("Average_Readmission_Ratio.png", dpi=300)
plt.show()

# 9. Scatter plot of hospital readmission ratios
plt.figure(figsize=(10, 5))

for cluster_name in cluster_order_names:
    cluster_data = hospital_data[
        hospital_data["Cluster Name"] == cluster_name
    ]

    plt.scatter(
        cluster_data.index,
        cluster_data["Average_Ratio"],
        color=cluster_colors[cluster_name],
        label=cluster_name,
        alpha=0.65,
        s=25
    )

plt.axhline(
    y=1.0,
    color="black",
    linestyle="--",
    label="Expected Ratio (1.0)"
)

plt.title("Hospital Readmission Ratios by Cluster", fontsize=14)
plt.xlabel("Hospital Record")
plt.ylabel("Average Excess Readmission Ratio")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("Hospital_Readmission_Ratios.png", dpi=300)
plt.show()

# 10. Display the ten hospitals with the highest ratios
highest_hospitals = hospital_data.sort_values(
    "Average_Ratio", ascending=False
).head(10)

plt.figure(figsize=(12, 6))

sns.barplot(
    data=highest_hospitals,
    x="Average_Ratio",
    y="Facility Name",
    hue="Cluster Name",
    dodge=False,
    palette=cluster_colors
)

plt.axvline(
    x=1.0,
    color="black",
    linestyle="--",
    label="Expected Ratio (1.0)"
)

plt.title("Top 10 Hospitals with Highest Average Readmission Ratios")
plt.xlabel("Average Excess Readmission Ratio")
plt.ylabel("Hospital Name")
plt.legend(title="Cluster")
plt.tight_layout()

plt.savefig("Top_10_Hospitals.png", dpi=300)
plt.show()

print("\n10 HOSPITALS WITH THE HIGHEST AVERAGE RATIOS")

print(
    highest_hospitals[
        ["Facility Name", "State", "Average_Ratio", "Cluster Name"]
    ].to_string(index=False)
)

# 11. Save the results
hospital_data.to_csv("Hospital_Clustering_Results.csv", index=False)

print("\nAnalysis complete!")
print("Results saved to Hospital_Clustering_Results.csv")
