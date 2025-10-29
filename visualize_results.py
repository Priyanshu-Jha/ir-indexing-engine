import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Makes plots look nicer
import numpy as np

# --- 1. Load the Data ---
try:
    df = pd.read_csv("evaluation_results3.csv")
    print("Successfully loaded evaluation_results.csv")
except FileNotFoundError:
    print("Error: evaluation_results.csv not found. Make sure it's in the same directory.")
    exit()
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# --- Data Cleaning (Optional but Recommended) ---
# Ensure numeric columns are actually numeric, coercing errors
numeric_cols = ['avg', 'p95', 'p99', 'throughput', 'raw_bytes', 'total_queries_run', 'elapsed_time_sec']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows where essential numeric data might be missing after coercion
df.dropna(subset=['avg', 'throughput', 'raw_bytes'], inplace=True) # Adjust columns as needed

print("Data loaded and cleaned. Generating plots...")


# Set a nice style for the plots
sns.set_theme(style="whitegrid")

# Convert bytes to MB for y-axis formatting later
def bytes_to_mb(y, pos):
    """Formats bytes to MB for matplotlib labels"""
    if pd.isna(y) or y == 0:
        return '0 MB'
    return f'{y / (1024*1024):.1f} MB'

# --- Chart 1: Average Latency by Query Processing & Indexing Stage ---
plt.figure(figsize=(14, 7))
ax1 = sns.barplot(
    data=df,
    x='indexing_stage',
    y='avg', # <--- FIX 1
    hue='query_processing',
    palette='viridis',
    order=['positional', 'tf', 'tfidf']
)
plt.title('Average Query Latency (ms) by Index Type and Query Strategy', fontsize=16) # Removed doc count
plt.xlabel('Indexing Stage', fontsize=12)
plt.ylabel('Average Latency (ms)', fontsize=12)
# Consider log scale only if necessary based on data range
# plt.yscale('log')
plt.legend(title='Query Processing')
plt.tight_layout()
plt.savefig("chart1_latency_query_processing.png")
print("Generated chart1_latency_query_processing.png")
plt.close()


# --- Chart 2: Disk Footprint by Datastore and Indexing Stage ---
# Use raw_bytes for consistency before formatting
df_footprint = df[['datastore', 'compression', 'indexing_stage', 'raw_bytes']].drop_duplicates()

plt.figure(figsize=(12, 6))
ax2 = sns.barplot(
    data=df_footprint,
    x='indexing_stage',
    y='raw_bytes', # <--- Use raw_bytes here
    hue='datastore',
    palette='magma',
    order=['positional', 'tf', 'tfidf']
)

# Apply the MB formatter to the y-axis
ax2.yaxis.set_major_formatter(plt.FuncFormatter(bytes_to_mb))

plt.title('Disk Footprint by Index Type and Datastore', fontsize=16) # Removed doc count
plt.xlabel('Indexing Stage', fontsize=12)
plt.ylabel('Disk Footprint (MB)', fontsize=12)
plt.legend(title='Datastore')
plt.tight_layout()
plt.savefig("chart2_disk_footprint.png")
print("Generated chart2_disk_footprint.png")
plt.close()


# --- Chart 3: Impact of Skip Pointers on TAAT Latency ---
# Filter for TAAT results only
# Check if 'query_processing' column exists before filtering
if 'query_processing' in df.columns:
    df_taat = df[df['query_processing'] == 'taat'].copy()

    plt.figure(figsize=(14, 7))
    ax3 = sns.barplot(
        data=df_taat,
        x='indexing_stage',
        y='avg', # <--- FIX 2
        hue='optimization',
        palette='coolwarm',
        order=['positional', 'tf', 'tfidf']
    )

    plt.title('Impact of Skip Pointers on Average TAAT Latency (ms)', fontsize=16) # Removed doc count
    plt.xlabel('Indexing Stage', fontsize=12)
    plt.ylabel('Average TAAT Latency (ms)', fontsize=12)
    plt.legend(title='Optimization')
    plt.tight_layout()
    plt.savefig("chart3_latency_skip_pointers_taat.png")
    print("Generated chart3_latency_skip_pointers_taat.png")
    plt.close()
else:
    print("Skipping Chart 3: 'query_processing' column not found.")


# --- Chart 4: Latency (A) vs. Throughput (B) for Compression (z=1,2) ---
plt.figure(figsize=(14, 7))
# Ensure 'throughput' column exists and is numeric before plotting
if 'throughput' in df.columns:
    ax4 = sns.scatterplot(
        data=df,
        x='avg', # <--- FIX 3
        y='throughput', # Ensure this matches your CSV column name (it should be 'throughput')
        style='compression',
        hue='datastore',
        size='raw_bytes', # Use raw_bytes for size consistency
        sizes=(100, 1000),
        alpha=0.7
    )
    plt.title('Plot.AB: Latency vs. Throughput (Grouped by Compression & Datastore)', fontsize=16)
    plt.xlabel('Average Latency (ms) - (Lower is Better)', fontsize=12)
    plt.ylabel('Throughput (Queries/Sec) - (Higher is Better)', fontsize=12)
    # Adjust legend position slightly if needed
    ax4.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True, which="both", ls="--", c='0.7')
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    plt.savefig("chart4_latency_vs_throughput_compression.png")
    print("Generated chart4_latency_vs_throughput_compression.png")
    plt.close()
else:
    print("Skipping Chart 4: 'throughput' column not found or invalid.")


# --- Chart 5: Latency (A) vs. Footprint (C) for TAAT vs DAAT (q=Tn/Dn) ---
plt.figure(figsize=(14, 7))
if 'query_processing' in df.columns:
    ax5 = sns.scatterplot(
        data=df,
        x='avg', # <--- FIX 4
        y='raw_bytes', # Use raw_bytes for consistency
        style='query_processing',
        hue='indexing_stage',
        size='datastore',
        sizes={'pickle': 100, 'postgres': 500},
        alpha=0.7
    )

    # Apply the MB formatter to the y-axis
    ax5.yaxis.set_major_formatter(plt.FuncFormatter(bytes_to_mb))

    plt.title('Plot.AC: Latency vs. Footprint (Grouped by Query Processing)', fontsize=16)
    plt.xlabel('Average Latency (ms) - (Lower is Better)', fontsize=12)
    plt.ylabel('Disk Footprint (MB) - (Lower is Better)', fontsize=12)
    # Adjust legend position slightly if needed
    ax5.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True, which="both", ls="--", c='0.7')
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    plt.savefig("chart5_latency_vs_footprint_query_processing.png")
    print("Generated chart5_latency_vs_footprint_query_processing.png")
    plt.close()
else:
    print("Skipping Chart 5: 'query_processing' column not found.")

print("\nVisualization script finished.")