# SelfIndex-v1.0: A Scalable Information Retrieval System

This project is a comprehensive Information Retrieval (IR) system built from scratch in Python as part of an academic assignment. It implements the entire indexing and querying pipeline, supporting a wide matrix of configurations to evaluate performance trade-offs.

The system is designed to be benchmarked against 48 unique experimental configurations, fulfilling all assignment requirements by analyzing the trade-offs between:
* Indexing strategies (Positional, TF, TF-IDF)
* Persistence backends (Pickle vs. PostgreSQL)
* Compression techniques (None vs. Zlib)
* Query processing algorithms (TAAT vs. DAAT)
* Index optimizations (Skip Pointers)

##  Assignment Requirements & Features

This project fulfills all assignment specifications by providing the following features:

### 1. Core Indexing Strategies (`x=n`)

The system implements three distinct indexing strategies:
* **`x=1` (Positional/Boolean):** `PositionalIndexer` stores a dictionary mapping `\{doc_id: [pos1, pos2, ...]\}`, enabling boolean and phrase queries.
* **`x=2` (Term Frequency):** `TF_Indexer` stores `\{doc_id: \{'tf': N, 'pos': [...] \}\}`, enabling simple rank-aware queries.
* **`x=3` (TF-IDF):** `TFIDF_Indexer` uses the same structure as the TF index. The `doc_freq` (document frequency) is computed during the merge, allowing the `QueryEvaluator` to calculate TF-IDF scores dynamically at query time.

### 2. Persistence & Datastores (`y=n`)

The index is persisted to disk and loaded on start. Two datastore backends are supported:
* **`y=1` (Pickle):** `indexer.py` handles saving the final merged index to disk as a single file using Python's `pickle` module.
* **`y=2` (PostgreSQL):** `datastore.py` provides a `PostgresDataStore` class that manages all database interactions, using highly optimized `psycopg2.extras.execute_values` for bulk inserts.

### 3. Compression (`z=n`)

Posting lists can be compressed to save space:
* **`z=1` (None):** Data is pickled and written as-is.
* **`z=2` (Zlib):** The `zlib` library is used to compress the pickled posting lists before they are written to disk or the database `BYTEA` field.

### 4. Index Optimization (`i=n`)

* **`i=1` (Skip Pointers):** The `_add_skip_pointers` function can be enabled during the merge. It adds `_skips` entries to posting lists with a skip distance of $O(\sqrt{n})$, accelerating intersections (the `AND` operation).

### 5. Query Processing (`q=Tn/Dn`)

The query engine supports two fundamental processing algorithms:
* **`q=Tn` (Term-at-a-Time):** `evaluate_rpn_taat` evaluates the RPN query using a stack, applying set operations (`_intersect_with_skips`, `_union`, `_not`) for `AND`, `OR`, and `NOT`.
* **`q=Dn` (Document-at-a-Time):** `evaluate_daat_ranked` handles ranked retrieval. It "walks" through documents by iterating over sorted `doc_id` lists and uses a min-heap (`heapq`) to efficiently maintain the top-k highest-scoring documents.

### 6. Full Query Parser

The system supports a complete boolean and phrase query grammar:
* **Operators:** `AND`, `OR`, `NOT`, and `PHRASE` (e.g., `"quantum entanglement"`).
* **Grouping:** `(parentheses)` are fully supported.
* **Operator Precedence:** The `QueryParser` uses the Shunting-yard algorithm to correctly handle operator precedence: `PHRASE` > `NOT` > `AND` > `OR`.

### 7. Evaluation Artefacts (A, B, C)

The `evaluate.py` script measures and reports all required artefacts for each run:
* **A (Latency):** Average, 95th percentile (p95), and 99th percentile (p99) response times in milliseconds.
* **B (Throughput):** System performance in queries per second (QPS).
* **C (Footprint):** Disk space consumed by the index, measured using `os.path.getsize` (Pickle) or `pg_relation_size` (Postgres).

### 8. Scalable & Robust Architecture

* **Multi-processed Indexing:** `main.py` uses a `multiprocessing` pool to process document chunks in parallel, building partial indices which are saved to disk.
* **Memory-Optimized Merge:** A key feature is the merge process. Instead of loading all partial indices into memory, the system iterates through each term, loads only its relevant postings from disk, merges them, and writes the final list to the destination (Pickle or Postgres). This allows indexing datasets larger than available RAM.

---

##  Project Structure

. ├── main.py #  Main entry point: Runs the full 48-run evaluation.
  ├── indexer.py #  Indexing logic: Positional, TF, TF-IDF, merging, skip pointers. 
  ├── query.py #  Query logic: QueryParser (Shunting-yard) & QueryEvaluator (TAAT/DAAT). 
  ├── datastore.py #  Persistence: PostgresDataStore class for all DB interactions. 
  ├── core.py #  Text processing: TextProcessor (tokenize, stem, stopword removal). 
  ├── evaluate.py #  Evaluation: EvaluationFramework, query set, A/B/C metric measurement. 
  ├── visualize_results.py #  Visualization: Reads the output CSV and generates the 5 plots. 
  └── evaluation_results.csv # (Output) The raw data from all 48 runs.


  ##  Getting Started

### Prerequisites

* Python 3.9+
* A running PostgreSQL server.
* You must **manually create the database** (e.g., `CREATE DATABASE ir_project;`).
* The code assumes a user/password of `postgres` / `root`. This can be changed in `datastore.py`.

### 1. Installation

1.  Clone this repository.
2.  Install the required Python packages. A `requirements.txt` would look like this:
    ```txt
    # requirements.txt
    datasets
    pandas
    nltk
    psycopg2-binary
    tqdm
    matplotlib
    seaborn
    ```
    Install them:
    ```sh
    pip install -r requirements.txt
    ```

3.  Download NLTK data (stopwords, tokenizer):
    ```sh
    python -m nltk.downloader stopwords punkt
    ```

### 2. Data Setup

The system is designed to use the `wikimedia/wikipedia` dataset (split `20231101.en`) from Hugging Face.

1.  **Download and save the dataset:** You only need to do this once. You can use a simple helper script:
    ```python
    # save_dataset.py
    from datasets import load_dataset
    print("Loading dataset from Hugging Face...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.en")
    print("Saving to disk...")
    ds.save_to_disk("./wikipediaDataset")
    print("Done.")
    ```
    Run it: `python save_dataset.py`

2.  **Update `main.py`:** Open `main.py` and update the `dataset_path` variable to point to the directory where you just saved the dataset (e.g., `./wikipediaDataset`).

    ```python
    # In main.py
    dataset_path = r"./wikipediaDataset" # <-- UPDATE THIS PATH
    ```

## 🏁 How to Run

### 1. Run the Full Evaluation

The main script will automatically build all 24 index configurations and run both TAAT and DAAT query modes on each, for a total of 48 evaluation runs.

```sh
python main.py
