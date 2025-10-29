# evaluate.py
import time
import numpy as np
import os
import re
from tqdm import tqdm
from elasticsearch import Elasticsearch # Ensure this is installed
import psycopg2 # For measuring Postgres size
import traceback # Added for better error details

# Assuming QueryParser is in query.py
from query import QueryParser # Make sure query.py exists and is correct

class EvaluationFramework:
    def __init__(self, query_evaluator, config, pg_password='root'):
        """
        Initializes the evaluation framework. No longer connects to Elasticsearch.
        Args:
            query_evaluator: An instantiated QueryEvaluator object.
            config: The configuration dictionary from main.py.
            pg_password: Password for PostgreSQL connection.
        """
        self.query_evaluator = query_evaluator
        self.config = config
        self.query_parser = QueryParser() # Needed for TAAT
        self.pg_password = pg_password # Store password for DB size check

        # --- REMOVED Elasticsearch Connection Block ---

        # --- Unified & Strengthened Query Set ---
        # (This remains the same)
        self.unified_query_set = [
            # Single Terms
            {"id": "q01", "taat": '"history"', "daat": "history"},
            {"id": "q02", "taat": '"science"', "daat": "science"},
            {"id": "q03", "taat": '"anarchism"', "daat": "anarchism"},
            {"id": "q04", "taat": '"philosophy"', "daat": "philosophy"},
            {"id": "q05", "taat": '"thermonuclear astrophysics"', "daat": "thermonuclear astrophysics"}, # Treat as phrase for TAAT
            {"id": "q06", "taat": '"zygote"', "daat": "zygote"},
            # Simple Boolean AND (to test skips)
            {"id": "q07", "taat": '"art" AND "history"', "daat": "art history"},
            {"id": "q08", "taat": '"war" AND "united" AND "state"', "daat": "war united state"},
            {"id": "q09", "taat": '"world" AND "war"', "daat": "world war"},
            # Simple Boolean OR
            {"id": "q10", "taat": '"computer" OR "science"', "daat": "computer science"},
            {"id": "q11", "taat": '"language" OR "linguistics"', "daat": "language linguistics"},
            # Complex Boolean
            {"id": "q13", "taat": '("art" AND "history") OR "science"', "daat": "art history science"},
            {"id": "q14", "taat": '"language" AND ("syntax" OR "grammar")', "daat": "language syntax grammar"},
            {"id": "q15", "taat": '("world" AND "war") AND NOT "nuclear"', "daat": "world war nuclear"},
            # Phrase Queries
            {"id": "q16", "taat": '"albedo effect"', "daat": "albedo effect"},
            {"id": "q17", "taat": '"quantum entanglement"', "daat": "quantum entanglement"},
            {"id": "q18", "taat": '"theory of relativity"', "daat": "theory of relativity"},
            # More Ranking Style Queries
            {"id": "q19", "taat": '"political" AND "philosophy"', "daat": "political philosophy"},
            {"id": "q20", "taat": '"ancient" AND "greek" AND "mythology"', "daat": "ancient greek mythology"}
        ]

    def _run_single_query(self, query_dict):
        """Runs a single query using the configured evaluator and measures time."""
        # (This method remains the same)
        start_time = time.perf_counter()
        results = []
        query_str_to_run = ""
        try:
            if self.config['query_processing'] == 'taat':
                query_str_to_run = query_dict.get('taat')
                if not query_str_to_run: raise ValueError("TAAT query format missing")
                rpn_query = self.query_parser.to_rpn(query_str_to_run)
                if hasattr(self.query_evaluator, 'evaluate_rpn_taat'):
                     results = self.query_evaluator.evaluate_rpn_taat(rpn_query)
                else: raise NotImplementedError("TAAT evaluator method 'evaluate_rpn_taat' not found")
            elif self.config['query_processing'] == 'daat':
                 query_str_to_run = query_dict.get('daat')
                 if not query_str_to_run: raise ValueError("DAAT query format missing")
                 if hasattr(self.query_evaluator, 'evaluate_daat_ranked'):
                     results = self.query_evaluator.evaluate_daat_ranked(query_str_to_run, k=10)
                 else: raise NotImplementedError("DAAT evaluator method 'evaluate_daat_ranked' not found")
            else:
                raise ValueError(f"Unknown query_processing config: {self.config['query_processing']}")
        except Exception as e:
            print(f"\nError processing query '{query_str_to_run}': {e}")
            traceback.print_exc()
            results = []
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        return latency_ms, results


    # In evaluate.py

    def run_latency_throughput_test(self):
        """Measures latency (Avg, p95, p99) and simple throughput."""
        latencies = []
        query_set = self.unified_query_set # Use the unified set
        print(f"\nRunning {len(query_set)} queries for latency/throughput test ({self.config['query_processing']} mode)...")
        
        total_queries_run = 0 # Track successful queries
        total_start_time = time.perf_counter()
        
        for query_dict in tqdm(query_set, desc="Executing Queries"):
            latency, results = self._run_single_query(query_dict)
            if results is not None: # Count as run only if not an error
                latencies.append(latency)
                total_queries_run += 1
        total_end_time = time.perf_counter()

        if not latencies:
            print("No queries were successfully executed for latency test.")
            return {
                "avg": None, "p95": None, "p99": None, "throughput": 0,
                "total_queries_run": 0, "elapsed_time_sec": 0
            }

        avg = np.mean(latencies)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        total_time_sec = total_end_time - total_start_time
        # Calculate throughput based on all queries attempted
        throughput = len(query_set) / total_time_sec if total_time_sec > 0 else 0 

        print("Latency and Throughput Test Complete.")
        
        # --- THIS IS THE FIX ---
        # Add total_queries_run and total_time_sec to the returned dictionary
        return {
            "avg": avg, "p95": p95, "p99": p99, "throughput": throughput,
            "total_queries_run": total_queries_run, "elapsed_time_sec": total_time_sec
        }

    def measure_disk_footprint(self, index_path_or_dbname='ir_project'):
        """Measures the disk size of the index (Pickle file or Postgres DB tables)."""
        # (This method remains the same)
        size_bytes = 0; unit = "bytes"
        if self.config['datastore'] == 'pickle':
            try:
                if os.path.exists(index_path_or_dbname): size_bytes = os.path.getsize(index_path_or_dbname)
                else: print(f"Warning: Pickle file not found at {index_path_or_dbname}")
            except Exception as e: print(f"Error getting file size: {e}")
        elif self.config['datastore'] == 'postgres':
            conn = None
            try:
                conn = psycopg2.connect(dbname=index_path_or_dbname, user='postgres', password=self.pg_password, host='localhost')
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT sum(pg_relation_size(quote_ident(schemaname) || '.' || quote_ident(tablename)))::bigint
                    FROM pg_tables WHERE schemaname = 'public' AND tablename IN ('terms', 'postings');
                """)
                result = cursor.fetchone()
                if result and result[0] is not None: size_bytes = result[0]
                else: print("Warning: Could not retrieve table size from PostgreSQL.")
                cursor.close()
            except Exception as e: print(f"Error connecting to PostgreSQL or getting size: {e}")
            finally:
                 if conn: conn.close()
        else: print(f"Warning: Unknown datastore type: {self.config['datastore']}")
        # Convert to human-readable
        if size_bytes >= 1024**3: size = size_bytes / (1024**3); unit = "GB"
        elif size_bytes >= 1024**2: size = size_bytes / (1024**2); unit = "MB"
        elif size_bytes >= 1024: size = size_bytes / 1024; unit = "KB"
        else: size = size_bytes; unit = "Bytes"
        return {"size": size, "unit": unit, "raw_bytes": size_bytes}