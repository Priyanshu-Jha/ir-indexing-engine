from datasets import load_from_disk
# Make sure other required files (indexer.py, query.py, datastore.py, evaluate.py, core.py) exist
from indexer import PositionalIndexer, TF_Indexer, TFIDF_Indexer, Indexer
from query import QueryParser, QueryEvaluator
from datastore import PostgresDataStore
from evaluate import EvaluationFramework
import math
import re
import itertools # To generate combinations
import pandas as pd # To store results easily (pip install pandas)
import traceback # For detailed error printing
import os # For checking file paths
import shutil # For removing temporary directory
from tqdm import tqdm # For progress bars
import multiprocessing # <-- Import multiprocessing
import time # For timing

# --- Data Loading Functions ---
def generate_document_chunks(path, sample_size, chunk_size=500):
    """
    Loads the dataset from disk and yields chunks of (offset, document_texts).
    """
    print(f"Preparing to load data from {path} in chunks...")
    try:
        # Check if dataset path exists before loading
        if not os.path.exists(path):
             raise FileNotFoundError(f"Dataset directory not found at {path}")
        # Load dataset lazily
        wikiData = load_from_disk(path)
        effective_size = min(sample_size, len(wikiData))
        # Don't select slice yet, iterate through indices
        num_chunks = math.ceil(effective_size / chunk_size)
        print(f"Dataset (up to {effective_size} docs) will be processed in {num_chunks} chunks of size {chunk_size}.")
        for i in range(0, effective_size, chunk_size):
            # Select only the current chunk
            chunk_indices = range(i, min(i + chunk_size, effective_size))
            chunk_texts = wikiData.select(chunk_indices)['text']
            # Pass the starting doc_id offset for this chunk
            yield (i, chunk_texts) # Yield tuple (offset, texts)
    except FileNotFoundError as e_fnf:
        print(f"ERROR: {e_fnf}. Cannot generate chunks.")
        yield "GENERATOR_FAILED" # Use a specific signal for failure
    except Exception as e:
        print(f"Error loading or chunking dataset: {e}")
        traceback.print_exc() # Print full error
        yield "GENERATOR_FAILED" # Use a specific signal for failure


# --- Multiprocessing Helper Function ---
# Needs to be defined at the top level so it can be pickled for processes
def process_chunk_wrapper(chunk_id, doc_id_offset, document_chunk, indexer_class_name, config):
    """
    Wrapper function for multiprocessing. Instantiates indexer and processes one chunk.
    Saves the partial index and returns the path.
    """
    # Instantiate the correct indexer class dynamically
    if indexer_class_name == 'PositionalIndexer':
        chunk_processor = PositionalIndexer()
    elif indexer_class_name == 'TF_Indexer':
        chunk_processor = TF_Indexer()
    else: # Default or TFIDF uses TF logic for chunks
         chunk_processor = TF_Indexer()

    # Build the partial index for this chunk
    partial_index_data = chunk_processor.build_index_chunk(document_chunk, doc_id_offset)

    # Save the partial index block to the globally known temp directory
    use_compression = (config['compression'] == 'zlib')
    # Ensure temp dir exists (might be needed if workers start before main creates it)
    temp_dir = "temp_index_blocks"
    os.makedirs(temp_dir, exist_ok=True)
    temp_block_path = os.path.join(temp_dir, f"block_{chunk_id}.pkl")
    Indexer.save_partial(temp_block_path, partial_index_data, compress=use_compression)

    return temp_block_path # Return the path where the block was saved


# --- Main Evaluation Loop ---
def main():
    # --- Configuration Options ---
    # 1. DEFINE BUILD CONFIGS
    indexing_stages = ['positional', 'tf', 'tfidf']
    datastores = ['pickle', 'postgres']
    compressions = ['none', 'zlib']
    optimizations = ['none', 'skip_pointers']
    
    # 2. DEFINE QUERY CONFIGS
    query_processings = ['taat', 'daat'] # These will run on *each* built index

    # 3. CREATE BUILD CONFIGS (24 total)
    build_configs_list = list(itertools.product(
        indexing_stages, datastores, compressions, optimizations
    ))

    results_list = []

    # --- Setup ---
    dataset_path = r"C:\Users\priya\OneDrive\Desktop\SEM3\IRE\Assignment\assignment2\wikipediaDataset" # Use raw string for Windows paths
    total_docs_to_process = 50000 # <-- SET YOUR DESIRED DOCUMENT COUNT HERE
    chunk_size_for_indexing = 5000 # Adjust based on available RAM/cores

    total_runs = len(build_configs_list) * len(query_processings)
    print(f"--- Starting Automated Evaluation for {total_runs} Total Runs ({len(build_configs_list)} Builds) ---")
    print(f"Targeting {total_docs_to_process} documents, processed in chunks of {chunk_size_for_indexing}.")
    print(f"Dataset Path: {os.path.abspath(dataset_path)}")

    # Load the document list needed by QueryEvaluator context
    # WARNING: This can consume significant RAM for large 'total_docs_to_process'
    print("Skipping raw document text load to save RAM. Setting documents_list_for_eval to None.")
    documents_list_for_eval = None

    temp_index_dir = "temp_index_blocks" # Define temp directory name

    # Determine Number of Processes
    try:
        # Consider leaving more cores free if memory is tight
        num_processes = 5
    except NotImplementedError:
        print("Warning: cpu_count() not implemented. Defaulting to 1 process.")
        num_processes = 1
    print(f"Using {num_processes} worker processes for indexing chunks.")

    current_run_count = 0

    # --- Loop Through Each BUILD Configuration (24 loops) ---
    for build_i, build_cfg_tuple in enumerate(build_configs_list):
        
        # Create the BASE config for this BUILD
        build_config = {
            'indexing_stage': build_cfg_tuple[0],
            'datastore':      build_cfg_tuple[1],
            'compression':    build_cfg_tuple[2],
            'optimization':   build_cfg_tuple[3]
        }
        
        # Generate a filename based ONLY on build config
        pickle_fname = f"final_idx_{build_config['indexing_stage']}_{build_config['datastore']}_{build_config['compression']}_{build_config['optimization']}.pkl"

        print(f"\n===== Building Index {build_i+1}/{len(build_configs_list)} =====")
        print(f"Build Config: {build_config}")
        print("------------------------------------------")

        # --- This is your existing build logic ---
        # --- (It runs ONCE per 24 configs) ---
        loaded_index_data_pickle = None # Store loaded pickle data
        pg_store_eval = None # Store PG connection for evaluation
        use_compression = (build_config['compression'] == 'zlib')
        add_skips = (build_config['optimization'] == 'skip_pointers')
        final_index_location = None # Path/Identifier for the final index
        save_load_build_status = "Not Run" # Track status
        build_needed = True

        pg_store_build = None # Connection for building/saving PG index
        pool = None # Pool for multiprocessing

        try: # Wrap config build
            # --- Determine Final Index Location & Check/Load Existence ---
            if build_config['datastore'] == 'pickle':
                final_index_location = pickle_fname
                if os.path.exists(final_index_location):
                    try:
                        print(f"Found existing FINAL Pickle index: {final_index_location}. Loading...")
                        loaded_index_data_pickle = Indexer.load(final_index_location)
                        if loaded_index_data_pickle and loaded_index_data_pickle.get('total_docs') == total_docs_to_process:
                             print(f"Load successful (Matched {total_docs_to_process} docs).")
                             build_needed = False; save_load_build_status = "Loaded Existing"
                        elif loaded_index_data_pickle: print(f"Doc count mismatch (Loaded: {loaded_index_data_pickle.get('total_docs')}). Will rebuild."); build_needed = True; loaded_index_data_pickle = None
                        else: print("Loaded data invalid. Will rebuild."); build_needed = True; loaded_index_data_pickle = None
                    except Exception as e: print(f"Error loading final pickle: {e}. Will rebuild."); build_needed = True; loaded_index_data_pickle = None

            elif build_config['datastore'] == 'postgres':
                final_index_location = 'ir_project' # DB name for footprint
                print("PostgreSQL selected. Index will be built/overwritten (previous data truncated on save).")
                build_needed = True


            # --- Build and Merge Index Incrementally ONLY IF NEEDED ---
            if build_needed:
                print("Building index incrementally using multiprocessing...")
                start_build_time = time.perf_counter()

                if os.path.exists(temp_index_dir): shutil.rmtree(temp_index_dir)
                os.makedirs(temp_index_dir)
                temp_block_locations = []

                tasks_args = []
                chunk_id_counter = 0
                indexer_class_name = 'PositionalIndexer' if build_config['indexing_stage'] == 'positional' else 'TF_Indexer'

                print("Preparing chunk arguments for workers...")
                doc_chunk_generator = generate_document_chunks(dataset_path, sample_size=total_docs_to_process, chunk_size=chunk_size_for_indexing)
                for chunk_data in doc_chunk_generator:
                    if chunk_data == "GENERATOR_FAILED": raise RuntimeError("Document generator failed.")
                    doc_id_offset_arg, document_chunk = chunk_data
                    if not isinstance(document_chunk, list): document_chunk = list(document_chunk)
                    tasks_args.append((chunk_id_counter, doc_id_offset_arg, document_chunk, indexer_class_name, build_config))
                    chunk_id_counter += 1
                if not tasks_args: raise RuntimeError("No chunks generated.")

                pool = None # Define pool outside try for finally block
                try: # Wrap pool execution
                    print(f"Distributing {len(tasks_args)} chunks to {num_processes} worker processes...")
                    pool = multiprocessing.Pool(processes=num_processes)
                    results_iterator = pool.starmap(process_chunk_wrapper, tasks_args)
                    temp_block_locations = list(tqdm(results_iterator, total=len(tasks_args), desc="Processing Chunks"))
                    pool.close(); pool.join() # Wait for processes
                    print(f"\nProcessed documents across {len(temp_block_locations)} blocks.")
                except KeyboardInterrupt:
                    print("\n!!! KeyboardInterrupt! Terminating workers... !!!")
                    if pool: pool.terminate(); pool.join()
                    raise # Re-raise the interrupt to stop the outer loop
                except Exception as e_pool: # Catch other potential pool errors
                    print(f"\n!!! Error during multiprocessing: {e_pool} !!!")
                    traceback.print_exc()
                    if pool: pool.terminate(); pool.join()
                    raise # Re-raise the error

                print("Merging partial indices...")
                merge_start_time = time.perf_counter()
                if build_config['indexing_stage'] == 'positional': final_indexer = PositionalIndexer()
                elif build_config['indexing_stage'] == 'tf': final_indexer = TF_Indexer()
                else: final_indexer = TFIDF_Indexer()


                # --- *** START OF RAM OPTIMIZATION (FIX 2) *** ---
                if build_config['datastore'] == 'pickle':
                    # Use the original, RAM-heavy merge for Pickle
                    print("Using standard merge for Pickle...")
                    final_indexer.merge_indices(temp_block_locations, final_index_location, compress=use_compression, add_skips=add_skips)
                    # Load after merge for evaluation
                    loaded_index_data_pickle = Indexer.load(final_index_location)
                    save_load_build_status = "Built and Loaded" if loaded_index_data_pickle else "Build Failed"

                elif build_config['datastore'] == 'postgres':
                    # Use the NEW, memory-optimized merge for Postgres
                    print("Starting optimized merge-to-PostgreSQL...")
                    add_skips = False
                    pg_store_build = None
                    try:
                         pg_store_build = PostgresDataStore(password='root')
                         # Call the new function
                         final_indexer.merge_indices_to_postgres(
                             temp_block_locations, 
                             pg_store_build, 
                             compress=use_compression, 
                             add_skips=add_skips
                         )
                         print("Final index saved to PostgreSQL via memory-optimized merge.")
                         save_load_build_status = "Built and Saved to DB"
                    except Exception as e_pg_save: 
                         print(f"Error saving merged index: {e_pg_save}"); traceback.print_exc()
                         save_load_build_status = "Build Failed (Save Error)"
                    finally:
                         if pg_store_build: pg_store_build.close()
                # --- *** END OF RAM OPTIMIZATION (FIX 2) *** ---


                merge_end_time = time.perf_counter()
                print(f"Merging took {merge_end_time - merge_start_time:.2f} seconds.")
                if os.path.exists(temp_index_dir): shutil.rmtree(temp_index_dir)
                end_build_time = time.perf_counter()
                print(f"Total build and merge time: {end_build_time - start_build_time:.2f} seconds.")

            # --- *** END OF BUILD LOGIC *** ---

            # --- *** START OF INNER EVALUATION LOOP (FIX 1) *** ---
            for query_proc_mode in query_processings:
                current_run_count += 1
                print(f"\n--- Evaluating Run {current_run_count}/{total_runs} (Query Mode: {query_proc_mode}) ---")
                
                # Create the FINAL config for this specific run
                config = build_config.copy()
                config['query_processing'] = query_proc_mode

                # --- This is your existing evaluation logic ---
                query_eval_input = None
                eval_connection_status = "OK"
                pg_store_eval = None # Ensure it's reset for each loop

                if config['datastore'] == 'pickle':
                    # Use data loaded either from existing file or after build/merge
                    query_eval_input = loaded_index_data_pickle
                    if query_eval_input is None:
                        # Try to load it if it wasn't just built (e.g., build_needed=False)
                        if os.path.exists(final_index_location):
                            print(f"Loading {final_index_location} for eval...")
                            query_eval_input = Indexer.load(final_index_location)
                            if query_eval_input is None:
                                eval_connection_status = "Pickle Load/Build Result Invalid"
                        else:
                            eval_connection_status = "Pickle file not found and build failed"

                elif config['datastore'] == 'postgres':
                    try:
                        # Create/Ensure connection specifically for evaluation
                        print("Connecting to PostgreSQL for evaluation...")
                        pg_store_eval = PostgresDataStore(password='root')
                        query_eval_input = pg_store_eval # Pass the connected instance
                    except Exception as e_pg_conn:
                        print(f"Failed to connect to PostgreSQL for evaluation: {e_pg_conn}")
                        eval_connection_status = "DB Connection Failed for Eval"

                # Proceed only if we have valid input for the evaluator
                if query_eval_input and eval_connection_status == "OK":
                    current_results = config.copy()
                    current_results['build_status'] = save_load_build_status # Record how index was obtained
                    try:
                        # Pass the correct input (dict or datastore instance) and config
                        query_evaluator = QueryEvaluator(query_eval_input, documents_list_for_eval, config['indexing_stage'], config)
                        evaluator = EvaluationFramework(query_evaluator, config, pg_password='root')

                        # A & B: Latency & Throughput
                        print(f"Running latency/throughput tests (Mode: {query_proc_mode})...")
                        latency_results = evaluator.run_latency_throughput_test()
                        current_results.update({k: v for k, v in latency_results.items()})

                        # C: Disk Footprint
                        print("Measuring disk footprint...")
                        # final_index_location should be correctly set
                        footprint_results = evaluator.measure_disk_footprint(final_index_location)
                        current_results.update({k: v for k, v in footprint_results.items()})

                        results_list.append(current_results)
                        print("Evaluation for this query mode complete.")

                    except Exception as e_eval:
                         print(f"Error during evaluation: {e_eval}"); traceback.print_exc()
                         current_results['evaluation_error'] = str(e_eval); results_list.append(current_results)
                    finally:
                         # Close the evaluation connection if it was Postgres
                         if config['datastore'] == 'postgres' and pg_store_eval:
                              pg_store_eval.close()
                              print("Closed PostgreSQL connection for evaluation.")
                else:
                    # Log why evaluation was skipped
                    print(f"Evaluation skipped. Status: {save_load_build_status}, Connection Status: {eval_connection_status}")
                    error_res = config.copy(); error_res['run_error'] = f"Eval skipped: {save_load_build_status}, {eval_connection_status}"; results_list.append(error_res)
            
            # --- *** END OF INNER EVALUATION LOOP *** ---

        except KeyboardInterrupt: # Catch interrupt during the outer config loop
            print("\n!!! KeyboardInterrupt detected! Stopping evaluation loop. !!!")
            break # Exit the main configuration loop
        except Exception as e_main_loop:
             print(f"\n--- !!! UNEXPECTED ERROR during build config run !!! ---")
             print(f"Config: {build_config}"); print(f"Error: {e_main_loop}"); traceback.print_exc()
             error_results = build_config.copy(); error_results['run_error'] = f"Outer loop error: {e_main_loop}"; results_list.append(error_results)
        finally:
             # Cleanup temp dir if loop crashes mid-build
             if 'build_needed' in locals() and build_needed and os.path.exists(temp_index_dir):
                  print(f"Cleaning up temp dir: {temp_index_dir}"); shutil.rmtree(temp_index_dir)
             # Ensure any potentially remaining PG connections are closed (build or eval)
             # Check if variables exist and if connection is open
             if 'pg_store_build' in locals() and pg_store_build and pg_store_build.conn and pg_store_build.conn.closed == 0:
                  print("Closing build PG connection in finally..."); pg_store_build.close()
             # Note: pg_store_eval is now closed inside the inner loop, but this is a safety catch
             if 'pg_store_eval' in locals() and pg_store_eval and pg_store_eval.conn and pg_store_eval.conn.closed == 0:
                  print("Closing eval PG connection in outer finally..."); pg_store_eval.close()
             # Ensure pool is cleaned up if error happened before 'with' block finished or if interrupted
             if pool and hasattr(pool, '_state') and pool._state == multiprocessing.pool.RUN: # Check if pool is running
                 print("Terminating pool in outer finally block...")
                 pool.terminate()
                 pool.join()


    # --- Save or Print Final Results ---
    print("\n===== Automated Evaluation Finished =====")
    if results_list:
        try:
            results_df = pd.DataFrame(results_list)
            print("Results Summary:")
            pd.set_option('display.max_rows', None); pd.set_option('display.max_columns', None); pd.set_option('display.width', 2000); pd.set_option('display.max_colwidth', 50)
            print(results_df)
            results_df.to_csv("evaluation_results.csv", index=False)
            print("\nResults saved to evaluation_results.csv")
        except ImportError: print("Pandas not found. Install pandas to save results to CSV.")
        except Exception as e_save: print(f"Error saving results: {e_save}")
        if 'results_df' not in locals(): print("\nRaw results list:"); [print(res) for res in results_list]
    else:
        print("No results were collected.")


# --- Script Entry Point ---
if __name__ == "__main__":
    # Crucial for multiprocessing compatibility on Windows/macOS
    multiprocessing.freeze_support()
    main()