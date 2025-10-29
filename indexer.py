import os
import pickle
import zlib
from collections import defaultdict
from core import TextProcessor # Ensure core.py exists
import math
from tqdm import tqdm
import psycopg2
import psycopg2.extras
import heapq # Needed for efficient multi-way merge (optional but better)

def _add_skip_pointers(postings_dict):
    """Adds skip pointers to a dictionary of postings {doc_id: data}.
    Returns a new dictionary potentially containing a '_skips' key.
    """
    doc_ids_only = [doc_id for doc_id in postings_dict if doc_id != '_skips']
    sorted_doc_ids = sorted(doc_ids_only)
    n = len(sorted_doc_ids)
    skip_distance = int(math.sqrt(n))

    if skip_distance <= 1:
        return postings_dict # No skips needed for short lists

    skips = {}
    i = 0
    while i < n:
        skip_to_index = min(i + skip_distance, n - 1)
        # Ensure we are actually skipping forward
        if i < skip_to_index:
             skips[sorted_doc_ids[i]] = sorted_doc_ids[skip_to_index]
        # Always advance by skip_distance to cover the list
        i += skip_distance

    if skips:
        postings_dict['_skips'] = skips
    return postings_dict


class Indexer:
    """Base class for different indexer types. Modified for incremental building."""
    def __init__(self):
        self.text_processor = TextProcessor()
        # These will hold the *final* merged index data after merge completes
        self.final_index = defaultdict(dict)
        self.final_doc_freq = defaultdict(int)
        self.final_total_docs = 0

    # In indexer.py, inside the base class Indexer:

    def merge_indices_to_postgres(self, block_filepaths, pg_store, compress=False, add_skips=False):
        """
        Memory-optimized merge function that writes directly to PostgreSQL
        one term at a time, instead of building a final index in memory.
        """
        print(f"Starting memory-optimized merge of {len(block_filepaths)} blocks to PostgreSQL...")

        if not block_filepaths:
            print("Warning: No temporary blocks found to merge.")
            return
        
        # --- Step 1: Build the term-to-block map (Still RAM-intensive) ---
        # This part is unavoidable with the current "load all blocks" design
        # It builds a map: {term: [(block_id, postings_dict), ...]}
        term_postings_map = defaultdict(list)
        total_docs_merged = 0
        print("Step 1: Reading terms from blocks (this may use significant RAM)...")
        for idx, block_path in enumerate(tqdm(block_filepaths, desc="Scanning Blocks")):
            try:
                 block_data = self.load(block_path) # Use static load method
                 block_index = block_data.get('index', {})
                 total_docs_merged += block_data.get('total_docs', 0)
                 for term, postings in block_index.items():
                      term_postings_map[term].append((idx, postings)) 
            except Exception as e:
                 print(f"Warning: Error loading or reading block {block_path}: {e}. Skipping.")

        print(f"\nFound {len(term_postings_map)} unique terms.")
        
        # --- Step 2: Clear DB and Insert Terms ---
        print("Step 2: Clearing old DB data and inserting terms...")
        pg_store.clear_index_data() # Drops indexes/constraints
        
        terms_to_insert = list(term_postings_map.keys())
        psycopg2.extras.execute_values(
            pg_store.cursor,
            "INSERT INTO terms (term_text) VALUES %s ON CONFLICT (term_text) DO NOTHING;",
            [(term,) for term in terms_to_insert],
            page_size=10000
        )
        pg_store.cursor.execute("SELECT term_text, term_id FROM terms;")
        term_map = dict(pg_store.cursor.fetchall())
        print("Terms inserted.")
        
        # --- Step 3: Merge, Prepare, and Insert Postings (Term-by-Term) ---
        print("Step 3: Merging and inserting postings term-by-term...")
        
        # Use a large batch size for execute_values
        BATCH_SIZE = 10000
        postings_to_insert_batch = []
        total_postings_saved = 0
        
        # Iterate and clear map to save RAM
        term_keys = list(term_postings_map.keys()) # Get keys first
        
        for term in tqdm(term_keys, desc="Merging & Saving Terms"):
            if term not in term_map:
                del term_postings_map[term] # Free memory
                continue
                
            term_id = term_map[term]
            all_postings_for_term = {}

            # Merge postings for this single term
            term_data_list = term_postings_map.pop(term) # pop() frees memory
            for block_id, postings in term_data_list:
                 all_postings_for_term.update(postings)
            
            # Add skips if needed
            if add_skips:
                all_postings_for_term = _add_skip_pointers(all_postings_for_term)
            
            # Prepare postings for insertion
            for doc_id, data in all_postings_for_term.items():
                if doc_id == '_skips': continue

                if isinstance(data, dict) and 'tf' in data and 'pos' in data:
                    tf, pos = data['tf'], data['pos']
                elif isinstance(data, list): 
                    tf, pos = len(data), data
                else:
                    continue # Skip malformed data

                positions_data = pickle.dumps(pos)
                if compress:
                    positions_data = zlib.compress(positions_data)
                
                postings_to_insert_batch.append((term_id, doc_id, tf, positions_data))
                total_postings_saved += 1
                
                # If batch is full, execute it
                if len(postings_to_insert_batch) >= BATCH_SIZE:
                    psycopg2.extras.execute_values(
                        pg_store.cursor,
                        "INSERT INTO postings (term_id, doc_id, term_frequency, positions) VALUES %s",
                        postings_to_insert_batch,
                        page_size=BATCH_SIZE
                    )
                    postings_to_insert_batch = [] # Reset batch
        
        # Insert any remaining postings
        if postings_to_insert_batch:
            psycopg2.extras.execute_values(
                pg_store.cursor,
                "INSERT INTO postings (term_id, doc_id, term_frequency, positions) VALUES %s",
                postings_to_insert_batch,
                page_size=BATCH_SIZE
            )
        
        print(f"Bulk data insertion complete ({total_postings_saved} total postings).")
        
        # --- Step 4: Recreate Constraints ---
        print("Step 4: Recreating constraints and indexes...")
        pg_store._recreate_constraints() # Call the datastore's helper
        
        print("PostgreSQL merge complete.")
        # Store total docs for the evaluator
        self.final_total_docs = total_docs_merged

    def build_index_chunk(self, document_chunk, doc_id_offset):
        """Processes a single chunk of documents and returns a partial index."""
        raise NotImplementedError("Subclasses must implement build_index_chunk")

    # In indexer.py, replace the 'merge_indices' function with this:

    def merge_indices(self, block_filepaths, final_filepath, compress=False, add_skips=False):
        """
        Merges partial index blocks (Pickle files) into a final index file.
        This version is memory-optimized to pop from the term map as it builds
        the final index, reducing peak RAM from ~2x to ~1x.
        """
        print(f"Starting memory-optimized merge of {len(block_filepaths)} into {final_filepath}...")

        if not block_filepaths:
            print("Warning: No temporary blocks found to merge.")
            return

        term_postings_map = defaultdict(list)
        total_docs_merged = 0

        print("Step 1: Reading terms from blocks (this may use significant RAM)...")
        for idx, block_path in enumerate(tqdm(block_filepaths, desc="Scanning Blocks")):
            try:
                 block_data = self.load(block_path) # Use static load method
                 block_index = block_data.get('index', {})
                 total_docs_merged += block_data.get('total_docs', 0)
                 for term, postings in block_index.items():
                      term_postings_map[term].append((idx, postings)) 
            except Exception as e:
                 print(f"Warning: Error loading or reading block {block_path}: {e}. Skipping.")

        print(f"\nFound {len(term_postings_map)} unique terms.")
        
        print(f"\nStep 2: Merging postings term-by-term...")
        final_merged_index = defaultdict(dict)
        final_doc_freq_merged = defaultdict(int)

        # Get all keys first, so we can safely modify the dict
        term_keys = list(term_postings_map.keys())

        for term in tqdm(term_keys, desc="Merging Terms"):
            all_postings_for_term = {}
            
            # Pop the term data from the map, freeing its memory
            term_data_list = term_postings_map.pop(term) 
            
            for block_id, postings in term_data_list:
                 all_postings_for_term.update(postings)

            # Add skip pointers here if requested
            if add_skips:
                all_postings_for_term = _add_skip_pointers(all_postings_for_term)

            # Calculate doc freq *before* storing, excluding skips
            final_doc_freq_merged[term] = len([doc_id for doc_id in all_postings_for_term if doc_id != '_skips'])
            
            # Store the final merged data for this term
            final_merged_index[term] = all_postings_for_term
            
        # After loop, term_postings_map is empty and final_merged_index is full
        
        print("\nStep 3: Storing final merged data...")
        self.final_index = final_merged_index
        self.final_doc_freq = final_doc_freq_merged
        self.final_total_docs = total_docs_merged

        # Save the final merged index
        print("\nStep 4: Saving final merged index to Pickle...")
        self.save(final_filepath, compress=compress)
        print("Merge complete.")


    def finalize_index(self):
        """Adds skip pointers to the final merged index. (Now called within merge_indices if needed)."""
        # This logic is now integrated into merge_indices to apply skips *before* the final save.
        # If you need to add skips *after* loading a final index, you could keep this,
        # but it's more efficient to do it during the merge.
        pass # Keep structure but logic moved


    def save(self, filepath, compress=False):
        """Saves the FINAL merged index to disk using Pickle."""
        data_to_save = {
            'index': self.final_index,
            'doc_freq': self.final_doc_freq,
            'total_docs': self.final_total_docs
        }
        serialized_data = pickle.dumps(data_to_save)
        if compress:
            print("Compressing final index with zlib...")
            serialized_data = zlib.compress(serialized_data)
        with open(filepath, 'wb') as f:
            f.write(serialized_data)
        print(f"Final index saved to {filepath}")

    @staticmethod
    def save_partial(filepath, index_data, compress=False):
        """Saves a PARTIAL index chunk to disk using Pickle."""
        try:
             serialized_data = pickle.dumps(index_data)
             if compress:
                  serialized_data = zlib.compress(serialized_data)
             with open(filepath, 'wb') as f:
                  f.write(serialized_data)
        except Exception as e:
             print(f"Error saving partial index {filepath}: {e}")


    @staticmethod
    def load(filepath):
        """Loads an index (partial or final) from disk."""
        try:
             with open(filepath, 'rb') as f:
                  raw_data = f.read()
             try:
                  # Try decompressing first
                  data = zlib.decompress(raw_data)
                  # print(f"Compressed block {filepath} loaded.") # Quieter log
                  return pickle.loads(data)
             except zlib.error:
                  # If decompression fails, it wasn't compressed
                  # print(f"Uncompressed block {filepath} loaded.") # Quieter log
                  return pickle.loads(raw_data)
        except FileNotFoundError:
             print(f"Error: Index file not found at {filepath}")
             return None # Return None if file doesn't exist
        except Exception as e:
             print(f"Error loading index file {filepath}: {e}")
             return None # Return None on other errors


# --- Stage 1: Positional Index ---
class PositionalIndexer(Indexer):
    def build_index_chunk(self, document_chunk, doc_id_offset):
        partial_index = defaultdict(dict)
        chunk_doc_count = 0
        for local_id, doc_text in enumerate(document_chunk):
            global_doc_id = doc_id_offset + local_id
            try: # Add basic error handling for text processing
                 tokens = self.text_processor.process(doc_text)
                 for pos, token in enumerate(tokens):
                      if global_doc_id not in partial_index[token]:
                           partial_index[token][global_doc_id] = []
                      partial_index[token][global_doc_id].append(pos)
            except Exception as e:
                 print(f"Warning: Error processing doc ID {global_doc_id}: {e}")
            chunk_doc_count += 1
        # Return data structure compatible with save/load format
        return {'index': partial_index, 'total_docs': chunk_doc_count}

# --- Stage 2: TF Index ---
class TF_Indexer(Indexer):
    def build_index_chunk(self, document_chunk, doc_id_offset):
        partial_index = defaultdict(dict)
        chunk_doc_count = 0
        for local_id, doc_text in enumerate(document_chunk):
            global_doc_id = doc_id_offset + local_id
            try: # Add basic error handling for text processing
                 tokens = self.text_processor.process(doc_text)
                 term_positions = defaultdict(list)
                 for pos, token in enumerate(tokens):
                      term_positions[token].append(pos)

                 for token, positions in term_positions.items():
                      tf = len(positions)
                      partial_index[token][global_doc_id] = {'tf': tf, 'pos': positions}
            except Exception as e:
                 print(f"Warning: Error processing doc ID {global_doc_id}: {e}")

            chunk_doc_count += 1
        return {'index': partial_index, 'total_docs': chunk_doc_count}

# --- Stage 3: TF-IDF Index ---
# TFIDF Indexer now primarily manages the merge and DF calculation
# It relies on TF_Indexer for chunk processing implicitly via merge
class TFIDF_Indexer(Indexer):
    # This class inherits the merge_indices logic from the base Indexer.
    # The merge_indices method correctly recalculates doc_freq after merging,
    # fulfilling the needs for TF-IDF.
    # It will use the TF_Indexer's build_index_chunk method via the main script logic.
    pass