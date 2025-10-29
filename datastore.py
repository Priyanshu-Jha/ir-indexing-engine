import zlib
import pickle
import psycopg2
import psycopg2.extras # Needed for the efficient bulk insert
from collections import defaultdict

from tqdm import tqdm

# This is a placeholder class definition.
# In your project, your 'indexer.py' file will define this properly.
# This is just here so the 'save_index(self, indexer, ...)' method signature is valid.
class Indexer:
    pass

class PostgresDataStore:
    def __init__(self, dbname='ir_project', user='postgres', password='root', host='localhost'):
        """Establishes a connection to the database and creates the schema."""
        try:
            # The script now connects DIRECTLY to your new database
            self.conn = psycopg2.connect(
                dbname=dbname, user=user, password=password, host=host
            )
            self.conn.autocommit = True 
            self.cursor = self.conn.cursor()
            print(f"Successfully connected to PostgreSQL database '{dbname}'!")
            self._create_schema()
        except psycopg2.OperationalError as e:
            print(f"FATAL: Could not connect to database '{dbname}'. Please ensure it has been created manually.")
            print(f"Error details: {e}")
            raise
        except Exception as e:
            print(f"An unexpected database error occurred: {e}")
            raise
    
    # In datastore.py, inside class PostgresDataStore:

    def get_doc_freqs_batch(self, term_texts: list):
        """Gets document frequencies (DFs) for a list of terms in one query."""
        if not term_texts:
            return {}
        
        # Get term_ids for the given texts
        term_id_map = self.get_term_ids(term_texts) # This is {term_text: term_id}
        if not term_id_map:
            return {}
            
        term_id_to_text_map = {v: k for k, v in term_id_map.items()} # Flip to {term_id: term_text}

        # Run one query to get counts for all term_ids
        sql_query = """
            SELECT term_id, COUNT(doc_id)
            FROM postings
            WHERE term_id IN %s
            GROUP BY term_id;
        """
        self.cursor.execute(sql_query, (tuple(term_id_map.values()),))
        results = self.cursor.fetchall() # This is [(term_id, count), ...]
        
        # Map back to term_text
        df_map = {}
        for term_id, count in results:
            term_text = term_id_to_text_map.get(term_id)
            if term_text:
                df_map[term_text] = count
                
        # Any terms not in the result have a DF of 0
        for term_text in term_texts:
            if term_text not in df_map:
                df_map[term_text] = 0
                
        return df_map
    
    def get_term_id(self, term_text):
        """Finds the term_id for a given term text."""
        self.cursor.execute("SELECT term_id FROM terms WHERE term_text = %s;", (term_text,))
        result = self.cursor.fetchone()
        return result[0] if result else None

    def get_term_ids(self, term_texts: list):
        """Finds term_ids for a list of term texts in one query."""
        if not term_texts:
            return {}
        # Use a temporary table or VALUES list for efficient lookup
        # Using VALUES is generally simpler
        sql_query = "SELECT term_text, term_id FROM terms WHERE term_text IN %s;"
        try:
            self.cursor.execute(sql_query, (tuple(term_texts),))
            results = self.cursor.fetchall()
            return dict(results) # Returns a map {term_text: term_id}
        except Exception as e:
            print(f"Error getting batch term IDs: {e}")
            return {}

    def get_doc_freq(self, term_text):
        """Gets the document frequency (DF) for a term."""
        term_id = self.get_term_id(term_text)
        if term_id is None:
            return 0
        self.cursor.execute("SELECT COUNT(doc_id) FROM postings WHERE term_id = %s;", (term_id,))
        result = self.cursor.fetchone()
        return result[0] if result else 0

    def get_total_docs_from_db(self):
         """ Gets the total number of indexed documents, e.g., by counting distinct doc_ids """
         print("Estimating total docs from postings...")
         self.cursor.execute("SELECT COUNT(DISTINCT doc_id) FROM postings;")
         result = self.cursor.fetchone()
         count = result[0] if result else 0
         print(f"Estimated {count} total documents.")
         return count


    def get_postings_for_term(self, term_text, compress=False):
        """
        Fetches postings {doc_id: {'tf': tf, 'pos': [positions]}} for a single term
        directly from the database. Handles decompression.
        Returns a dictionary or None if term not found.
        """
        term_id = self.get_term_id(term_text)
        if term_id is None:
            return None # Term not in index

        self.cursor.execute("""
            SELECT doc_id, term_frequency, positions
            FROM postings
            WHERE term_id = %s;
        """, (term_id,))

        rows = self.cursor.fetchall()
        # Use the helper function to reconstruct
        return self._reconstruct_postings_from_rows(rows, term_text)


    def get_postings_for_terms_batch(self, term_ids: list):
        """
        Fetches all postings for a list of term_ids in a single query.
        Returns a map {term_id: {doc_id: {'tf': tf, 'pos': [positions]}}}
        """
        if not term_ids:
            return {}

        # Fetch all postings related to the term_ids
        try:
            self.cursor.execute("""
                SELECT term_id, doc_id, term_frequency, positions
                FROM postings
                WHERE term_id IN %s;
            """, (tuple(term_ids),))
            
            rows = self.cursor.fetchall()
            
            # Group rows by term_id
            term_id_to_rows = defaultdict(list)
            for term_id, doc_id, tf, positions_data in rows:
                # Store the row data *without* the term_id
                term_id_to_rows[term_id].append((doc_id, tf, positions_data))

            # Reconstruct postings for each term
            final_postings_map = {}
            for term_id, term_rows in term_id_to_rows.items():
                # Pass a dummy term_text for error logging, as we don't have it here
                final_postings_map[term_id] = self._reconstruct_postings_from_rows(term_rows, f"term_id_{term_id}")
                
            return final_postings_map
        except Exception as e:
            print(f"Error getting batch postings: {e}")
            return {}

    def _reconstruct_postings_from_rows(self, rows, term_identifier):
        """
        Helper function to turn DB rows into a postings dictionary.
        Rows are expected to be (doc_id, tf, positions_data).
        """
        postings = {}
        for doc_id, tf, positions_data in rows:
            try:
                # --- Robust Loading Logic ---
                if isinstance(positions_data, memoryview):
                     positions_data = bytes(positions_data)
                decompressed_data = zlib.decompress(positions_data)
                positions = pickle.loads(decompressed_data)
            except zlib.error:
                try:
                    positions = pickle.loads(positions_data)
                except (pickle.UnpicklingError, TypeError) as e_pickle:
                    print(f"Warning: DB Query - Could not decode positions for term '{term_identifier}' in doc {doc_id}. Skipping posting. Error: {e_pickle}")
                    continue # Skip this specific posting
            except (pickle.UnpicklingError, TypeError) as e_pickle_after_decompress:
                 print(f"Warning: DB Query - Decompressed data invalid for term '{term_identifier}' in doc {doc_id}. Skipping posting. Error: {e_pickle_after_decompress}")
                 continue

            postings[doc_id] = {'tf': tf, 'pos': positions}
        return postings if postings else None

    
    def _create_schema(self):
        """Creates the necessary tables if they don't already exist."""
        self.cursor.execute("CREATE TABLE IF NOT EXISTS terms (term_id SERIAL PRIMARY KEY, term_text TEXT UNIQUE NOT NULL);")
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS postings (
                term_id INTEGER, -- Foreign key added *after* bulk insert
                doc_id INTEGER,
                term_frequency INTEGER,
                positions BYTEA  -- Use BYTEA for flexibility
                -- Primary key added *after* bulk insert
            );
        """)
        # Indexes are now created *after* insertion in save_index
        print("Schema verified (tables created if not exist).")
        
    def clear_index_data(self):
        """Deletes all data from the tables AND drops constraints/indexes for fast re-insertion."""
        # Drop constraints and indexes first, as TRUNCATE is faster without them
        self.cursor.execute("ALTER TABLE postings DROP CONSTRAINT IF EXISTS postings_term_id_fkey;")
        self.cursor.execute("ALTER TABLE postings DROP CONSTRAINT IF EXISTS postings_pkey;")
        self.cursor.execute("DROP INDEX IF EXISTS postings_term_id_idx;")
        self.cursor.execute("DROP INDEX IF EXISTS postings_doc_id_idx;")
        
        self.cursor.execute("TRUNCATE TABLE postings, terms RESTART IDENTITY;")
        print("Existing index data, constraints, and indexes cleared from PostgreSQL.")

    def _recreate_constraints(self):
        """
        Re-creates all indexes and constraints on the tables.
        Call this AFTER all data has been inserted.
        """
        print("Re-creating primary key on postings...")
        # Add primary key (term_id, doc_id)
        self.cursor.execute("""
            ALTER TABLE postings
            ADD PRIMARY KEY (term_id, doc_id);
        """)
        
        print("Re-creating foreign key constraint on postings...")
        # Add foreign key constraint
        self.cursor.execute("""
            ALTER TABLE postings
            ADD CONSTRAINT postings_term_id_fkey
            FOREIGN KEY (term_id) REFERENCES terms(term_id) ON DELETE CASCADE;
        """)

        print("Re-creating index postings_term_id_idx...")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS postings_term_id_idx ON postings (term_id);")
        
        print("Re-creating index postings_doc_id_idx...")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS postings_doc_id_idx ON postings (doc_id);")
        print("All constraints and indexes recreated.")

    def save_index(self, indexer, compress=False):
        """
        Saves the FINAL MERGED index (from indexer.final_index)
        into the PostgreSQL database using a high-performance method.
        """
        print(f"Saving FINAL index to PostgreSQL (Compression: {compress})...")
        
        # 1. Clear tables AND drop constraints for high-speed insert
        self.clear_index_data() # This now drops constraints/indexes too

        if not hasattr(indexer, 'final_index') or not indexer.final_index:
             print("Error: Indexer object does not have final_index data to save.")
             return

        terms_to_insert = list(indexer.final_index.keys())

        # 2. Insert terms (this is fast and fine, but let's add page_size)
        print("Inserting terms...")
        psycopg2.extras.execute_values(
            self.cursor,
            "INSERT INTO terms (term_text) VALUES %s ON CONFLICT (term_text) DO NOTHING;",
            [(term,) for term in terms_to_insert],
            page_size=10000 # Add page_size here too
        )
        self.cursor.execute("SELECT term_text, term_id FROM terms;")
        term_map = dict(self.cursor.fetchall())

        # 3. Prepare postings data (this is fine)
        postings_to_insert = []
        print("Preparing final postings for database insertion...")
        for term, postings_dict in tqdm(indexer.final_index.items(), desc="Preparing Final Postings"):
            if term not in term_map:
                # This might happen if ON CONFLICT DO NOTHING skipped a term somehow, unlikely but safe check
                print(f"Warning: Term '{term}' from final index not found in term_map after insert. Skipping.")
                continue
            term_id = term_map[term]

            # Iterate through doc_ids and their data in the postings dictionary
            for doc_id, data in postings_dict.items():
                if doc_id == '_skips': continue # Skip the skip pointer key

                # Extract TF and Pos, handling different structures
                if isinstance(data, dict) and 'tf' in data and 'pos' in data:
                    tf, pos = data['tf'], data['pos']
                elif isinstance(data, list): # Positional index structure
                    tf, pos = len(data), data
                else:
                     print(f"Warning: Unexpected data format for term '{term}', doc_id {doc_id}. Skipping posting.")
                     continue

                # --- Compression Logic ---
                positions_data = pickle.dumps(pos)
                if compress:
                    positions_data = zlib.compress(positions_data)

                postings_to_insert.append((term_id, doc_id, tf, positions_data))

        # 4. Insert postings (now very fast, as there are no indexes/constraints)
        print(f"Inserting {len(postings_to_insert)} posting entries into database...")
        
        # --- THIS IS THE CRITICAL FIX ---
        psycopg2.extras.execute_values(
            self.cursor,
            "INSERT INTO postings (term_id, doc_id, term_frequency, positions) VALUES %s",
            postings_to_insert,
            page_size=10000 # <--- THIS IS THE FIX
        )
        # ---------------------------------
        
        print("Bulk data insertion complete.")
        
        # 5. Recreate all constraints and indexes
        self._recreate_constraints()
        
        print("Save to PostgreSQL complete.")

    def load_index_from_db(self, compress=None): # compress flag is optional/ignored
        """
        Loads the entire index from PostgreSQL, automatically detecting compression,
        calculates total_docs, and reconstructs the Python dictionary structure.
        """
        print(f"Loading index from PostgreSQL (auto-detecting {compress} compression)...")
        index = defaultdict(dict)
        doc_freq = defaultdict(int)
        loaded_doc_ids = set() # --- ADDED: To track distinct doc IDs ---

        self.cursor.execute("""
            SELECT t.term_text, p.doc_id, p.term_frequency, p.positions
            FROM terms t
            JOIN postings p ON t.term_id = p.term_id
        """)

        rows = self.cursor.fetchall()

        for term_text, doc_id, tf, positions_data in tqdm(rows, desc="Reconstructing Index"):
            loaded_doc_ids.add(doc_id) # --- ADDED: Record every doc ID encountered ---
            try:
                # --- Robust Loading Logic (Unchanged) ---
                if isinstance(positions_data, memoryview): # Handle memoryview if returned
                     positions_data = bytes(positions_data)
                decompressed_data = zlib.decompress(positions_data)
                positions = pickle.loads(decompressed_data)
            except zlib.error:
                try:
                    positions = pickle.loads(positions_data)
                except (pickle.UnpicklingError, TypeError) as e_pickle: # Added TypeError
                    print(f"Warning: Could not decode positions for term '{term_text}' in doc {doc_id}. Skipping. Error: {e_pickle}")
                    continue
            except (pickle.UnpicklingError, TypeError) as e_pickle_after_decompress: # Added TypeError
                 print(f"Warning: Decompressed data invalid for term '{term_text}' in doc {doc_id}. Skipping. Error: {e_pickle_after_decompress}")
                 continue
            # --- END ROBUST LOADING ---

            # Rebuild index structure (Unchanged)
            if tf is not None:
                index[term_text][doc_id] = {'tf': tf, 'pos': positions}
            else:
                index[term_text][doc_id] = positions # Fallback

        # Reconstruct doc_freq (Unchanged)
        print("\nCalculating document frequencies from loaded data...")
        for term, postings in index.items():
            doc_freq[term] = len(postings)

        # --- ADDED: Calculate total_docs ---
        total_docs_loaded = len(loaded_doc_ids)
        print(f"Successfully loaded index data for {total_docs_loaded} documents from PostgreSQL.")

        # --- Include total_docs in the returned dictionary ---
        return {'index': index, 'doc_freq': doc_freq, 'total_docs': total_docs_loaded}

    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        print("PostgreSQL connection closed.")