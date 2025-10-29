import re
from collections import defaultdict
from core import TextProcessor  # Ensure core.py is present and correct
import math
import tqdm
import heapq # For efficiently finding top-k documents
import traceback # For debugging
import pickle 
import zlib   
try:
    from datastore import PostgresDataStore
except ImportError:
    print("Warning: PostgresDataStore not found in datastore.py. Postgres mode might fail.")
    PostgresDataStore = None 


class QueryParser:
    """
    Parses a boolean/phrase query string into Reverse Polish Notation (RPN)
    using the Shunting-yard algorithm.
    """
    def __init__(self):
        # Operator precedence for the Shunting-yard algorithm
        self.precedence = {'NOT': 3, 'AND': 2, 'OR': 1}

    def _tokenize_query(self, query_string):
        """
        Tokenizes the query string into terms, operators, and parentheses.
        """
        # Regex to find quoted terms, operators, or parentheses
        token_regex = r'\"(?:\\.|[^\"\\])*\"|\bAND\b|\bOR\b|\bNOT\b|\(|\)'
        tokens = re.findall(token_regex, query_string)
        return tokens

    def to_rpn(self, query_string):
        """
        Converts an infix query string to Reverse Polish Notation (RPN).
        """
        tokens = self._tokenize_query(query_string)
        output_queue = []
        operator_stack = []

        for token in tokens:
            if token.startswith('"'):  # It's a term or phrase
                output_queue.append(token)
            elif token in self.precedence:  # It's an operator (AND, OR, NOT)
                while (operator_stack and operator_stack[-1] != '(' and
                       self.precedence.get(operator_stack[-1], 0) >= self.precedence.get(token, 0)):
                    output_queue.append(operator_stack.pop())
                operator_stack.append(token)
            elif token == '(':
                operator_stack.append(token)
            elif token == ')':
                while operator_stack and operator_stack[-1] != '(':
                    output_queue.append(operator_stack.pop())
                # If stack runs out without finding '(', parentheses were mismatched
                if not operator_stack or operator_stack[-1] != '(':
                    raise ValueError("Mismatched parentheses in query")
                operator_stack.pop()  # Discard the left parenthesis '('

        # Pop any remaining operators from stack to output
        while operator_stack:
            # If remaining operator is '(', parentheses were mismatched
            if operator_stack[-1] == '(':
                 raise ValueError("Mismatched parentheses in query")
            output_queue.append(operator_stack.pop())

        return output_queue


class QueryEvaluator:
    """
    Evaluates queries. Loads index for Pickle, queries DB directly for PostgreSQL.
    Supports TAAT (Boolean/Phrase) and DAAT (Ranked) methods.
    """
    def __init__(self, index_data_or_datastore, documents_list_for_eval, indexing_stage='positional', config=None):
        """
        Initializes based on datastore type.
        Args:
            index_data_or_datastore: EITHER the loaded index dict (Pickle) OR a PostgresDataStore instance.
            documents_list_for_eval: List of document texts (used for context/snippets).
            indexing_stage: The type of index ('positional', 'tf', 'tfidf').
            config: The full configuration dictionary.
        """
        self.config = config if config else {} # Store config
        self.datastore_type = self.config.get('datastore', 'pickle')
        # Compression needed for DB fetch interpretation
        self.use_compression = (self.config.get('compression') == 'zlib')
        # Optimization relevant only for Pickle TAAT
        self.use_skip_pointers = (self.config.get('optimization') == 'skip_pointers')
        self.indexing_stage = indexing_stage
        self.documents = documents_list_for_eval # Keep reference (can use lots of RAM)
        self.text_processor = TextProcessor()

        self.index = None           # In-memory index (Pickle only)
        self.doc_freq = None        # In-memory doc_freq (Pickle only)
        self.total_docs = 0         # Total documents in the index
        self.pg_store = None        # Reference to PostgresDataStore

        if self.datastore_type == 'postgres':
            # Ensure a valid, connected PostgresDataStore instance is provided
            if PostgresDataStore is not None and isinstance(index_data_or_datastore, PostgresDataStore) \
               and index_data_or_datastore.conn and index_data_or_datastore.conn.closed == 0:
                 self.pg_store = index_data_or_datastore # Keep reference
                 # Get total_docs from DB. This could be slow. Consider storing separately.
                 self.total_docs = self.pg_store.get_total_docs_from_db()
            else:
                 # If no valid connection, evaluator cannot function in Postgres mode
                 raise ValueError("Postgres mode requires a valid, connected PostgresDataStore instance.")

        else: # Pickle mode - Load index data directly into memory
             if not isinstance(index_data_or_datastore, dict):
                  raise TypeError("Expected index_data dictionary for pickle datastore mode.")
             self.index = index_data_or_datastore.get('index', defaultdict(dict))
             self.doc_freq = index_data_or_datastore.get('doc_freq', defaultdict(int))
             self.total_docs = index_data_or_datastore.get('total_docs', 0)
             # Fallback if total_docs missing from pickle
             if self.total_docs == 0 and documents_list_for_eval:
                  self.total_docs = len(documents_list_for_eval)
                  if self.total_docs > 0: print("Warning: total_docs read as 0 from Pickle, using len(documents_list_for_eval).")

        # Ensure total_docs is reasonable, use list length as fallback
        self.total_docs = max(self.total_docs, len(documents_list_for_eval or []))
        if self.total_docs == 0: print("Warning: total_docs is zero. IDF calculations will be affected.")

        # Create set of all doc IDs based on total_docs
        self.all_doc_ids = set(range(self.total_docs))


    # --- Data Fetching Methods (Abstracted) ---
    def _fetch_postings_dict(self, term):
        """
        Fetches postings dictionary {doc_id: data} for a single processed term.
        Handles DB query or memory lookup. Returns empty dict if not found or on error.
        (Used primarily by TAAT for single-term lookups)
        """
        processed_tokens = self.text_processor.process(term.strip('"'))
        if not processed_tokens: return {}
        processed_term = processed_tokens[0]

        try:
            if self.datastore_type == 'postgres':
                if not self.pg_store or self.pg_store.conn.closed != 0:
                     print(f"Error: No valid PostgreSQL connection to fetch term '{processed_term}'.")
                     return {}
                # Fetch from DB (datastore method handles decompression based on its knowledge)
                # It returns dict {doc_id: {'tf':tf, 'pos':pos_list}} or None
                postings = self.pg_store.get_postings_for_term(processed_term)
                # Skips are NOT loaded/reconstructed from DB in current design
                return postings if postings else {}
            else: # Pickle mode
                # Return a copy from the in-memory index (may include skips)
                return self.index.get(processed_term, {}).copy()
        except Exception as e:
            print(f"Error fetching postings for term '{processed_term}': {e}")
            traceback.print_exc() # Print full error for debugging
            return {}

    def _get_doc_freq_for_term(self, term):
         """ Gets DF for a processed term, from DB or memory """
         # *** NOTE: This still re-processes the term every time ***
         # This is fine for the DAAT pre-fetch (called once per term)
         # but was the cause of the bug inside _calculate_score.
         processed_tokens = self.text_processor.process(term)
         if not processed_tokens: return 0
         processed_term = processed_tokens[0]

         try:
             if self.datastore_type == 'postgres':
                  if not self.pg_store or self.pg_store.conn.closed != 0: return 0
                  # Fetch DF directly from the database
                  return self.pg_store.get_doc_freq(processed_term)
             else: # Pickle
                  # Use the in-memory doc_freq dictionary
                  return self.doc_freq.get(processed_term, 0)
         except Exception as e:
              print(f"Error getting doc freq for term '{processed_term}': {e}")
              return 0 # Return 0 on error


    # --- TAAT Helper Methods ---
    def _intersect_simple(self, op1_data, op2_data):
         """ Simple set intersection using only doc_ids """
         try:
              # Input can be dict or set
              set1 = set(op1_data.keys()) - {'_skips'} if isinstance(op1_data, dict) else set(op1_data)
              set2 = set(op2_data.keys()) - {'_skips'} if isinstance(op2_data, dict) else set(op2_data)
              return set1 & set2
         except TypeError as e:
              print(f"Error creating sets for intersection: {e}. Op1 type: {type(op1_data)}, Op2 type: {type(op2_data)}")
              return set() # Return empty set on error

    def _intersect_with_skips(self, postings1_data, postings2_data):
        """Intersects two posting list dictionaries using skip pointers (Pickle Only). Returns SET."""
        result = set()
        # Ensure inputs are dictionaries for skip logic
        if not isinstance(postings1_data, dict) or not isinstance(postings2_data, dict):
             print(f"Warning: _intersect_with_skips called with non-dict types. Falling back.")
             return self._intersect_simple(postings1_data, postings2_data)
        try:
            # Ensure doc_ids are integers
            doc_ids1 = sorted([int(doc_id) for doc_id in postings1_data if doc_id != '_skips'])
            doc_ids2 = sorted([int(doc_id) for doc_id in postings2_data if doc_id != '_skips'])
        except (ValueError, TypeError) as e: print(f"Error converting doc_ids: {e}"); return set()

        if not doc_ids1 or not doc_ids2: return result # One list is empty

        skips1 = postings1_data.get('_skips', {}); skips2 = postings2_data.get('_skips', {})
        p1, p2 = 0, 0

        while p1 < len(doc_ids1) and p2 < len(doc_ids2):
            doc1 = doc_ids1[p1]; doc2 = doc_ids2[p2]
            if doc1 == doc2: result.add(doc1); p1 += 1; p2 += 1
            elif doc1 < doc2:
                skip_target = skips1.get(doc1)
                if skip_target and skip_target <= doc2:
                    while p1 < len(doc_ids1) and doc_ids1[p1] < skip_target: p1 += 1
                else: p1 += 1
            else: # doc2 < doc1
                skip_target = skips2.get(doc2)
                if skip_target and skip_target <= doc1:
                    while p2 < len(doc_ids2) and doc_ids2[p2] < skip_target: p2 += 1
                else: p2 += 1
        return result

    def _union(self, set1, set2):
        """Returns the union of two sets of doc IDs."""
        return set1 | set2

    def _not(self, doc_set):
        """Returns the complement of a set of doc IDs."""
        if not isinstance(doc_set, set):
             try: doc_set = set(doc_set)
             except TypeError: print("Error: Cannot convert operand for NOT to set."); return set()
        return self.all_doc_ids - doc_set

    # --- Phrase Query Methods ---
    def _phrase_query_logic(self, phrase, postings_source_map):
         """Core phrase query logic, fetching postings from a pre-fetched map."""
         terms = self.text_processor.process(phrase.strip('"'))
         if not terms: return set()

         # Fetch postings for the first term
         first_term = terms[0]
         first_term_postings = postings_source_map.get(first_term, {})
         if not first_term_postings: return set()
         # Use only doc_ids present in the dictionary keys
         result_doc_ids = set(int(k) for k in first_term_postings.keys() if k != '_skips')
         if not result_doc_ids: return set()

         # Iteratively filter based on subsequent terms
         for i in range(1, len(terms)):
             current_term = terms[i]
             current_term_postings = postings_source_map.get(current_term, {})
             if not current_term_postings: return set()

             docs_with_current_term = set(int(k) for k in current_term_postings.keys() if k != '_skips')
             result_doc_ids &= docs_with_current_term # Intersect possible documents
             if not result_doc_ids: return set()

             # Check positions in the remaining candidates
             docs_to_remove = set()
             for doc_id in result_doc_ids:
                 try:
                     # Get positions, handling different structures
                     prev_data = first_term_postings[doc_id]
                     curr_data = current_term_postings[doc_id]
                     prev_pos = prev_data['pos'] if isinstance(prev_data, dict) else prev_data
                     curr_pos = curr_data['pos'] if isinstance(curr_data, dict) else curr_data
                     curr_pos_set = set(curr_pos)
                     found_adj = any((p + i) in curr_pos_set for p in prev_pos)
                     if not found_adj: docs_to_remove.add(doc_id)
                 except KeyError: docs_to_remove.add(doc_id) # Doc missing term after intersection? Safety.
                 except Exception as e_phrase: print(f"Warn: Phrase pos check error doc {doc_id}: {e}"); docs_to_remove.add(doc_id)

             result_doc_ids -= docs_to_remove
             if not result_doc_ids: return set()

         return result_doc_ids

    def _phrase_query_pickle(self, phrase):
         """Wrapper for phrase query using in-memory index."""
         if self.index is None: return set() # Check if index loaded
         # We need to provide a map of {processed_term: postings}
         # This is slightly inefficient but works
         processed_terms = list(set(self.text_processor.process(phrase.strip('"'))))
         pre_fetched_map = {}
         for term in processed_terms:
             if term in self.index:
                 pre_fetched_map[term] = self.index[term]
         
         return self._phrase_query_logic(phrase, pre_fetched_map)

    def _phrase_query_db(self, phrase):
         """Wrapper for phrase query fetching term-by-term from DB."""
         if not self.pg_store: return set()
         
         processed_terms = list(set(self.text_processor.process(phrase.strip('"'))))
         if not processed_terms:
             return set()
             
         # 1. Get all term_ids in one go
         term_id_map = self.pg_store.get_term_ids(processed_terms) # {term_text: term_id}
         if not term_id_map:
             return set()
             
         # 2. Fetch all postings in one go
         term_id_to_postings_map = self.pg_store.get_postings_for_terms_batch(list(term_id_map.values())) # {term_id: postings_dict}
         
         # 3. Create the term_text -> postings_dict map needed by the logic function
         pre_fetched_map = {}
         for term_text, term_id in term_id_map.items():
             postings = term_id_to_postings_map.get(term_id)
             if postings:
                 pre_fetched_map[term_text] = postings
                 
         return self._phrase_query_logic(phrase, pre_fetched_map)


    def evaluate_rpn_taat(self, rpn_query):
        """Evaluates Boolean/Phrase RPN query using Term-at-a-Time."""
        if not rpn_query: return []
        stack = []
        for token in rpn_query:
            try:
                if token == "AND":
                    op2 = stack.pop() # Can be dict (Pickle) or set (intermediate/Postgres term result)
                    op1 = stack.pop()

                    # Use skips ONLY if using Pickle AND optimization enabled AND both are dicts
                    if (self.datastore_type == 'pickle' and self.use_skip_pointers and
                        isinstance(op1, dict) and isinstance(op2, dict)):
                         result_data = self._intersect_with_skips(op1, op2) # Returns SET
                    else: # Otherwise, use simple set intersection
                         result_data = self._intersect_simple(op1, op2) # Returns SET
                    stack.append(result_data)

                elif token == "OR":
                    op2 = stack.pop()
                    op1 = stack.pop()
                    # Convert dicts to sets if necessary
                    set1 = set(op1.keys()) - {'_skips'} if isinstance(op1, dict) else set(op1)
                    set2 = set(op2.keys()) - {'_skips'} if isinstance(op2, dict) else set(op2)
                    stack.append(self._union(set1, set2)) # Push SET

                elif token == "NOT":
                    operand = stack.pop()
                    op_set = set(operand.keys()) - {'_skips'} if isinstance(operand, dict) else set(operand)
                    stack.append(self._not(op_set)) # Push SET

                else: # Term or Phrase
                    if ' ' in token.strip('"'): # Phrase
                         if self.datastore_type == 'postgres':
                              stack.append(self._phrase_query_db(token)) # Returns SET
                         else: # Pickle
                              stack.append(self._phrase_query_pickle(token)) # Returns SET
                    else: # Single Term
                        # Fetch postings data
                        # For Pickle, includes skips if optimization enabled (returns dict)
                        # For Postgres, just {doc_id: data} (returns dict)
                        postings_data = self._fetch_postings_dict(token)
                        # Push dict for Pickle, or SET of doc_ids for Postgres/empty
                        if self.datastore_type == 'postgres':
                             stack.append(set(postings_data.keys()))
                        else: # Pickle
                             stack.append(postings_data if postings_data else set())

            except IndexError: raise ValueError(f"Invalid RPN sequence near operator '{token}'.")
            except Exception as e: print(f"Error during RPN eval token '{token}': {e}"); traceback.print_exc(); raise

        if not stack: return []
        final_result = stack[0]
        # Ensure final result is a set before sorting
        final_result_set = set()
        if isinstance(final_result, dict): final_result_set = set(final_result.keys()) - {'_skips'}
        elif isinstance(final_result, set): final_result_set = final_result
        else:
            try: final_result_set = set(final_result)
            except TypeError: print(f"Error: Cannot convert final RPN result to set."); return []

        return sorted(list(final_result_set))


    # --- DAAT Methods (Ranked Retrieval) ---

    # --- FIX 1: MODIFIED _calculate_score ---
    def _calculate_score(self, tf, df, total_docs):
        """
        Calculates TF or TF-IDF score based on the indexing stage.
        Accepts pre-fetched df and total_docs to avoid re-querying/re-processing.
        """
        try: tf = float(tf)
        except (ValueError, TypeError): return 0.0

        if self.indexing_stage == 'tf':
            return tf if tf > 0 else 0.0 # Score is just TF
        
        elif self.indexing_stage == 'tfidf':
            # Check for valid inputs
            if tf <= 0 or total_docs <= 0 or df <= 0: return 0.0
            try:
                 idf = math.log10(total_docs / df) # Standard IDF
            except ValueError: # Avoid math domain error if df > total_docs
                 idf = 0.0
            tf_score = tf # Using raw TF
            return tf_score * idf
        else: # Positional fallback score
            return 1.0 if tf > 0 else 0.0

    def _get_daat_postings_pickle(self, query_terms):
        """Fetches DAAT data from in-memory Pickle index."""
        term_postings_data = {}
        active_terms = []
        for term in query_terms:
            postings = self.index.get(term, {})
            if postings:
                # Filter out skips key before passing to iterator logic
                term_postings_data[term] = {k:v for k,v in postings.items() if k != '_skips'}
                active_terms.append(term)
        return term_postings_data, active_terms

    def _get_daat_postings_db(self, query_terms):
        """Fetches DAAT data from Postgres in a single batch."""
        term_postings_data = {}
        active_terms = []
        if not self.pg_store or not query_terms:
            return {}, []
            
        # 1. Get all term_ids in one go
        term_id_map = self.pg_store.get_term_ids(query_terms) # {term_text: term_id}
        if not term_id_map:
            return {}, []
            
        # 2. Fetch all postings in one go
        term_id_to_postings_map = self.pg_store.get_postings_for_terms_batch(list(term_id_map.values())) # {term_id: postings_dict}
         
        # 3. Create the term_text -> postings_dict map
        for term_text, term_id in term_id_map.items():
            postings = term_id_to_postings_map.get(term_id)
            if postings:
                term_postings_data[term_text] = postings
                active_terms.append(term_text)
                
        return term_postings_data, active_terms


    # --- FIX 2: MODIFIED evaluate_daat_ranked ---
    def evaluate_daat_ranked(self, query_string, k=10):
        """
        Evaluates a simple bag-of-words query using Document-at-a-Time
        and TF or TF-IDF ranking. (Optimized to pre-fetch DFs)
        """
        if self.indexing_stage == 'positional':
             # Fallback logic: treat as OR, score is 1 if present
             print("Warning: DAAT Ranked retrieval called on Positional index. Using simple presence scoring.")
             query_terms = self.text_processor.process(query_string)
             if not query_terms: return []
             present_docs = set()
             for term in query_terms:
                  postings = self._fetch_postings_dict(f'"{term}"') # Fetch relevant postings
                  present_docs.update(postings.keys() - {'_skips'}) # Exclude skips just in case
             # Return doc_ids with a dummy score of 1.0, limited to k
             return [(doc_id, 1.0) for doc_id in sorted(list(present_docs))[:k]]

        # --- Main DAAT logic for TF or TF-IDF ---
        query_terms = list(set(self.text_processor.process(query_string))) # Unique processed terms
        if not query_terms: return []

        term_postings_data = {} # Store postings {doc_id: data} fetched/retrieved
        posting_iterators = {}  # Store iterators over sorted doc_ids
        active_terms = []       # List of terms found in the index

        # --- Fetch all necessary data upfront (Optimized) ---
        print(f"DAAT ({self.datastore_type}): Fetching postings...")
        if self.datastore_type == 'postgres':
            term_postings_data, active_terms = self._get_daat_postings_db(query_terms)
        else: # Pickle
            term_postings_data, active_terms = self._get_daat_postings_pickle(query_terms)
            
        # --- NEW OPTIMIZATION: PRE-FETCH ALL DFs ---
        term_dfs = {}
        if self.indexing_stage == 'tfidf':
            print("DAAT: Pre-fetching document frequencies (DFs)...")
            # For Postgres, this is N queries, but it's N queries *once*
            # not N*M queries inside the loop.
            # For Pickle, this is N lookups in the self.doc_freq dict.
            if self.datastore_type == 'postgres' and self.pg_store:
                term_dfs = self.pg_store.get_doc_freqs_batch(active_terms)
            else:
                # Pickle mode (or fallback)
                for term in active_terms:
                    term_dfs[term] = self._get_doc_freq_for_term(term)
        # ---------------------------------------------

        # --- Build Iterators (common logic) ---
        for term in active_terms:
            postings = term_postings_data.get(term, {})
            if postings:
                 try:
                      # Doc IDs are already filtered for _skips by the fetch helpers
                      doc_ids = sorted([int(doc_id) for doc_id in postings])
                      if doc_ids:
                           posting_iterators[term] = iter(doc_ids)
                 except (ValueError, TypeError) as e_docid:
                      print(f"Warning: Could not parse doc IDs for term '{term}'. Skipping. Error: {e_docid}")
                      if term in active_terms: active_terms.remove(term) # Remove from active list
                      continue 
        print("DAAT: Data/Iterators ready.")

        if not posting_iterators: return [] # No query terms found

        # --- Initialize iterators ---
        current_doc_ids = {}
        # Make copy of active_terms to safely remove from it during init
        terms_to_init = list(posting_iterators.keys()) # Use keys from created iterators
        for term in terms_to_init:
            try:
                current_doc_ids[term] = next(posting_iterators[term])
            except StopIteration:
                # This term's list was empty after filtering?
                posting_iterators.pop(term, None) # Remove if empty

        top_k_heap = [] # Stores (score, doc_id) tuples
        total_docs_for_idf = float(self.total_docs) # Cache this as float

        # --- Main DAAT Loop ---
        while current_doc_ids:
            # Find the smallest current document ID across all active iterators
            min_doc_id = min(current_doc_ids.values())

            current_doc_total_score = 0.0
            terms_in_min_doc = [] # Track which terms need their iterator advanced

            # Calculate score for this document by summing contributions
            # Iterate through terms currently active in the iterators
            # Use list copy for safe removal within loop if StopIteration occurs unexpectedly
            for term in list(current_doc_ids.keys()):
                if current_doc_ids.get(term) == min_doc_id:
                    # This term is in the current minimum document ID
                    posting_info = term_postings_data.get(term, {}).get(min_doc_id)

                    if posting_info is not None:
                        # Extract TF robustly (handles list for positional case - although filtered above)
                        tf = 0
                        if isinstance(posting_info, dict) and 'tf' in posting_info:
                             tf = posting_info['tf']
                        elif isinstance(posting_info, list):
                             tf = len(posting_info)
                        
                        # --- USE THE PRE-FETCHED DF ---
                        df = term_dfs.get(term, 0) # Get the DF we fetched earlier
                        current_doc_total_score += self._calculate_score(tf, df, total_docs_for_idf) # Pass pre-fetched values

                    # Mark this term's iterator for advancement
                    terms_in_min_doc.append(term)

            # --- Update Top-K Heap ---
            if current_doc_total_score > 0:
                 # Use negative doc_id for tie-breaking (lower doc_id preferred for same score)
                 heap_entry = (current_doc_total_score, -min_doc_id)
                 if len(top_k_heap) < k:
                     heapq.heappush(top_k_heap, heap_entry)
                 # Only replace if current score is strictly greater than smallest in heap
                 elif current_doc_total_score > top_k_heap[0][0]:
                     heapq.heapreplace(top_k_heap, heap_entry)
                 # Handle tie-breaking: if scores are equal, prefer lower doc ID (higher negative value)
                 elif current_doc_total_score == top_k_heap[0][0] and -min_doc_id > top_k_heap[0][1]:
                      heapq.heapreplace(top_k_heap, heap_entry)


            # --- Advance Iterators that pointed to min_doc_id ---
            for term in terms_in_min_doc:
                try:
                    current_doc_ids[term] = next(posting_iterators[term])
                except StopIteration:
                    # This term's posting list is exhausted, remove it
                    current_doc_ids.pop(term, None)
                    posting_iterators.pop(term, None)
                    # No need to modify active_terms inside loop if iterating current_doc_ids.keys()

        # Convert heap to sorted list (highest score first)
        # Remember heap stores (score, -doc_id), so negate doc_id back
        result_list = [(score, -neg_doc_id) for score, neg_doc_id in top_k_heap]
        sorted_results = sorted(result_list, key=lambda x: x[0], reverse=True)

        return sorted_results