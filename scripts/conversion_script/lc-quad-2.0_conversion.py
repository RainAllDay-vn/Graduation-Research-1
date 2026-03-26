import requests
import json
import time
import os
from typing import List, Dict, Tuple, Set

# Constants - Using raw strings for Windows paths
BASE_PATH = r'd:\Graduation-Research-1'
INPUT_FILE = os.path.join(BASE_PATH, 'dataset', 'lc-quad-2.0', 'raw', 'entities_covered.txt')
ENTITIES_COVERED_JSON = os.path.join(BASE_PATH, 'dataset', 'lc-quad-2.0', 'entities_covered.json')
ENTITY_LABELS_JSON = os.path.join(BASE_PATH, 'dataset', 'lc-quad-2.0', 'entity_labels.json')
MISSING_LABELS_JSON = os.path.join(BASE_PATH, 'dataset', 'lc-quad-2.0', 'missing_labels.json')

SPARQL_URL = "https://query.wikidata.org/sparql"
HEADERS = {
    'User-Agent': 'GraduationProjectEntityMapper/1.0 (vlong@example.com) requests-python',
    'Accept': 'application/sparql-results+json'
}

def load_unique_ids(file_path: str) -> List[str]:
    """Reads IDs from a text file, handles BOM, and returns a sorted list of unique non-empty IDs."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
        
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        # Filter out empty strings and whitespace
        ids = [line.strip() for line in f if line.strip()]
    
    return sorted(list(set(ids)))

def fetch_labels_with_errors(qids: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Fetches labels for a list of QID/PID strings from Wikidata.
    Returns (labels_dict, errors_dict)
    """
    if not qids:
        return {}, {}
    
    qid_set = set(qids)
    formatted_ids = " ".join([f"wd:{q}" if not q.startswith('wd:') else q for q in qids])
    
    query = f"""
    SELECT ?item ?itemLabel WHERE {{
      VALUES ?item {{ {formatted_ids} }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """
    
    labels = {}
    batch_errors = {}
    
    try:
        response = requests.post(
            SPARQL_URL, 
            data={'query': query, 'format': 'json'}, 
            headers=HEADERS,
            timeout=120 # Higher timeout for larger batches (2000)
        )
        response.raise_for_status()
        data = response.json()
        
        found_qids = set()
        for binding in data['results']['bindings']:
            uri = binding['item']['value']
            qid = uri.split('/')[-1]
            label = binding['itemLabel']['value']
            
            # Verify the QID belongs to this batch to avoid phantom entries
            if qid in qid_set:
                if label != qid:
                    labels[qid] = label
                    found_qids.add(qid)
                else:
                    batch_errors[qid] = "No English label found in Wikidata"
                    found_qids.add(qid)

        # Catch IDs that weren't returned by Wikidata at all
        for q in qids:
            if q not in found_qids:
                batch_errors[q] = "Entity not found/returned by Wikidata"
                
        return labels, batch_errors

    except Exception as e:
        error_msg = f"Batch fetch error: {str(e)}"
        print(f"  [!] {error_msg}")
        return {}, {q: error_msg for q in qids}

def save_json_file(data, file_path: str, description: str):
    """Saves data to a JSON file and prints status."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Saved {len(data)} entries to {description} ({os.path.basename(file_path)})")

def process_all_labels(all_ids: List[str], batch_size: int = 2000) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Iterates through all IDs in batches and fetches labels."""
    entity_labels = {}
    missing_labels = {}
    total_count = len(all_ids)
    num_batches = (total_count + batch_size - 1) // batch_size
    
    print(f"Starting extraction for {total_count} unique IDs in batches of {batch_size}...")
    
    for i in range(0, total_count, batch_size):
        batch = all_ids[i:i + batch_size]
        batch_num = i // batch_size + 1
        print(f"  Processing batch {batch_num}/{num_batches}...")
        
        labels, errors = fetch_labels_with_errors(batch)
        entity_labels.update(labels)
        missing_labels.update(errors)
        
        # Rate limiting: Sleep slightly between batches
        time.sleep(1.5)
        
    return entity_labels, missing_labels

def perform_final_verification(total_unique: int, entity_labels: dict, missing_labels: dict):
    """Checks if the sum of results matches the original ID count."""
    total_processed = len(entity_labels) + len(missing_labels)
    intersection = set(entity_labels.keys()) & set(missing_labels.keys())
    
    print("\n--- Final Verification Summary ---")
    print(f"Original unique ID count: {total_unique}")
    print(f"Labeled entities found:   {len(entity_labels)}")
    print(f"Missing/No label count:   {len(missing_labels)}")
    print(f"Total processed:          {total_processed}")
    
    if intersection:
        print(f"WARNING: {len(intersection)} IDs appear in BOTH files!")
    
    if total_processed == total_unique:
        print("✅ SUCCESS: All IDs accounted for!")
    else:
        diff = total_unique - total_processed
        print(f"❌ DISCREPANCY: {diff} IDs are missing from the mapping files.")

def main():
    try:
        # 1. Load data
        all_ids = load_unique_ids(INPUT_FILE)
        save_json_file(all_ids, ENTITIES_COVERED_JSON, "list of covered entities")

        # 2. Process labels
        batch_size = 2000 # Increased as requested
        labels, missing = process_all_labels(all_ids, batch_size)

        # 3. Save results
        save_json_file(labels, ENTITY_LABELS_JSON, "entity labels mapping")
        save_json_file(missing, MISSING_LABELS_JSON, "missing labels reason mapping")

        # 4. Verify
        perform_final_verification(len(all_ids), labels, missing)

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")

if __name__ == "__main__":
    main()
