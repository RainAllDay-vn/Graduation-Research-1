import requests
import json
import time
import os
import re
from typing import List, Dict, Tuple, Set

# Constants
BASE_PATH = r'd:\Graduation-Research-1'
INPUT_FILE = os.path.join(BASE_PATH, 'dataset', 'lc-quad-2.0', 'raw', 'entities_covered.txt')
ENTITIES_COVERED_JSON = os.path.join(BASE_PATH, 'dataset', 'temp', 'lc-quad-2.0', 'entities_covered.json')
ENTITY_LABELS_JSON = os.path.join(BASE_PATH, 'dataset', 'temp', 'lc-quad-2.0', 'entity_labels.json')
MISSING_LABELS_JSON = os.path.join(BASE_PATH, 'dataset', 'temp', 'lc-quad-2.0', 'missing_labels.json')

TRAIN_RAW = os.path.join(BASE_PATH, 'dataset', 'lc-quad-2.0', 'raw', 'train.json')
TEST_RAW = os.path.join(BASE_PATH, 'dataset', 'lc-quad-2.0', 'raw', 'test.json')
TRAIN_FILTERED = os.path.join(BASE_PATH, 'dataset', 'lc-quad-2.0', 'train.json')
TEST_FILTERED = os.path.join(BASE_PATH, 'dataset', 'lc-quad-2.0', 'test.json')

SPARQL_URL = "https://query.wikidata.org/sparql"
HEADERS = {
    'User-Agent': 'GraduationProjectEntityMapper/1.0 (vlong@example.com) requests-python',
    'Accept': 'application/sparql-results+json'
}

def load_unique_ids(file_path: str) -> List[str]:
    print(f"Loading unique IDs from {file_path}...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        ids = [line.strip() for line in f if line.strip()]
    return sorted(list(set(ids)))

def fetch_labels_with_errors(qids: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    if not qids: return {}, {}
    qid_set = set(qids)
    formatted_ids = " ".join([f"wd:{q}" if not q.startswith('wd:') else q for q in qids])
    query = f"SELECT ?item ?itemLabel WHERE {{ VALUES ?item {{ {formatted_ids} }} SERVICE wikibase:label {{ bd:serviceParam wikibase:language 'en'. }} }}"
    try:
        response = requests.post(SPARQL_URL, data={'query': query, 'format': 'json'}, headers=HEADERS, timeout=120)
        response.raise_for_status()
        data = response.json()
        labels, batch_errors, found_qids = {}, {}, set()
        for binding in data['results']['bindings']:
            qid = binding['item']['value'].split('/')[-1]
            if qid in qid_set:
                label = binding['itemLabel']['value']
                if label != qid: labels[qid] = label
                else: batch_errors[qid] = "No English label found in Wikidata"
                found_qids.add(qid)
        for q in qids:
            if q not in found_qids: batch_errors[q] = "Entity not found/returned by Wikidata"
        return labels, batch_errors
    except Exception as e:
        print(f"  [!] Batch fetch error: {e}")
        return {}, {q: f"Batch fetch error: {str(e)}" for q in qids}

def save_json(data, file_path: str, desc: str):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Saved {len(data)} entries to {desc}")

def process_phase_one():
    """Handles fetching labels if they don't exist."""
    print("\n--- PHASE 1: Label Fetching ---")
    if os.path.exists(ENTITY_LABELS_JSON) and os.path.exists(MISSING_LABELS_JSON):
        print("Skipping Phase 1: Output files already exist.")
        with open(ENTITY_LABELS_JSON, 'r', encoding='utf-8') as f: labels = json.load(f)
        with open(MISSING_LABELS_JSON, 'r', encoding='utf-8') as f: missing = json.load(f)
        return labels, missing

    all_ids = load_unique_ids(INPUT_FILE)
    save_json(all_ids, ENTITIES_COVERED_JSON, "covered entities list")
    
    entity_labels, missing_labels = {}, {}
    batch_size = 2000
    num_batches = (len(all_ids) + batch_size - 1) // batch_size
    for i in range(0, len(all_ids), batch_size):
        batch = all_ids[i:i + batch_size]
        print(f"  Batch {i // batch_size + 1}/{num_batches}...")
        l, e = fetch_labels_with_errors(batch)
        entity_labels.update(l)
        missing_labels.update(e)
        time.sleep(1.2)
        
    save_json(entity_labels, ENTITY_LABELS_JSON, "entity labels")
    save_json(missing_labels, MISSING_LABELS_JSON, "missing labels")
    return entity_labels, missing_labels

def extract_qids(sparql_query: str) -> Set[str]:
    """Extracts all QIDs from a SPARQL query string."""
    return set(re.findall(r'Q[0-9]+', sparql_query))

def filter_dataset(input_path: str, output_path: str, labels: dict, missing: dict) -> dict:
    """Filters a dataset file based on missing entity labels."""
    print(f"Filtering {os.path.basename(input_path)}...")
    if not os.path.exists(input_path):
        print(f"  [!] Warning: {input_path} not found.")
        return {"total": 0, "kept": 0, "filtered": 0}

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    filtered_data = []
    stats = {"total": len(data), "kept": 0, "filtered": 0, "reasons": {}}
    
    for entry in data:
        sparql = entry.get('sparql_wikidata', "")
        qids = extract_qids(sparql)
        
        is_valid = True
        failed_qid = None
        reason = None
        
        for qid in qids:
            if qid in missing:
                is_valid = False
                failed_qid = qid
                reason = missing[qid]
                break
            if qid not in labels:
                # If it's not in labels and not in missing, it was never fetched (missing from entities_covered.txt)
                is_valid = False
                failed_qid = qid
                reason = "Entity not in covered list"
                break
        
        if is_valid:
            filtered_data.append(entry)
            stats["kept"] += 1
        else:
            stats["filtered"] += 1
            stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1

    save_json(filtered_data, output_path, f"filtered {os.path.basename(input_path)}")
    return stats

def main():
    try:
        # Phase 1
        labels, missing = process_phase_one()

        # Phase 2
        print("\n--- PHASE 2: Dataset Filtering ---")
        train_stats = filter_dataset(TRAIN_RAW, TRAIN_FILTERED, labels, missing)
        test_stats = filter_dataset(TEST_RAW, TEST_FILTERED, labels, missing)

        print("\n--- Filtering Summary ---")
        for name, s in [("Train", train_stats), ("Test", test_stats)]:
            print(f"{name} Dataset:")
            print(f"  Original Count: {s['total']}")
            print(f"  Kept Count:     {s['kept']}")
            print(f"  Filtered Count: {s['filtered']}")
            if s['filtered'] > 0:
                print(f"  Top reasons for filtering:")
                for r, count in sorted(s['reasons'].items(), key=lambda x: x[1], reverse=True)[:3]:
                    print(f"    - {r}: {count}")
        
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")

if __name__ == "__main__":
    main()
