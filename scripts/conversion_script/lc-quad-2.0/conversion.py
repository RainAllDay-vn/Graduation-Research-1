import sys
import os
import requests
import json
import time
import re
from typing import List, Dict, Tuple, Set

# Add the scripts directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
if scripts_dir not in sys.path:
    sys.path.append(scripts_dir)

import utils

# Constants

# Discover project root (3 levels up from scripts/conversion_script/lc-quad-2.0/)
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Output labels into the dataset's main folder
ENTITIES_COVERED_JSON = os.path.join(BASE_PATH, 'dataset', 'lc-quad-2.0', 'temp', 'entities_covered.json')
ENTITY_LABELS_JSON = os.path.join(BASE_PATH, 'dataset', 'lc-quad-2.0', 'temp', 'entity_labels.json')
MISSING_LABELS_JSON = os.path.join(BASE_PATH, 'dataset', 'lc-quad-2.0', 'temp', 'missing_labels.json')

TRAIN_RAW = os.path.join(BASE_PATH, 'dataset', 'lc-quad-2.0', 'raw', 'train.json')
TEST_RAW = os.path.join(BASE_PATH, 'dataset', 'lc-quad-2.0', 'raw', 'test.json')
TRAIN_FILTERED = os.path.join(BASE_PATH, 'dataset', 'lc-quad-2.0', 'temp', 'train_filtered.json')
TEST_FILTERED = os.path.join(BASE_PATH, 'dataset', 'lc-quad-2.0', 'temp', 'test_filtered.json')
TRAIN_CYPHER = os.path.join(BASE_PATH, 'dataset', 'lc-quad-2.0', 'train.json')
TEST_CYPHER = os.path.join(BASE_PATH, 'dataset', 'lc-quad-2.0', 'test.json')
CONVERSION_ERRORS_LOG = os.path.join(BASE_PATH, 'dataset', 'lc-quad-2.0', 'temp', 'conversion_errors.log')

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
    """Handles extracting all IDs from original queries and fetching labels."""
    print("\n--- PHASE 1: ID Extraction & Label Fetching ---")
    if os.path.exists(ENTITY_LABELS_JSON) and os.path.exists(MISSING_LABELS_JSON):
        print("Skipping Phase 1 Calculation: Output files already exist.")
        # Optional: Load anyway if we need them in main, but they are loaded from files if skipped
        with open(ENTITY_LABELS_JSON, 'r', encoding='utf-8') as f: labels = json.load(f)
        with open(MISSING_LABELS_JSON, 'r', encoding='utf-8') as f: missing = json.load(f)
        return labels, missing

    all_ids_set = set()
    for dataset_path in [TRAIN_RAW, TEST_RAW]:
        if not os.path.exists(dataset_path):
            print(f"  [!] Warning: {dataset_path} not found during extraction.")
            continue
        print(f"  Scanning {os.path.basename(dataset_path)}...")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for entry in data:
                sparql = entry.get('sparql_wikidata', "")
                all_ids_set.update(extract_ids_from_sparql(sparql))
    
    all_ids = sorted(list(all_ids_set))
    save_json(all_ids, ENTITIES_COVERED_JSON, "extracted unique IDs")
    
    entity_labels, missing_labels = {}, {}
    batch_size = 2000
    num_batches = (len(all_ids) + batch_size - 1) // batch_size
    for i in range(0, len(all_ids), batch_size):
        batch = all_ids[i:i + batch_size]
        print(f"  Batch {i // batch_size + 1}/{num_batches}...")
        l, e = fetch_labels_with_errors(batch)
        entity_labels.update(l)
        missing_labels.update(e)
        time.sleep(2)
        
    save_json(entity_labels, ENTITY_LABELS_JSON, "entity labels")
    save_json(missing_labels, MISSING_LABELS_JSON, "missing labels")
    return entity_labels, missing_labels

def extract_ids_from_sparql(sparql_query: str) -> Set[str]:
    """Extracts all QIDs and PIDs from a SPARQL query string."""
    return set(re.findall(r'[QP][0-9]+', sparql_query))

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
        ids = extract_ids_from_sparql(sparql)
        
        is_valid = True
        failed_id = None
        reason = None
        
        for id in ids:
            if id in missing:
                is_valid = False
                failed_id = id
                reason = missing[id]
                break
            if id not in labels:
                # If it's not in labels and not in missing, it was never fetched
                is_valid = False
                failed_id = id
                reason = "ID not in fetched labels"
                break
        
        if is_valid:
            filtered_data.append(entry)
            stats["kept"] += 1
        else:
            stats["filtered"] += 1
            stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1

    save_json(filtered_data, output_path, f"filtered {os.path.basename(input_path)}")
    return stats

# --- PHASE 3: SPARQL to Cypher Conversion ---

def convert_t1(sparql: str, labels: Dict[str, str]) -> str:
    """Template 1: E REF ?F"""
    match = re.search(r'wd:(Q\d+) wdt:(P\d+)', sparql)
    try:
        assert match is not None
        entity = match.group(1)
        entity = labels[entity]
        relation = match.group(2)
        relation = labels[relation]
        relation = utils.to_screaming_snake_case(relation)

        return f"MATCH (e:Entity {{name: '{entity}'}})-[:{relation}]->(f:Entity) RETURN f"
    except Exception as e:
        raise ValueError(f"Failed parse the query: {e}")
    return ""

def convert_t2(sparql: str, labels: Dict[str, str]) -> str:
    """Template 2: (E pred F) prop ?value"""
    return ""

def convert_t3(sparql: str, labels: Dict[str, str]) -> str:
    """Template 3: (E pred ?Obj ) prop value"""
    return ""

def convert_t4(sparql: str, labels: Dict[str, str]) -> str:
    """Template 4: E REF ?F . ?F RFG G"""
    return ""

def convert_t5(sparql: str, labels: Dict[str, str]) -> str:
    """Template 5: <?S P O ; ?S InstanceOf Type>"""
    return ""

def convert_t6(sparql: str, labels: Dict[str, str]) -> str:
    """Template 6: E REF xF . xF RFG ?G"""
    return ""

def convert_t7(sparql: str, labels: Dict[str, str]) -> str:
    """Template 7: <S P ?O ; ?O instanceOf Type>"""
    return ""

def convert_t8(sparql: str, labels: Dict[str, str]) -> str:
    """Template 8: C RCD xD . xD RDE ?E"""
    return ""

def convert_t9(sparql: str, labels: Dict[str, str]) -> str:
    """Template 9: ASK ?sbj ?pred ?obj filter ?obj = num"""
    return ""

def convert_t10(sparql: str, labels: Dict[str, str]) -> str:
    """Template 10: []"""
    return ""

def convert_t11(sparql: str, labels: Dict[str, str]) -> str:
    """Template 11: <?S P O ; ?S instanceOf Type ; starts with character >"""
    return ""

def convert_t12(sparql: str, labels: Dict[str, str]) -> str:
    """Template 12: <?S P O ; ?S instanceOf Type ; contains word >"""
    return ""

def convert_t13(sparql: str, labels: Dict[str, str]) -> str:
    """Template 13: Count ent (ent-pred-obj)"""
    return ""

def convert_t14(sparql: str, labels: Dict[str, str]) -> str:
    """Template 14: select where (ent-pred-obj1 . ent-pred-obj2)"""
    return ""

def convert_t15(sparql: str, labels: Dict[str, str]) -> str:
    """Template 15: ?D RDE E"""
    return ""

def convert_t16(sparql: str, labels: Dict[str, str]) -> str:
    """Template 16: Count Obj (ent-pred-obj)"""
    return ""

def convert_t17(sparql: str, labels: Dict[str, str]) -> str:
    """Template 17: ?E is_a Type, ?E pred Obj  value. MAX/MIN (value)"""
    return ""

def convert_t18(sparql: str, labels: Dict[str, str]) -> str:
    """Template 18: ?E is_a Type. ?E pred Obj. ?E-secondClause value. MIN (value)"""
    return ""

def convert_t19(sparql: str, labels: Dict[str, str]) -> str:
    """Template 19: ?E is_a Type. ?E pred Obj. ?E-secondClause value. MAX (value)"""
    return ""

def convert_t20(sparql: str, labels: Dict[str, str]) -> str:
    """Template 20: Ask (ent-pred-obj)"""
    return ""

def convert_t21(sparql: str, labels: Dict[str, str]) -> str:
    """Template 21: Ask (ent-pred-obj1 . ent-pred-obj2)"""
    return ""

def convert_t22(sparql: str, labels: Dict[str, str]) -> str:
    """Template 22: Ask (ent-pred-obj`)"""
    return ""

def convert_t23(sparql: str, labels: Dict[str, str]) -> str:
    """Template 23: Ask (ent-pred-obj1` . ent-pred-obj2)"""
    return ""

def convert_t24(sparql: str, labels: Dict[str, str]) -> str:
    """Template 24: Ask (ent-pred-obj1 . ent-pred-obj2`)"""
    return ""

def convert_t25(sparql: str, labels: Dict[str, str]) -> str:
    """Template 25: Ask (ent`-pred-obj1 . ent`-pred-obj2)"""
    return ""

def convert_t26(sparql: str, labels: Dict[str, str]) -> str:
    """Template 26: Ask (ent`-pred-obj)"""
    return ""

# Mapping from template string to conversion method
TEMPLATE_CONVERTERS = {
    "E REF ?F": convert_t1,
    "(E pred F) prop ?value": convert_t2,
    "(E pred ?Obj ) prop value": convert_t3,
    "E REF ?F . ?F RFG G": convert_t4,
    "<?S P O ; ?S InstanceOf Type>": convert_t5,
    "E REF xF . xF RFG ?G": convert_t6,
    " <S P ?O ; ?O instanceOf Type>": convert_t7,
    "C RCD xD . xD RDE ?E": convert_t8,
    "ASK ?sbj ?pred ?obj filter ?obj = num": convert_t9,
    "[]": convert_t10,
    " <?S P O ; ?S instanceOf Type ; starts with character >": convert_t11,
    " <?S P O ; ?S instanceOf Type ; contains word >": convert_t12,
    "Count ent (ent-pred-obj)": convert_t13,
    "select where (ent-pred-obj1 . ent-pred-obj2)": convert_t14,
    "?D RDE E": convert_t15,
    "Count Obj (ent-pred-obj)": convert_t16,
    "?E is_a Type, ?E pred Obj  value. MAX/MIN (value)": convert_t17,
    "?E is_a Type. ?E pred Obj. ?E-secondClause value. MIN (value)": convert_t18,
    "?E is_a Type. ?E pred Obj. ?E-secondClause value. MAX (value)": convert_t19,
    "Ask (ent-pred-obj)": convert_t20,
    "Ask (ent-pred-obj1 . ent-pred-obj2)": convert_t21,
    "Ask (ent-pred-obj`)": convert_t22,
    "Ask (ent-pred-obj1` . ent-pred-obj2)": convert_t23,
    "Ask (ent-pred-obj1 . ent-pred-obj2`)": convert_t24,
    "Ask (ent`-pred-obj1 . ent`-pred-obj2)": convert_t25,
    "Ask (ent`-pred-obj)": convert_t26
}

def process_phase_three(input_path: str, output_path: str, labels: Dict[str, str]) -> dict:
    """Converts SPARQL to Cypher for a filtered dataset."""
    print(f"Converting {os.path.basename(input_path)} to Cypher...")
    if not os.path.exists(input_path):
        print(f"  [!] Warning: {input_path} not found.")
        return {"total": 0, "processed": 0}

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    output_data = []
    stats = {"total": len(data), "processed": 0, "unsupported": 0, "failed": 0}

    for entry in data:
        template = entry.get('template')
        # Handle lists/dicts if template is not a string
        if isinstance(template, list) or isinstance(template, dict):
            template_key = json.dumps(template, sort_keys=True)
        else:
            template_key = template

        converter = TEMPLATE_CONVERTERS.get(template_key)
        if converter:
            sparql = entry.get('sparql_wikidata')
            entry['original_question'] = entry['question']
            entry['question'] = entry['paraphrased_question'] if entry['paraphrased_question'] else entry['question']
            try:
                cypher = converter(sparql, labels)
                if cypher:
                    entry['cypher_query'] = cypher
                    stats["processed"] += 1
                else:
                    stats["unsupported"] += 1
                output_data.append(entry)
            except Exception as e:
                stats["failed"] += 1
                with open(CONVERSION_ERRORS_LOG, 'a', encoding='utf-8') as log_f:
                    log_f.write(f"UID: {entry.get('uid')} | Template: {template_key} | Error: {str(e)}\n")
        else:
            stats["unsupported"] += 1

    save_json(output_data, output_path, os.path.basename(output_path))
    return stats

def main():
    try:
        # Clear/Create log file
        with open(CONVERSION_ERRORS_LOG, 'w', encoding='utf-8') as f:
            pass

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

        # Phase 3
        print("\n--- PHASE 3: SPARQL to Cypher Conversion ---")
        train_cypher_stats = process_phase_three(TRAIN_FILTERED, TRAIN_CYPHER, labels)
        test_cypher_stats = process_phase_three(TEST_FILTERED, TEST_CYPHER, labels)

        print("\n--- Conversion Summary ---")
        for name, s in [("Train", train_cypher_stats), ("Test", test_cypher_stats)]:
            print(f"{name} Dataset:")
            print(f"  Available for conversion: {s['total']}")
            print(f"  Successfully processed:   {s['processed']}")
            print(f"  Unsupported/Placeholders: {s['unsupported']}")
            print(f"  Failed with error:        {s.get('failed', 0)}")

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")

if __name__ == "__main__":
    main()
