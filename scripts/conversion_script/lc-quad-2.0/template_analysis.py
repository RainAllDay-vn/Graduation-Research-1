import json
import os
from typing import Dict, Any

# Configurations
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

DATASET_PATH = os.path.join(BASE_PATH, 'dataset', 'lc-quad-2.0')
TRAIN_FILE = os.path.join(DATASET_PATH, 'train.json')
TEST_FILE = os.path.join(DATASET_PATH, 'test.json')
TEMP_OUTPUT_DIR = os.path.join(DATASET_PATH, 'temp')

def analyze_templates(file_path: str, dataset_name: str) -> Dict[str, Any]:
    """Extracts unique templates and counts from a dataset file."""
    print(f"Analyzing {dataset_name} templates from {os.path.basename(file_path)}...")
    if not os.path.exists(file_path):
        print(f"  [!] Error: File not found: {file_path}")
        return {"count": 0, "distribution": {}}

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Mapping of template (as JSON string for hashability) to frequency
    distribution = {}
    total_entries = len(data)

    for entry in data:
        t = entry.get('template')
        if t is None:
            continue
        
        # We use a sorted JSON string to handle both strings and lists/dicts consistently
        key = json.dumps(t, sort_keys=True)
        distribution[key] = distribution.get(key, 0) + 1

    unique_count = len(distribution)
    
    # Create a nice report structure
    report = {
        "dataset": dataset_name,
        "total_entries": total_entries,
        "unique_templates_count": unique_count,
        # Convert back from JSON strings for the final JSON dump
        "templates": [
            {"template": json.loads(k), "frequency": v} 
            for k, v in sorted(distribution.items(), key=lambda x: x[1], reverse=True)
        ]
    }
    
    return report

def save_report(report: Dict[str, Any], filename: str):
    """Saves the template analysis report to the temp folder."""
    output_path = os.path.join(TEMP_OUTPUT_DIR, filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    print(f"  Saved template info to {output_path}")

def main():
    # 1. Ensure output directory exists
    if not os.path.exists(TEMP_OUTPUT_DIR):
        os.makedirs(TEMP_OUTPUT_DIR)
        print(f"Created folder: {TEMP_OUTPUT_DIR}")

    # 2. Analyze datasets
    train_report = analyze_templates(TRAIN_FILE, "Train")
    test_report = analyze_templates(TEST_FILE, "Test")

    # 3. Save findings
    save_report(train_report, "train_templates_info.json")
    save_report(test_report, "test_templates_info.json")

    # 4. Global summary
    all_train_keys = {json.dumps(t['template'], sort_keys=True) for t in train_report['templates']}
    all_test_keys = {json.dumps(t['template'], sort_keys=True) for t in test_report['templates']}
    
    union_count = len(all_train_keys.union(all_test_keys))
    intersect_count = len(all_train_keys.intersection(all_test_keys))

    summary = {
        "summary": {
            "train_unique_templates": train_report['unique_templates_count'],
            "test_unique_templates": test_report['unique_templates_count'],
            "combined_unique_templates": union_count,
            "overlapping_templates": intersect_count,
            "train_only": len(all_train_keys - all_test_keys),
            "test_only": len(all_test_keys - all_train_keys)
        }
    }
    
    print("\n--- Summary ---")
    print(f"Unique Templates (Train): {summary['summary']['train_unique_templates']}")
    print(f"Unique Templates (Test):  {summary['summary']['test_unique_templates']}")
    print(f"Combined Unique:          {summary['summary']['combined_unique_templates']}")

if __name__ == "__main__":
    main()
