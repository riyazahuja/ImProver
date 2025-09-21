import json

def check_contamination(json_file_path):
    """
    Check for contamination between train and test splits by finding files that appear in both.
    Returns the count and list of contaminated files, and a cleaned version of the data.
    """
    # Load the JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Extract train file names (flatten all categories)
    train_files = set()
    for category, files in data['train'].items():
        train_files.update(files)
    
    # Extract test file names (from all test entries)
    test_files = set()
    for category, test_entries in data['test'].items():
        for entry in test_entries:
            test_files.add(entry['file'])
    
    # Find contaminated files (intersection)
    contaminated_files = train_files.intersection(test_files)
    
    print(f"Contamination found: {len(contaminated_files)} files appear in both train and test splits")
    print("\nContaminated files:")
    for file in sorted(contaminated_files):
        print(f"  - {file}")
    
    # Create cleaned data by removing contaminated files from train split
    cleaned_data = {
        'train': {},
        'test': data['test'].copy()  # Keep test split unchanged
    }
    
    # Remove contaminated files from each train category
    for category, files in data['train'].items():
        cleaned_files = [f for f in files if f not in contaminated_files]
        cleaned_data['train'][category] = cleaned_files
        
        removed_count = len(files) - len(cleaned_files)
        if removed_count > 0:
            print(f"\nRemoved {removed_count} contaminated files from train/{category}")
    
    return contaminated_files, cleaned_data

def main():
    print("starting")
    json_file_path = '/home/riyaza/eval_improver/improver/data/final_dataset_fixed.json'
    
    contaminated_files, cleaned_data = check_contamination(json_file_path)
    
    # Save cleaned data
    output_path = '/home/riyaza/eval_improver/improver/data/final_dataset_decontaminated.json'
    with open(output_path, 'w') as f:
        json.dump(cleaned_data, f, indent=2)
    
    print(f"\nCleaned dataset saved to: {output_path}")
    print(f"Total contaminated files removed: {len(contaminated_files)}")

if __name__ == "__main__":
    main()