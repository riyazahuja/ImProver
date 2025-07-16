import json

def fix_ds():
    # Load the JSON file
    with open('/home/riyaza/eval_improver/improver/train/data/train/final_dataset.json', 'r') as f:
        data = json.load(f)
    
    # Process each project in the test set
    for project in data['test']:
        file_dict = {}
        
        # Group theorems by file name
        for item in data['test'][project]:
            file_name = item['file']
            theorems = item['theorems']
            
            if file_name in file_dict:
                # Combine theorems and deduplicate
                file_dict[file_name].extend(theorems)
            else:
                file_dict[file_name] = theorems.copy()
        
        # Deduplicate theorems for each file
        for file_name in file_dict:
            file_dict[file_name] = list(set(file_dict[file_name]))
        
        # Create new list with unique file names
        new_test_list = []
        for file_name, theorems in file_dict.items():
            new_test_list.append({
                'file': file_name,
                'theorems': theorems
            })
        
        # Replace the old test data with the new one
        data['test'][project] = new_test_list
    
    # Save the modified dataset
    with open('/home/riyaza/eval_improver/improver/train/data/train/final_dataset_fixed.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print("Dataset has been processed and saved as 'final_dataset_fixed.json'")

# Run the function
fix_ds()