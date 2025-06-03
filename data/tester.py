import os
import random

def get_random_files(directory_path, count=20):
    """
    Return a list of random file paths from the specified directory and its subdirectories.
    
    Args:
        directory_path: Path to the root directory
        count: Number of random files to return
        
    Returns:
        List of random file paths
    """
    all_files = []
    
    # Walk through the directory and collect all files
    for root, _, files in os.walk(directory_path):
        for file in files:
            all_files.append(os.path.join(root, file))
    
    # If there are fewer files than requested, return all of them
    if len(all_files) <= count:
        return all_files
    
    # Otherwise, return a random sample
    return random.sample(all_files, count)

if __name__ == "__main__":
    directory = ".lake/packages/mathlib/Mathlib/Analysis/"
    
    try:
        random_files = get_random_files(directory)
        print(f"Found {len(random_files)} random files:")
        for file in random_files:
            print(file)
    except Exception as e:
        print(f"Error: {e}")