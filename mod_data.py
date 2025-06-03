def find_unique_lines(file1_path, file2_path):
    """
    Compare two files and find lines that exist in one file but not the other.
    
    Args:
        file1_path (str): Path to the first file
        file2_path (str): Path to the second file
        
    Returns:
        tuple: Two sets containing lines unique to file1 and file2 respectively
    """
    # Read the content of both files
    with open(file1_path, 'r') as f1:
        lines1 = set(f1.readlines())
    
    with open(file2_path, 'r') as f2:
        lines2 = set(f2.readlines())
    
    # Find unique lines in each file
    unique_to_file1 = lines1 - lines2
    unique_to_file2 = lines2 - lines1
    
    return unique_to_file1, unique_to_file2

def main():
    file1_path = "/home/riyaza/eval_improver/improver/data2.txt"
    file2_path = "/home/riyaza/eval_improver/improver/eval.txt"
    
    unique_to_file1, unique_to_file2 = find_unique_lines(file1_path, file2_path)
    
    print(f"Lines unique to {file1_path}:")
    for line in sorted(unique_to_file1):
        print(f"  {line.strip()}")
    
    print(f"\nLines unique to {file2_path}:")
    for line in sorted(unique_to_file2):
        print(f"  {line.strip()}")

if __name__ == "__main__":
    main()