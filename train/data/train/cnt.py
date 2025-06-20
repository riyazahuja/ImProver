import json


def cnt(data):
    if type(data) == list:
        return len(data)
    elif type(data) == dict:
        sum = 0
        for key in data:
            sum += cnt(data[key])
        return sum
    elif type(data) == str:
        return 1
    else:
        return 1


def load_and_count(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    for k in data.keys():
        print(k)
    count = cnt(data)
    print(f"Count: {count}")
    return count


# Example usage
if __name__ == "__main__":
    # Replace with your JSON file path
    file_path = "scripts/data/train/train_set.json"
    load_and_count(file_path)
