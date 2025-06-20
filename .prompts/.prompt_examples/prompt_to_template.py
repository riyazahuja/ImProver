import json
import re
import sys


input_file = sys.argv[1] if len(sys.argv) > 1 else "_ALL.txt"
output_file = sys.argv[2] if len(sys.argv) > 2 else "output.json"

# Read the contents of _ALL.txt
try:
    with open(input_file, "r", encoding="utf-8") as file:
        content = file.read()
except FileNotFoundError:
    print(f"Error: {input_file} file not found.")
    exit(1)
except Exception as e:
    print(f"Error reading file: {e}")
    exit(1)

# Split the file into examples
examples = content.split("<EXAMPLE>")
examples = [ex for ex in examples if ex.strip()]  # Remove empty examples

parsed_examples = []

for example in examples:
    example_data = {}

    # Parse context items
    context_match = re.search(r"<CONTEXT>(.*?)</CONTEXT>", example, re.DOTALL)
    if context_match:
        context_text = context_match.group(1)
        items = re.findall(r"<ITEM>(.*?)</ITEM>", context_text, re.DOTALL)

        context_items = []
        for item in items:
            name_match = re.search(r"--name=(.*?)(?:\n|$)", item)
            type_match = re.search(r"--context_item_type=(.*?)(?:\n|$)", item)

            name = name_match.group(1) if name_match else ""
            item_type = type_match.group(1) if type_match else ""

            # Get contents by removing the metadata lines
            contents = re.sub(r"--.*?(?:\n|$)", "", item, count=2).strip()

            context_items.append(
                {"name": name, "context_item_type": item_type, "contents": contents}
            )

        example_data["context"] = context_items

    # Parse retrieved documents
    retrieved_match = re.search(r"<RETRIEVED>(.*?)</RETRIEVED>", example, re.DOTALL)
    if retrieved_match:
        retrieved_text = retrieved_match.group(1)
        docs = re.findall(r"<DOC>(.*?)</DOC>", retrieved_text, re.DOTALL)

        retrieved_docs = []
        for doc in docs:
            src_match = re.search(r"--src:\s*(.*?)(?:\n|$)", doc)
            src = src_match.group(1) if src_match else ""

            # Get content by removing the src line
            content = re.sub(r"--src:.*?(?:\n|$)", "", doc, count=1).strip()

            retrieved_docs.append({"src": src, "content": content})

        example_data["retrieved"] = retrieved_docs

    # Parse annotation
    annotation_match = re.search(r"<ANNOTATION>(.*?)</ANNOTATION>", example, re.DOTALL)
    if annotation_match:
        example_data["annotated"] = annotation_match.group(1).strip()

    # Parse current and improved
    current_match = re.search(r"<CURRENT>(.*?)</CURRENT>", example, re.DOTALL)
    if current_match:
        # This is just for reference, not part of output
        example_data["current"] = current_match.group(1).strip()

    improved_match = re.search(r"<IMPROVED>(.*?)</IMPROVED>", example, re.DOTALL)
    if improved_match:
        example_data["improved"] = improved_match.group(1).strip()

    if example_data:
        parsed_examples.append(example_data)

# Write the data to a JSON file
try:
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(parsed_examples, json_file, ensure_ascii=False, indent=4)
    print("Successfully created output.json")
except Exception as e:
    print(f"Error writing to JSON file: {e}")
