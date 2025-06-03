import os
import pandas as pd
import json
import datetime
import multiprocessing
import argparse


def construct_prompts(config_data, data, args):
    idx = 0
    items = []
    for name, decl_data in data.items():
        prompt = config_data["system_prompt"] + "Be sure to output your response as a Lean4 theorem wrapped in <IMPROVED>...</IMPROVED> tags, as shown in the example. Namely, only return the statment and proof of the current theorem in Lean4 code, wrapped in <IMPROVED>...</IMPROVED> tags. Do not include any other text or comments.\n\n"
        
        if args.examples != 0:
            prompt += config_data["example_prompt"] + "\n"

        if args.annotation:
            prompt += config_data["annotation_prompt"] + "\n"

        if args.context != 0:
            prompt += config_data["context_prompt"] + "\n"

        if args.rag != 0:
            prompt += config_data["rag_prompt"] + "\n"

        prompt += "\n"

        if args.examples != 0:
            
            with open(os.path.join(config_data["example_dir"], f"{config_data["metric"]}.json"), "r") as f:
                examples_data = json.load(f)
            
            prompt += f"<EXAMPLES>\n\n"
            for example in examples_data[: min(args.examples,len(examples_data))]:
                try:
                    ex_prompt = "<EXAMPLE>\n\n"
                    if args.context:
                        ex_prompt += f"<CONTEXT>\n"
                        for context in example["context"]:
                            ex_prompt += f"<ITEM>\n--name={context['name']}\n--type={context['context_item_type']}\n{context['content']}\n</ITEM>\n"
                        ex_prompt += f"</CONTEXT>\n\n"
                    if args.rag != 0:
                        ex_prompt += f"<RAG>\n"
                        for rag in example["rag"][: args.rag]:
                            ex_prompt += (
                                f"<DOC>\n--src={rag['src']}\n{rag['content']}\n</DOC>\n"
                            )
                        ex_prompt += f"</RAG>\n\n"
                    if args.annotation:
                        ex_prompt += (
                            f"<ANNOTATION>\n{example['annotation']}\n</ANNOTATION>\n\n"
                        )
                    ex_prompt += f"<CURRENT>\n{example['current']}\n</CURRENT>\n\n"
                    ex_prompt += f"<IMPROVED>\n{example['improved']}\n</IMPROVED>\n\n"
                    ex_prompt += f"</EXAMPLE>\n\n"
                    prompt += ex_prompt
                except:
                    pass
            prompt += f"</EXAMPLES>\n\n"

        if args.context != 0:
            prompt += f"<CONTEXT>\n"
            for context in decl_data["context"][: min(args.context,len(decl_data["context"]))]:
                prompt += f"<ITEM>\n--name={context['name']}\n--type={context['context_item_type']}\n{context['content']}\n</ITEM>\n"
            prompt += f"</CONTEXT>\n\n"

        if args.rag != 0:
            prompt += f"<RAG>\n"
            for rag in decl_data["rag"][: min(args.rag,len(decl_data["rag"]))]:
                prompt += f"<DOC>\n--src={rag['src']}\n{rag['content']}\n</DOC>\n"
            prompt += f"</RAG>\n\n"

        if args.annotation:
            prompt += f"<ANNOTATION>\n{decl_data['annotation']}\n</ANNOTATION>\n\n"

        prompt += f"\n<CURRENT>\n{decl_data['current_sorry']}\n</CURRENT>\n\n"
        prompt += "<IMPROVED>"

        output = f"{decl_data['current']}\n\n</IMPROVED>"


        data = {
            "decl": name,
            "decl_idx": idx,
            "raw_prompt": prompt,
            "output": output
        }
        items.append(data)
        idx += 1
    return items


from pathlib import Path

def get_custom_stem(file_path: str) -> str:
    p = Path(file_path)
    parts = p.parts

    if len(parts) >= 4 and parts[2] == "Mathlib":
        return str(Path(*parts[:4]))
    else:
        return str(Path(*parts[:3]))

def main(args):    
    with open(args.dataset_path, "r") as f:
        all = json.load(f)
        dataset = all[args.split]
    files_to_process = []
    for repo in dataset.keys():
        files_to_process = files_to_process + dataset[repo]

    prompt_root = os.path.join(args.prompts_dir, args.metric)
    config_path = os.path.join(prompt_root, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, "r") as f:
        config_data = json.load(f)
    
    # df = pd.DataFrame(columns=["file_path", "decl", "decl_idx", "raw_prompt", "output"])
    df = []
    data = {}
    for file in files_to_process:
        file_path = os.path.join(prompt_root, file.replace(".lean", ".json"))
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                data_raw = json.load(f)
                prompt_data = construct_prompts(config_data, data_raw, args)
            # print(f"Processing {file_path} with {len(prompt_data)} prompts")
            stem = get_custom_stem(file_path)
            if stem not in data:
                data[stem] = len(prompt_data)
            else:
                data[stem] += len(prompt_data)
            
            
            for item in prompt_data:
                df.append({"instruction": item["raw_prompt"], "output": item["output"]})
            print(f"Processed {file_path} with {len(prompt_data)} prompts")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create output filename with timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(args.output_dir, f"{args.metric}_{args.split}_{timestamp}.jsonl")
    
    # Write DataFrame to JSONL file
    with open(output_file, 'w') as f:
        for item in df:
            f.write(json.dumps(item) + '\n')
    
    print(f"Successfully wrote {len(df)} records to {output_file}")
    
    # # Print statistics about data distribution
    # print(f"Data distribution across {len(data)} files:")
    # for stem, count in sorted(data.items(), key=lambda x: x[1], reverse=True):
    #     print(f"  {stem}: {count} prompts")
                
    
    
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate prompts for ImProver")
    parser.add_argument("metric", type=str, help="Metric to use for evaluation")
    parser.add_argument("dataset_path", type=str, help="Path to dataset JSON file")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--prompts_dir",
        type=str,
        default="prompts/",
        help="Directory of prompt data (default: prompts/)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/",
        help="Directory to output data (default: data/)",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of CPUs to use (default: all available)",
    )


    parser.add_argument(
        "--annotation", type=bool, default=False, help="Annotation? (default: False)"
    )
    parser.add_argument(
        "--context",
        type=int,
        default=0,
        help="Number of context retrievals (default: 0)",
    )
    parser.add_argument(
        "--rag", type=int, default=0, help="Number of RAG retrievals (default: 0)"
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=0,
        help="Number of few-shot example retrievals (default: 0)",
    )

    args = parser.parse_args()

    main(args)
