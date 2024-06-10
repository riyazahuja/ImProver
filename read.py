import json

with open('.cache/Tests/TacticPrediction/«Tests»_Basic.jsonl', 'r') as json_file:
    json_list = list(json_file)



for json_str in json_list:
    result = json.loads(json_str)

    text = ""
    for k,v in result.items():
        if k != 'srcUpToTactic' and k != 'srcUpToDecl':
            text = text + f'{k} : {v}\n'

    print(f"\n\nLINE!\n\n\n{text}\n\n")