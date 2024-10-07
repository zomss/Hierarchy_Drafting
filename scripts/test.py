import json
import argparse
from transformers import AutoTokenizer
import numpy as np
import os
from fastchat.model import get_conversation_template

def read_jsonl(jsonl_file):
    data = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data
            
input_file = "/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/question.jsonl"
input_data = read_jsonl(input_file)

output_file = "/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/vicuna-7b-v1.3-pld-float16.jsonl"
output_data = read_jsonl(output_file)

tokenizer = AutoTokenizer.from_pretrained("../../models/vicuna-7b-v1.3")

count = []
for input, output in zip(input_data, output_data):
    conv = get_conversation_template("vicuna")
    input_texts = input["turns"]
    output_texts = output["choices"][0]["turns"]
    try:
        for input_text, output_text in zip(input_texts, output_texts):
            conv.append_message(conv.roles[0], input_text)
            conv.append_message(conv.roles[1], output_text)
            prompt = conv.get_prompt()
            inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
            input_ids = inputs.input_ids
            count.append(input_ids.shape[-1] - 4)
    except:
        from IPython import embed; embed(); exit(0)


import numpy as np
print(np.mean(count))
print(np.std(count))
