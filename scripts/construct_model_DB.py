from random import shuffle
import os, torch
from fastchat.model import get_conversation_template
from vllm import LLM, SamplingParams
from collections import Counter
from tqdm import tqdm
from datasets import load_dataset
import json

hf_name_mapping = {
    "vicuna-7b-v1.3": "vicuna-7b-v1.3",
    "vicuna-13b-v1.3": "vicuna-13b-v1.3",
    "Llama-2-7b" : "Llama-2-7b-chat-hf",
    "Llama-2-13b" : "Llama-2-13b-chat-hf"
}

system_prompt_mapping = {
    "default" : "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\'t know the answer to a question, please don\'t share false information."
}

import argparse

assert torch.cuda.is_available()

def parse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_dir", type=str, default="../models")
    parser.add_argument("--model", type=str, default="vicuna-7b-v1.3")

    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--dataset", type=str, default="OASST")

    parser.add_argument("--datastore_dir", type=str, default="./datastore/history")

    parser.add_argument("--system_prompt", type=str, default="default")

    parser.add_argument("--LEVEL", type=int , default=5)
    opt, _ = parser.parse_known_args()
    return opt

def main():

    opt = parse()

    if hf_name_mapping.get(opt.model, None):
        model_name = os.path.join(opt.model_dir, hf_name_mapping[opt.model])
    else:
        NotImplementedError()

    model = LLM(model=model_name)
    gen_sampling = SamplingParams(max_tokens=1024, temperature=0.0)

    ds = load_dataset("OpenAssistant/oasst1")
        
    history_path = os.path.join(opt.data_dir, opt.dataset, opt.model, opt.system_prompt)
    os.makedirs(history_path, exist_ok=True)

    if not os.path.exists(os.path.join(history_path, f"{opt.model}_history.jsonl")):
        with open(os.path.join(history_path, f"{opt.model}_history.jsonl"), 'a') as f:
            for i, d in enumerate(ds['train']):
                if d['lang'] == 'en':
                    question = d['text']
                    conv = get_conversation_template(opt.model)
                    if "Llama" in opt.model:
                        conv.set_system_message(system_message=system_prompt_mapping["default"])
                    conv.append_message(conv.roles[0], question)
                    conv.append_message(conv.roles[1], None)
                    conv.stop_str = "</s>"
                    prompt = conv.get_prompt()
                    prediction = model.generate(prompt, use_tqdm=False, sampling_params=gen_sampling)
                    prediction = prediction[0].outputs[0]
                    f.write(json.dumps({
                        "question_id": i,
                        "tokens": prediction.token_ids,
                        "text": prediction.text
                    }) + "\n")
                if i == 10:
                    break

    LEVEL = opt.LEVEL

    results = []
    with open(os.path.join(history_path, f"{opt.model}_history.jsonl"), 'r') as f:
        for line in f:
            results.append(json.loads(line))

    history = Counter()
    for d in tqdm(results):
        tokens = [str(token) for token in d["tokens"]]
        ["/".join(tokens[i:i+LEVEL]) for i in range(0, len(tokens)-LEVEL)]
        history += Counter(list(set(["/".join(tokens[i:i+LEVEL]) for i in range(0, len(tokens)-LEVEL)])))
    os.makedirs(os.path.join(opt.datastore_dir, opt.dataset, opt.model, opt.system_prompt), exist_ok=True)
    with open(os.path.join(opt.datastore_dir, opt.dataset, opt.model, opt.system_prompt, "train_{}.json".format(LEVEL)), 'w') as f: json.dump(dict(history), f)

if __name__=="__main__":
    main()