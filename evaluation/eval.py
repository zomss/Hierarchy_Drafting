"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
# adapted from fastchat: https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/gen_model_answer.py

import json
import os
import time
import torch
import numpy as np
import shortuuid

from fastchat.llm_judge.common import load_questions
from fastchat.model import get_conversation_template
from tqdm import tqdm

system_prompt_mapping = {
    "Llama" : "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\'t know the answer to a question, please don\'t share false information."
}

def run_eval(
        model,
        tokenizer,
        forward_func,
        model_id,
        question_file,
        question_begin,
        question_end,
        answer_file,
        max_new_tokens,
        num_choices,
        num_gpus_per_model,
        num_gpus_total,
        **kwargs,
):
    questions = load_questions(question_file, question_begin, question_end)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        import ray
        ray.init()
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)  # // 2
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model,
                tokenizer,
                forward_func,
                model_id,
                questions[i: i + chunk_size],
                answer_file,
                max_new_tokens,
                num_choices,
                **kwargs,
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
        model,
        tokenizer,
        forward_func,
        model_id,
        questions,
        answer_file,
        max_new_tokens,
        num_choices,
        system_prompt="default",
        **kwargs,
):

    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    question = questions[0]

    # warmup
    for _ in range(3):
        torch.manual_seed(0)
        # conv = get_conversation_template("vicuna")

        conv = get_conversation_template("Llama-2-7b")
        conv.set_system_message(system_message=system_prompt_mapping["Llama"])
        
        turns = []
        steps = []
        new_tokens = []
        wall_time = []
        for j in range(len(question["turns"])):
            qs = question["turns"][j]
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            conv.stop_str = "</s>"
            prompt = conv.get_prompt()
            inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
            input_ids = inputs.input_ids
            try:
                torch.cuda.synchronize()
                start_time = time.time()
                output_ids, new_token, step, accept_length_tree = forward_func(
                    inputs,
                    model,
                    tokenizer,
                    max_new_tokens,
                    **kwargs,
                )
                torch.cuda.synchronize()
                total_time = time.time() - start_time
                output_ids = output_ids[0][len(input_ids[0]):]
                # be consistent with the template's stop_token_ids
                if conv.stop_token_ids:
                    stop_token_ids_index = [
                        i
                        for i, id in enumerate(output_ids)
                        if id in conv.stop_token_ids
                    ]
                    if len(stop_token_ids_index) > 0:
                        output_ids = output_ids[: stop_token_ids_index[0]]

                output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                )
                if conv.stop_str and output.find(conv.stop_str) > 0:
                    output = output[: output.find(conv.stop_str)]
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

                if conv.name == "xgen" and output.startswith("Assistant:"):
                    output = output.replace("Assistant:", "", 1).strip()
                turns.append(output)
                steps.append(int(step))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                conv.messages[-1][-1] = output
            except RuntimeError as e:
                print("ERROR question ID: ", question["question_id"])
                output = "ERROR"

    print('Warmup done')

    accept_lengths_tree = []
    for question in tqdm(questions):

        choices = []
        for i in range(num_choices):
            cur_accept_lengths_tree = []
            torch.manual_seed(i)
            # conv = get_conversation_template("vicuna")

            conv = get_conversation_template("Llama-2-7b")
            conv.set_system_message(system_message=system_prompt_mapping["Llama"])

            turns = []
            steps = []
            new_tokens = []
            wall_time = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                conv.stop_str = "</s>"
                prompt = conv.get_prompt()
                inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
                input_ids = inputs.input_ids
                try:
                    torch.cuda.synchronize()
                    start_time = time.time()
                    output_ids, new_token, step, accept_length_tree = forward_func(
                        inputs,
                        model,
                        tokenizer,
                        max_new_tokens,
                        **kwargs,
                    )
                    torch.cuda.synchronize()
                    total_time = time.time() - start_time
                    accept_lengths_tree.extend(accept_length_tree)
                    output_ids = output_ids[0][len(input_ids[0]):]
                    if conv.stop_token_ids:
                        stop_token_ids_index = [
                            i
                            for i, id in enumerate(output_ids)
                            if id in conv.stop_token_ids
                        ]
                        if len(stop_token_ids_index) > 0:
                            output_ids = output_ids[: stop_token_ids_index[0]]

                    output = tokenizer.decode(
                        output_ids,
                        spaces_between_special_tokens=False,
                    )
                    if conv.stop_str and output.find(conv.stop_str) > 0:
                        output = output[: output.find(conv.stop_str)]
                    for special_token in tokenizer.special_tokens_map.values():
                        if isinstance(special_token, list):
                            for special_tok in special_token:
                                output = output.replace(special_tok, "")
                        else:
                            output = output.replace(special_token, "")
                    output = output.strip()

                    if conv.name == "xgen" and output.startswith("Assistant:"):
                        output = output.replace("Assistant:", "", 1).strip()
                except RuntimeError as e:
                    print("ERROR question ID: ", question["question_id"])
                    output = "ERROR"

                turns.append(output)
                steps.append(int(step))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                cur_accept_lengths_tree.extend(accept_length_tree)
                conv.messages[-1][-1] = output
            # torch.cuda.empty_cache()
            choices.append({"index": i, "turns": turns, "decoding_steps": steps, "new_tokens": new_tokens, "wall_time": wall_time,
                            "accept_lengths": cur_accept_lengths_tree})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "category": question["category"] if 'category' in question else "None",
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")
    calculate_metric(answer_file)

def calculate_metric(answer_file):
    data = []
    with open(os.path.expanduser(answer_file), 'r') as f:
        for line in f:
            data.append(json.loads(line))
    speeds = []
    accept_lengths_list = []
    for datapoint in data:
        tokens=sum(datapoint["choices"][0]['new_tokens'])
        times = sum(datapoint["choices"][0]['wall_time'])
        if isinstance(datapoint["choices"][0]['accept_lengths'][0], int):
            accept_lengths_list.extend(datapoint["choices"][0]['accept_lengths'])
        else:
            accept_lengths_list.extend([d[0] for d in datapoint["choices"][0]['accept_lengths']])
        speeds.append(tokens/times)
    print("Token / Sec : {}".format(round(np.mean(speeds), 2)))
    print("#Mean accepted tokens: {}".format(round(np.mean(accept_lengths_list),2)))


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])

