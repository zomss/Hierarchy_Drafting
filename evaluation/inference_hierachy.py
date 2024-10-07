"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
# adapted from fastchat: https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/gen_model_answer.py
import argparse
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastchat.utils import str_to_torch_dtype

from evaluation.eval import run_eval, reorg_answer_file

from model.hierachy.utils import augment_all, config_lade
from model.hierachy.decoding import CONFIG_MAP, set_memory

def hierachy_forward(inputs, model, tokenizer, max_new_tokens, do_sample, temperature):
    model_inputs = inputs.to("cuda")
    output_ids, idx, accept_length_list = model.generate(
        **model_inputs,
        do_sample=do_sample,
        temperature=temperature,
        top_k = 0,
        max_new_tokens=max_new_tokens,
    )
    new_token = len(output_ids[0][len(model_inputs.input_ids[0]):])
    return output_ids, new_token, idx+1, accept_length_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--model-id", type=str, required=True, help="A custom name for the model."
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--level",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--guess",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU.",
    )

    parser.add_argument(
        "--do_sample",
        action="store_true"
    )
    parser.add_argument(
        "--temperature", 
        type=float,
        default=0.7
    )


    # Memory Setting
    parser.add_argument("--do_WM", action="store_true")
    parser.add_argument("--do_SM", action="store_true")
    parser.add_argument("--do_LM", action="store_true")
    parser.add_argument("--order", type=str, default="WSL")
    parser.add_argument("--history_file", type=str, default="train-Llama-2-7b-history2.json")
    parser.add_argument("--db_file", type=str, default="datastore_chat_large.idx")
    parser.add_argument("--previous_tokens", type=int, default=8)
    parser.add_argument("--system_prompt", type=str, default="default")


    args = parser.parse_args()
    if int(os.environ.get("USE_LADE", 0)):
        augment_all()
        config_lade(LEVEL=args.level, WINDOW_SIZE=args.window, GUESS_SET_SIZE=args.guess, DEBUG=0,
                         USE_FLASH=0, DIST_WORKERS=len(os.environ.get("CUDA_VISIBLE_DEVICES").split(",")), DO_WM=args.do_WM, DO_SM=args.do_SM, DO_LM=args.do_LM, ORDER=args.order, PREVIOUS_TOKENS=args.previous_tokens)
        print("hierahcy activated config: ", CONFIG_MAP)

    question_file = f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    set_memory(args)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=hierachy_forward,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        do_sample=args.do_sample,
        temperature=args.temperature,
        system_prompt=args.system_prompt
    )

    reorg_answer_file(answer_file)
