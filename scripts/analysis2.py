import json
import argparse
from transformers import AutoTokenizer
import numpy as np
import os

def z_function(z, x, y):
    # return  (((0.0335 + 0.00024368 + x) * z / y))  + 0.00274012
    return  (((0.03272 + 0.00022342 + x) * z / y)) 


specbench_map = {
    "mt_bench" : (81, 160),
    "translation" : (161, 240),
    "summarization": (241, 320),
    "qa" : (321, 400),
    "math_reasoning" : (401, 480),
    "rag" : (481, 560)
}


def get_single_speedup(jsonl_file):
    data = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    speeds=[]
    accept_lengths_list = []
    draft_accuracy = []
    add_latency = []
    gen_latency = []
    sources = [[],[],[],[]]

    tests = []
    test2 = []
    test3 = []
    test4 = []
    for datapoint in data:
        tokens = sum(datapoint["choices"][0]['new_tokens'])
        test4.append(tokens)
        times = sum(datapoint["choices"][0]['wall_time'])
        if isinstance(datapoint["choices"][0]['accept_lengths'][0], int):
            accept_lengths_list.extend([d for d in datapoint["choices"][0]['accept_lengths']])
            # accept_lengths_list.append(sum([d[0] for d in datapoint["choices"][0]['accept_lengths']])/len([d[0] for d in datapoint["choices"][0]['accept_lengths']]))
        else:
            accept_lengths_list.extend([d[0] for d in datapoint["choices"][0]['accept_lengths']])
            accepted_tokens = [d[0] for d in datapoint["choices"][0]['accept_lengths'] if d[0] > 1]
            draft_accuracy.append(sum(accepted_tokens)/tokens)
            add_latency.extend([d[2] for d in datapoint["choices"][0]['accept_lengths']])
            # gen_latency.extend([d[3] for d in datapoint["choices"][0]['accept_lengths']])
            temps = [[],[],[],[]]
            # for d in datapoint["choices"][0]['accept_lengths']:
                # temps[d[1]].append(1)
            # temps = [sum(temp) for temp in temps]
            for i in range(4):
                sources[i].extend(temps[i])

            estimate_time = z_function(tokens, 
                                       sum([d[2] for d in datapoint["choices"][0]['accept_lengths']])/len([d[2] for d in datapoint["choices"][0]['accept_lengths']]),
                                        sum([d[0] for d in datapoint["choices"][0]['accept_lengths']])/len([d[0] for d in datapoint["choices"][0]['accept_lengths']]))
            test2.append(abs(estimate_time-times))
            # test2.append(abs(estimate_time-times))
            # test3.append(len([d[2] for d in datapoint["choices"][0]['accept_lengths']]))
            # tests.append(times - sum([d[3] for d in datapoint["choices"][0]['accept_lengths']]))
        speeds.append(tokens/times)

    print(jsonl_file)
    print("Token / Sec : {} {}".format(round(np.mean(speeds), 2), round(np.std(speeds), 2)))
    print("#Mean accepted tokens: {}".format(round(np.mean(accept_lengths_list),2)))

    add_latency = [a * 1000 for a in add_latency]

    if len(draft_accuracy) > 0:
        print('Draft Accuracy Percent: {}%'.format(round(sum(draft_accuracy)/len(draft_accuracy) * 100 , 2)))
        print('Draft Latency: {}ms {}'.format(round(np.mean(add_latency), 4), round(np.std(add_latency), 4)))

def get_spec_speedup(jsonl_file):
    data = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    print(jsonl_file)
    sources = [[],[],[],[]]
    latencies = [[],[],[],[]]

    for subtask_name in ["mt_bench", "translation", "summarization", "qa", "math_reasoning", "rag"]:
        data_range = specbench_map[subtask_name]
        speeds=[]
        accept_lengths_list = []
        # draft_hit = []
        draft_accuracy = []
        add_latency = []


        for datapoint in data:
            if datapoint["question_id"] >= data_range[0] and datapoint["question_id"] <= data_range[1]:
                tokens = sum(datapoint["choices"][0]['new_tokens'])
                times = sum(datapoint["choices"][0]['wall_time'])
                if isinstance(datapoint["choices"][0]['accept_lengths'][0], int):
                    accept_lengths_list.extend([d for d in datapoint["choices"][0]['accept_lengths']])
                    # accept_lengths_list.append(sum([d[0] for d in datapoint["choices"][0]['accept_lengths']])/len([d[0] for d in datapoint["choices"][0]['accept_lengths']]))
                else:
                    accept_lengths_list.extend([d[0] for d in datapoint["choices"][0]['accept_lengths']])
                    # accept_lengths_list.append(sum([d[0] for d in datapoint["choices"][0]['accept_lengths']])/len([d[0] for d in datapoint["choices"][0]['accept_lengths']]))
                    accepted_tokens = [d[0] for d in datapoint["choices"][0]['accept_lengths'] if d[0] > 1]
                    draft_accuracy.append(sum(accepted_tokens)/tokens)
                    add_latency.extend([d[2] for d in datapoint["choices"][0]['accept_lengths']])

                    # for d in datapoint["choices"][0]['accept_lengths']:
                    #     sources[d[1]].append(d[0])
                    #     latencies[d[1]].append(d[2])


                speeds.append(tokens/times)
        print("Task : {}".format(subtask_name))
        print("Token / Sec : {}".format(round(np.mean(speeds), 2)))
        print("#Mean accepted tokens: {}".format(round(np.mean(accept_lengths_list),2)))
            
if __name__ == "__main__":

    datasets = [
        "spec_bench", 
        # "Alpaca", 
        # "GSM8K"
    ]
    size = 7
    temp = 1.0
    # model = f"Llama-2-{size}b"
    model = f"vicuna-{size}b-v1.3"
    for dataset in datasets:

        file_pathes = [
            "/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/vicuna-7b-v1.3-vanilla-float16-prompt-default-temp-0.1.jsonl",
            "/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/vicuna-7b-v1.3-vanilla-float16-prompt-default-temp-0.2.jsonl",
            "/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/vicuna-7b-v1.3-vanilla-float16-prompt-default-temp-0.3.jsonl",
            "/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/vicuna-7b-v1.3-vanilla-float16-prompt-default-temp-0.4.jsonl",
            "/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/vicuna-7b-v1.3-vanilla-float16-prompt-default-temp-0.5.jsonl",
            "/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/vicuna-7b-v1.3-vanilla-float16-prompt-default-temp-0.6.jsonl",
            "/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/vicuna-7b-v1.3-vanilla-float16-prompt-default-temp-0.7.jsonl",
            "/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/vicuna-7b-v1.3-vanilla-float16-prompt-default-temp-0.8.jsonl",
            "/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/vicuna-7b-v1.3-vanilla-float16-prompt-default-temp-0.9.jsonl",
            "/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/vicuna-7b-v1.3-vanilla-float16-prompt-default-temp-1.0.jsonl",
        ]
        # file_pathes = [
        # f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/{model}-float16-hierarchy.jsonl",
        # f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/{model}-float16-hierarchy-W.jsonl",
        # f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/{model}-float16-hierarchy-S.jsonl",
        # f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/{model}-float16-hierarchy-L.jsonl",
        # f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/{model}-float16-hierarchy-WS.jsonl",
        # f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/{model}-float16-hierarchy-WL.jsonl",
        # f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/{model}-float16-hierarchy-SL.jsonl",
        # # f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/{model}-float16-hierarchy-WLS.jsonl",
        # # f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/{model}-float16-hierarchy-SWL.jsonl",
        # # f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/{model}-float16-hierarchy-SLW.jsonl",
        # # f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/{model}-float16-hierarchy-LWS.jsonl",
        # # f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/{model}-float16-hierarchy-LSW.jsonl",
        # ]    

        # file_pathes = [
        #     f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/{model}-vanilla-float16-prompt-default.jsonl",
        #     f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/{model}-pld-float16.jsonl",
        #     f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/{model}-lade-5-win-7-guess-7-float16-prompt-default.jsonl",
        #     f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/{model}-rest-float16-prompt-default-temperature-0.0-top_p-0.jsonl",
        #     f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/{model}-float16-hierarchy.jsonl"
        # ]

        # file_pathes = [
        #     f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/{model}-vanilla-float16-prompt-default-temp-{temp}.jsonl",
        #     f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/{model}-lade-5-win-7-guess-7-float16-prompt-default-temp-{temp}.jsonl",
        #     f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/{model}-rest-float16-prompt-default-temperature-{temp}-top_p-0.0.jsonl",
        #     f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/{model}-float16-hierarchy-temp-{temp}.jsonl"
        # ]

        # file_pathes = [
        #     f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/vicuna-{size}b-v1.3-vanilla-float16-prompt-default.jsonl",
        #     f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/vicuna-{size}b-v1.3-pld-float16.jsonl",
        #     f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/vicuna-{size}b-v1.3-lade-5-win-7-guess-7-float16-prompt-default.jsonl",
        #     f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/vicuna-{size}b-v1.3-rest-float16-prompt-default-temperature-0.0-top_p-0.jsonl",
        #     f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/vicuna-{size}b-v1.3-float16-hierarchy.jsonl"
        # ]

        # file_pathes = [
        #     f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/vicuna-{size}b-v1.3-vanilla-float16-prompt-default-temp-{temp}.jsonl",
        #     f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/vicuna-{size}b-v1.3-lade-5-win-7-guess-7-float16-prompt-default-temp-{temp}.jsonl",
        #     f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/vicuna-{size}b-v1.3-rest-float16-prompt-default-temperature-{temp}-top_p-0.0.jsonl",
        #     f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/vicuna-{size}b-v1.3-float16-hierarchy-temp-{temp}.jsonl"
        # ]

        # file_pathes = [
        #     f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/vicuna-{size}b-v1.3-float16-hierarchy-W.jsonl",
        #     f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/vicuna-{size}b-v1.3-float16-hierarchy-S.jsonl",
        #     f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/vicuna-{size}b-v1.3-float16-hierarchy-L.jsonl"
        # ]
        
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--is_specbench",
            action="store_true",
            default=False,
            help="report mean speedup over different runs")

        args = parser.parse_args()

        for file_path in file_pathes:
            if dataset == 'spec_bench':
                print("-" * 100)
                get_single_speedup(file_path)
                # print("-" * 100)
                # get_spec_speedup(file_path)
            else:
                print("-" * 100)
                get_single_speedup(file_path)