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
            accept_lengths_list.extend(datapoint["choices"][0]['accept_lengths'])
        else:
            accept_lengths_list.append(sum([d[0] for d in datapoint["choices"][0]['accept_lengths']])/len([d[0] for d in datapoint["choices"][0]['accept_lengths']]))
            accepted_tokens = [d[0] for d in datapoint["choices"][0]['accept_lengths'] if d[0] > 1]
            draft_accuracy.append(sum(accepted_tokens)/tokens)
            add_latency.extend([d[2] for d in datapoint["choices"][0]['accept_lengths']])
            gen_latency.extend([d[3] for d in datapoint["choices"][0]['accept_lengths']])
            temps = [[],[],[],[]]
            for d in datapoint["choices"][0]['accept_lengths']:
                if d[0] > 1:
                    temps[d[1]].append(d[0])
            temps = [sum(temp)/tokens for temp in temps]
            for i in range(4):
                sources[i].append(temps[i])

            estimate_time = z_function(tokens, 
                                       sum([d[2] for d in datapoint["choices"][0]['accept_lengths']])/len([d[2] for d in datapoint["choices"][0]['accept_lengths']]),
                                        sum([d[0] for d in datapoint["choices"][0]['accept_lengths']])/len([d[0] for d in datapoint["choices"][0]['accept_lengths']]))
            test2.append(abs(estimate_time-times))
            # test2.append(abs(estimate_time-times))
            test3.append(len([d[2] for d in datapoint["choices"][0]['accept_lengths']]))
        speeds.append(tokens/times)
        tests.append(times - sum([d[3] for d in datapoint["choices"][0]['accept_lengths']]))
        # from IPython import embed; embed(); exit(0)
        # print(jsonl_file)
        # print("Token / Sec : {}".format(round(np.mean(speeds), 2)))
        # print("#Mean accepted tokens: {}".format(round(np.mean(accept_lengths_list),2)))
        # if len(draft_accuracy) > 0:
        #     print('Draft Accuracy Percent: {}%'.format(round(sum(draft_accuracy)/len(draft_accuracy) * 100 , 2)))
        #     print('Draft Latency: {}ms'.format(round(sum(add_latency)/len(add_latency) * 1000, 4)))
        #     print('Gen Latency: {}s'.format(round(sum(gen_latency)/len(gen_latency), 4) - round(sum(add_latency)/len(add_latency), 4)))
        #     print("Leaked Latency: {}s".format(round(sum(tests)/ len(tests), 4)))
        # break
    # from IPython import embed; embed(); exit(0)
    print(jsonl_file)
    print("Token / Sec : {}".format(round(np.mean(speeds), 2)))
    print("#Mean accepted tokens: {}".format(round(np.mean(accept_lengths_list),2)))
    if len(draft_accuracy) > 0:
        print('Draft Accuracy Percent: {}%'.format(round(sum(draft_accuracy)/len(draft_accuracy) * 100 , 2)))
        print('Draft Latency: {}ms'.format(round(sum(add_latency)/len(add_latency) * 1000, 4)))
        print('Gen Latency: {}s'.format(round(sum(gen_latency)/len(gen_latency), 4) - round(sum(add_latency)/len(add_latency), 4)))
        print("Leaked Latency: {}s".format(round(sum(tests)/ len(tests), 4)))
        print("|Est. - Act.|: {}s +- {}".format(round(sum(test2)/len(test2), 4), np.std(test2)))
    # from IPython import embed; embed(); exit(0)
    # for source in sources:
    #     print("Source Acceptance Ratio: {}".format(round(sum(source) * 100/len(source), 2)))

def get_spec_speedup(jsonl_file):
    data = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    print(jsonl_file)
    sources = [[],[],[],[]]
    latencies = [[],[],[],[]]

    for subtask_name in ["mt_bench" , "translation", "summarization", "qa", "math_reasoning", "rag"]:
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
                    accept_lengths_list.append(sum([d[0] for d in datapoint["choices"][0]['accept_lengths']])/len([d[0] for d in datapoint["choices"][0]['accept_lengths']]))
                else:
                    accept_lengths_list.append(sum([d[0] for d in datapoint["choices"][0]['accept_lengths']])/len([d[0] for d in datapoint["choices"][0]['accept_lengths']]))
                    accepted_tokens = [d[0] for d in datapoint["choices"][0]['accept_lengths'] if d[0] > 1]
                    draft_accuracy.append(sum(accepted_tokens)/tokens)
                    add_latency.extend([d[2] for d in datapoint["choices"][0]['accept_lengths']])

                    for d in datapoint["choices"][0]['accept_lengths']:
                        sources[d[1]].append(d[0])
                        latencies[d[1]].append(d[2])


                speeds.append(tokens/times)
        print("Task : {}".format(subtask_name))
        print("Token / Sec : {}".format(round(np.mean(speeds), 2)))
        print("#Mean accepted tokens: {}".format(round(np.mean(accept_lengths_list),2)))
        if len(draft_accuracy) > 0:
            print('Draft Accuracy Percent: {}%'.format(round(sum(draft_accuracy)/len(draft_accuracy) * 100 , 2)))
            print('Draft Latency: {}ms'.format(round(sum(add_latency)/len(add_latency) * 1000, 2)))

            # total_len = sum([len(a) for a in sources])
            # for source, latency in zip(sources, latencies):
            #     print("Source Hit Percent: {}".format(round(len(source) * 100 /total_len, 2)))
            #     try:
            #         print("Source MAT: {}".format(round(sum(source)/len(source),2)))
            #         print("Source Latency: {}ms".format(round(sum(latency)/len(latency)* 1000, 3)))
            #     except ZeroDivisionError:
            #         print("Source MAT: 0")
            #         print("Source Latency: 0ms")



if __name__ == "__main__":

    # file_pathes = [
    #     "/data/smcho/Spec-Bench/data/Alpaca/model_answer/hierarchy-test.jsonl",
    #     "/data/smcho/Spec-Bench/data/Alpaca/model_answer/LAD-test.jsonl",
    #     "/data/smcho/Spec-Bench/data/Alpaca/model_answer/REST-test-temperature-0.0-top_k0-top_p-0.jsonl"
    # ]

    datasets = [
        "spec_bench", 
        # "Alpaca", 
        # "GSM8K"
    ]

    for dataset in datasets:

        file_pathes = [
            # "/data/smcho/Spec-Bench/data/spec_bench/model_answer/hierarchy_ test_7.jsonl",
            # "/data/smcho/Spec-Bench/data/spec_bench/model_answer/hierarchy_test_8.jsonl",
            # "/data/smcho/Spec-Bench/data/spec_bench/model_answer/hierarchy_test_9.jsonl",
            # "/data/smcho/Spec-Bench/data/spec_bench/model_answer/hierarchy_test_10.jsonl",
            # "/data/smcho/Spec-Bench/data/spec_bench/model_answer/hierarchy_test_11.jsonl",
            # "/data/smcho/Spec-Bench/data/spec_bench/model_answer/hierarchy_test_12.jsonl",
            # "/data/smcho/Spec-Bench/data/spec_bench/model_answer/hierarchy_test_13.jsonl",
            # "/data/smcho/Spec-Bench/data/spec_bench/model_answer/hierarchy_test_14.jsonl",
            # "/data/smcho/Spec-Bench/data/spec_bench/model_answer/hierarchy_test_15.jsonl",
            # "/data/smcho/Spec-Bench/data/spec_bench/model_answer/hierarchy_test_16.jsonl",
            # "/data/smcho/Spec-Bench/data/spec_bench/model_answer/hierarchy_test_17.jsonl",
            # "./data/spec_bench/model_answer/hierarchy_test_18.jsonl",
            # "./data/spec_bench/model_answer/hierarchy_test_19.jsonl",
            # "./data/spec_bench/model_answer/hierarchy_test_20.jsonl",
            # "./data/spec_bench/model_answer/hierarchy_test_21.jsonl",
            # "./data/spec_bench/model_answer/hierarchy_test_22.jsonl",
            # "./data/spec_bench/model_answer/hierarchy_test_23.jsonl",
            # "./data/spec_bench/model_answer/hierarchy_test_24.jsonl",
            # "./data/spec_bench/model_answer/hierarchy_test_25.jsonl",
            # "./data/spec_bench/model_answer/hierarchy_test_26.jsonl",
            # "./data/spec_bench/model_answer/hierarchy_test_27.jsonl",
            # "./data/spec_bench/model_answer/hierarchy_test_28.jsonl",
            # "./data/spec_bench/model_answer/hierarchy_test_29.jsonl",
            # "./data/spec_bench/model_answer/hierarchy_test_30.jsonl",
            # "./data/spec_bench/model_answer/hierarchy_test_31.jsonl",
            # "./data/spec_bench/model_answer/hierarchy_test_32.jsonl",
            # "./data/spec_bench/model_answer/hierarchy_test_33.jsonl",
            # "./data/spec_bench/model_answer/hierarchy_test_34.jsonl",
            # "./data/spec_bench/model_answer/hierarchy_test_35.jsonl",
            # "./data/spec_bench/model_answer/hierarchy_test_36.jsonl",
            # "./data/spec_bench/model_answer/hierarchy_test_37.jsonl",
            # "./data/spec_bench/model_answer/hierarchy_test_38.jsonl",
            # "./data/spec_bench/model_answer/hierarchy_test_40.jsonl",
            # "./data/spec_bench/model_answer/hierarchy_test_41.jsonl",
            # "./data/spec_bench/model_answer/hierarchy_test_42.jsonl",
            # "./data/spec_bench/model_answer/hierarchy_test_43.jsonl",
            # "./data/spec_bench/model_answer/hierarchy_test_44.jsonl",
            "./data/spec_bench/model_answer/hierarchy_test_42.jsonl",
            "./data/spec_bench/model_answer/hierarchy_test_46.jsonl",
            "./data/spec_bench/model_answer/hierarchy_test_48.jsonl",
        ]

        # vicuna - spec_bench - greedy
        # file_pathes = [
        #     # f"/data/smcho/Spec-Bench/data/{dataset}/model_answer/vicuna-7b-v1.3-vanilla-float16-prompt-default.jsonl",
        #     # f"/data/smcho/Spec-Bench/data/{dataset}/model_answer/LAD-test.jsonl",
        #     # f"/data/smcho/Spec-Bench/data/{dataset}/model_answer/pld-test2.jsonl",
        #     "/data/smcho/Spec-Bench/data/spec_bench/model_answer/vicuna-7b-v1.3-hierachy-level-5-win-7-guess-7-previous-2-order-WSL-float16-prompt-default-size-100000.jsonl",
        #     # f"/data/smcho/Spec-Bench/data/{dataset}/model_answer/REST-test2-temperature-0.0-top_k0-top_p-0.jsonl"
        # ]

        # vicuna - others -greedy
        # file_pathes = [
        #    f"/data/smcho/Spec-Bench/data/{dataset}/model_answer/vicuna-7b-v1.3-vanilla-float16-prompt-default.jsonl",
        #    f"/data/smcho/Spec-Bench/data/{dataset}/model_answer/LAD-test.jsonl",
        #    f"/data/smcho/Spec-Bench/data/{dataset}/model_answer/pld-test.jsonl",
        #    f"/data/smcho/Spec-Bench/data/{dataset}/model_answer/vicuna-7b-v1.3-hierachy-level-5-win-7-guess-7-previous-2-order-WSL-float16-prompt-default.jsonl",
        #    f"/data/smcho/Spec-Bench/data/{dataset}/model_answer/REST-test-temperature-0.0-top_k0-top_p-0.jsonl"
        # ]

        # vicuna- Sampling
        # file_pathes = [
        #     f"/data/smcho/Spec-Bench/data/{dataset}/model_answer/vicuna-7b-v1.3-vanilla-float16-do_sample-prompt-default.jsonl",
        #     f"/data/smcho/Spec-Bench/data/{dataset}/model_answer/vicuna-7b-v1.3-lade-5-win-7-guess-7-float16-do_sample-prompt-default.jsonl",
        #     f"/data/smcho/Spec-Bench/data/{dataset}/model_answer/vicuna-7b-v1.3-rest-float16-do_sample-prompt-default-temperature-0.7-top_k50-top_p-0.9.jsonl",
        #     f"/data/smcho/Spec-Bench/data/{dataset}/model_answer/vicuna-7b-v1.3-hierachy-level-5-win-7-guess-7-previous-2-order-WSL-float16-do_sample-prompt-default.jsonl",
        # ]

        # Llama-2-7b - Greedy
        # file_pathes = [
            # f"/data/smcho/Spec-Bench/data/{dataset}/model_answer/Llama-2-7b-vanilla-float16-prompt-default.jsonl",
            # f"/data/smcho/Spec-Bench/data/{dataset}/model_answer/Llama-2-7b-pld-float16.jsonl",
            # f"/data/smcho/Spec-Bench/data/{dataset}/model_answer/Llama-2-7b-lade-5-win-7-guess-7-float16-prompt-default.jsonl",
            # f"/data/smcho/Spec-Bench/data/{dataset}/model_answer/Llama-2-7b-rest-float16-prompt-default-temperature-0.0-top_k0-top_p-0.jsonl",
            # f"/data/smcho/Spec-Bench/data/{dataset}/model_answer/Llama-2-7b-hierachy-level-5-win-7-guess-7-previous-2-order-WSL-float16-prompt-default.jsonl",
        # ]

        # Llama-2-7b - Sampling
        # file_pathes = [
        #     f"/data/smcho/Spec-Bench/data/{dataset}/model_answer/Llama-2-7b-vanilla-float16-do_sample-prompt-default.jsonl",
        #     f"/data/smcho/Spec-Bench/data/{dataset}/model_answer/Llama-2-7b-lade-5-win-7-guess-7-float16-do_sample-prompt-default.jsonl",
        #     f"/data/smcho/Spec-Bench/data/{dataset}/model_answer/Llama-2-7b-rest-float16-do_sample-prompt-default-temperature-0.7-top_k50-top_p-0.9.jsonl",
        #     f"/data/smcho/Spec-Bench/data/{dataset}/model_answer/Llama-2-7b-hierachy-level-5-win-7-guess-7-previous-2-order-WSL-float16-do_sample-prompt-default.jsonl",
        # ]

        # Size
        # file_pathes = [
            # "/data/smcho/Spec-Bench/data/spec_bench/model_answer/hierarchy-test-order-WSL-float16.jsonl",
            # "/data/smcho/Spec-Bench/data/spec_bench/model_answer/vicuna-7b-v1.3-hierachy-level-5-win-7-guess-7-previous-2-order-WSL-float16-prompt-default-size-50000.jsonl",
            # "/data/smcho/Spec-Bench/data/spec_bench/model_answer/vicuna-7b-v1.3-hierachy-level-5-win-7-guess-7-previous-2-order-WSL-float16-prompt-default-size-100000.jsonl",
            # "/data/smcho/Spec-Bench/data/spec_bench/model_answer/vicuna-7b-v1.3-hierachy-level-5-win-7-guess-7-previous-2-order-WSL-float16-prompt-default-size-1000000.jsonl",
            # "/data/smcho/Spec-Bench/data/spec_bench/model_answer/vicuna-7b-v1.3-hierachy-level-5-win-7-guess-7-previous-2-order-WSL-float16-prompt-default-size-10000000.jsonl"
        # ]
        
        # Order
        # file_pathes = [
        #     "/data/smcho/Spec-Bench/data/spec_bench/model_answer/vicuna-7b-v1.3-hierachy-level-5-win-7-guess-7-previous-2-order-WSL-float16-prompt-default-size-100000.jsonl",
        #     "/data/smcho/Spec-Bench/data/spec_bench/model_answer/hierarchy-test-order-WLS-float16.jsonl",
        #     "/data/smcho/Spec-Bench/data/spec_bench/model_answer/hierarchy-test-order-WSL-float16.jsonl",
        #     "/data/smcho/Spec-Bench/data/spec_bench/model_answer/hierarchy-test-order-SWL-float16.jsonl",
        #     "/data/smcho/Spec-Bench/data/spec_bench/model_answer/hierarchy-test-order-SLW-float16.jsonl",
        #     "/data/smcho/Spec-Bench/data/spec_bench/model_answer/hierarchy-test-order-LWS-float16.jsonl",
        #     "/data/smcho/Spec-Bench/data/spec_bench/model_answer/hierarchy-test-order-LSW-float16.jsonl"
        # ]

        # file_pathes = [
        #     "/data/smcho/Spec-Bench/data/spec_bench/model_answer/hierarchy-test-order-W-float16.jsonl",
        #     "/data/smcho/Spec-Bench/data/spec_bench/model_answer/hierarchy-test-order-S-float16.jsonl",
        #     "/data/smcho/Spec-Bench/data/spec_bench/model_answer/hierarchy-test-order-L-float16.jsonl",

        #     "/data/smcho/Spec-Bench/data/spec_bench/model_answer/vicuna-7b-v1.3-hierachy-level-5-win-7-guess-7-previous-2-order-WSL-float16-prompt-default-size-100000.jsonl",            "/data/smcho/Spec-Bench/data/spec_bench/model_answer/hierarchy-test-order-WLS-float16.jsonl",
        #     "/data/smcho/Spec-Bench/data/spec_bench/model_answer/hierarchy-test-order-SWL-float16.jsonl",
        #     "/data/smcho/Spec-Bench/data/spec_bench/model_answer/hierarchy-test-order-SLW-float16.jsonl",
        #     "/data/smcho/Spec-Bench/data/spec_bench/model_answer/hierarchy-test-order-LWS-float16.jsonl",
        #     "/data/smcho/Spec-Bench/data/spec_bench/model_answer/hierarchy-test-order-LSW-float16.jsonl",
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
