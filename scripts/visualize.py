import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.ticker as mticker
from matplotlib.ticker import MultipleLocator
from matplotlib_venn import venn3, venn3_circles
import numpy as np
import json
import matplotlib.lines as mlines
import venn

# plt.rcParams.update({'font.size': 12})  # Reduced font size for better fit
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

def read_data(jsonl_file):
    data = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def update_ticks(x, pos):
    if x >= 2.1:
        return ''
    else:
        return round(x,1)

def plot_radar(ax, data, categories, title):
    num_categories = len(categories)
    angles = [n / float(num_categories) * 2 * np.pi + np.pi/6 for n in range(num_categories)]
    angles += angles[:1]

    def plot_method(values, color, label):
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2, label=label, alpha=0.9)
        ax.fill(angles, values, color=color, alpha=0.1)

    plot_method(data['LAD'], 'tab:blue', 'LAD')
    plot_method(data['PLD'], 'tab:red', 'PLD')
    plot_method(data['REST'], 'tab:green', 'REST')
    plot_method(data['HD'], 'tab:purple', 'HD')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=14)

    for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
        if angle == angles[0]:
            label.set_ha('left')
            label.set_va('top')
        elif angle == angles[1]:
            label.set_ha('center')
            label.set_va('bottom')
        elif angle == angles[2]:
            label.set_ha('right')
            label.set_va('top')
        elif angle == angles[3]:
            label.set_ha('right')
            label.set_va('bottom')
        elif angle == angles[4]:
            label.set_ha('center')
            label.set_va('top')
        else:
            label.set_ha('left')
            label.set_va('bottom')

    if title == "Vicuna-7b":
        ax.set_ylim(0.9, 1.87)
    else:
        ax.set_ylim(0.9, 1.97)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(update_ticks))
    ax.tick_params(axis='y', which='major', labelsize=10)
    ax.grid(True)
    ax.xaxis.grid(True, color='gray', linestyle='--', alpha=0.3)
    ax.set_title(title, fontsize=22, pad=10)

def draw_specbench():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5), subplot_kw=dict(projection='polar'))

    # Data for the first plot (original data)
    categories1 = ['Multi-turn\nConversation', 'Translation', 'Summar\nization', 'QA', 'Math Reasoning', 'RAG']
    data1 = {
        'HD': [1.62, 1.07, 1.76, 1.39, 1.74, 1.51],
        'LAD': [1.28, 1.03, 1.14, 1.15, 1.42, 1.01],
        'PLD': [1.35, 0.93, 1.83, 1.05, 1.41, 1.41],
        'REST': [1.31, 0.97, 1.14, 1.36, 1.00, 1.25]
    }

    # Data for the second plot (new data)
    categories2 = ['Multi-turn\nConversation', 'Translation', 'Summar\nization', 'QA', 'Math Reasoning', 'RAG']
    data2 = {
        'HD': [1.70, 1.15, 1.86, 1.36, 1.78, 1.66],
        'LAD': [1.31, 1.03, 1.18, 1.12, 1.49, 1.09],
        'PLD': [1.37, 0.99, 1.82, 1.04, 1.50, 1.49],
        'REST': [1.56, 1.14, 1.33, 1.57, 1.21, 1.50]
    }

    # Data for the second plot (new data)
    categories3 = ['Multi-turn\nConversation', 'Translation', 'Summar\nization', 'QA', 'Math Reasoning', 'RAG']
    data3 = {
        'HD': [1.66, 1.67, 1.54, 1.56, 1.82, 1.53],
        'LAD': [1.28, 1.21, 1.14, 1.20, 1.36, 1.12],
        'PLD': [1.25, 1.11, 1.20, 1.03, 1.23, 1.26],
        'REST': [1.42, 1.26, 1.42, 1.50, 1.28, 1.41]
    }

    plot_radar(ax1, data1, categories1, "Vicuna-7b")
    plot_radar(ax2, data2, categories2, "Vicuna-13b")
    plot_radar(ax3, data3, categories3, "Llama-7b")
    # Add a single legend for both plots
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, +0.03), fontsize=14, ncol=4)

    plt.tight_layout()
    plt.savefig('./figures/spec_bench.jpg', dpi=600, bbox_inches='tight')

def plot_model(ax, times, performances, accelerations, methods, markers, colors, title):
    for method, time, performance, marker, color, acceleration in zip(methods, times, performances, markers, colors, accelerations):
        s = 150 if method == 'HD (Ours)' else 50
        ax.scatter(time, performance, marker=marker, color=color, s=s, edgecolor="black")
        
        weight = 'bold' if method == 'HD (Ours)' else 'normal'
        fontsize = 14 if method == 'HD (Ours)' else 12
        if method == 'HD (Ours)':
            xytext = (5, -10)
        elif method == "LADE":
            xytext = (5,-13)
        elif method == "REST":
            xytext = (-36, -10)
        else:
            xytext = (5, 5)
        ax.annotate(f"{acceleration:.2f}x", (time, performance), xytext=xytext, textcoords='offset points', fontsize=fontsize, weight=weight)

    ax.set_xlabel('Drafting Latency (ms)', fontsize=14)
    ax.set_ylabel('Acceptance Ratio (%)', fontsize=14)
    # ax.set_title(title, fontsize=22, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)
    if title == "Vicuna-7b":
        ax.set_xlim(-0.1, 3.0)
        ax.set_ylim(40, 80)
    else:
        ax.set_xlim(-0.1, 3.3)
        ax.set_ylim(30, 76)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.tick_params(axis='both', which='major', labelsize=10)

def draw_task_bar(ax, scores, speedup, tasks, decodings, colors):
    bar_width = 0.22
    index = np.arange(len(decodings))
    tasks = ['QA', 'Summarization']

    # # Add speedup values on top of the bars
    # for i, (pos, spd) in enumerate(zip(flat_positions, speedup)):
    #     total_height = sum(weight_counts[label][i] for label in weight_counts)
    #     if i == 0:
    #         ax.text(pos, total_height + 2, f'{spd:.2f}x', ha='center', va='bottom', fontsize=16, weight = 'bold')
    #     else:
    #         ax.text(pos, total_height + 2, f'{spd:.2f}x', ha='center', va='bottom', fontsize=14)

    # Plotting bars for each decoding method
    for i, (method, color) in enumerate(zip(decodings, colors)):
        weight = 'normal' if method != 'Ours' else 'bold'
        qa_score, summarization_score = scores[method]
        ax.bar(index[0] + i*bar_width, qa_score, bar_width-0.05, label=method, color=color, edgecolor='black')
        ax.text(index[0] + i*bar_width, qa_score + 1, speedup[method][0], ha='center', va='bottom', fontsize=7.5, weight=weight)
        ax.bar(index[1] + i*bar_width, summarization_score, bar_width-0.05, label=method, color=color, edgecolor='black')
        ax.text(index[1] + i*bar_width, summarization_score + 1, speedup[method][1], ha='center', va='bottom', fontsize=7.5, weight=weight)

    # Adding labels and titles
    ax.set_ylabel('Token/sec', fontsize=14)
    ax.set_xlabel('Task Type', fontsize=14)
    # ax.set_title('Scores for QA and Summarization Tasks')

    # Set x-ticks and labels
    a = index + bar_width * 1.5
    print(a[:2])
    ax.set_xticks(a[:2])
    ax.set_xticklabels(tasks, fontsize=10)
    ax.axvline(x=sum(a[:2]) / 2, color='grey', linestyle='--', linewidth=1)

    # Add a legend for decoding methods
    # ax.legend(title="Decoding Methods", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout to prevent clipping of labels
    plt.tight_layout()

    # Set y-axis limits
    ax.set_ylim(50,85)

    # Add gridlines
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def draw_motivation():
    # Data
    methods = ['HD (Ours)', "PLD", "REST", "LADE"]
    markers = ['*', 'x', 'x', 'x']
    colors = ['purple', 'blue', 'green', 'red']

    # Vicuna-7b data
    times_vicuna = [2.17, 0.31, 2.86, 0.01]
    performances_vicuna = [75.21, 45.22, 66.51, 43.92]
    accelerations_vicuna = [1.51, 1.33, 1.18, 1.17]

    # Llama2-7b data
    # times_llama2 = [0.44, 0.02, 0.78, 3.02]
    # performances_llama2 = [73.69, 50.22, 35.26, 72.33]
    # accelerations_llama2 = [1.71, 1.36, 1.20, 1.39]

    # Create the plot with two subplots side by side
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=600)
    fig, ax1 = plt.subplots(1, 1, figsize=(7.5, 3), dpi=600)

    # Plot Vicuna-7b data
    plot_model(ax1, times_vicuna, performances_vicuna, accelerations_vicuna, methods, markers, colors, "Vicuna-7b")

    # Plot Llama2-7b data
    # plot_model(ax2, times_llama2, performances_llama2, accelerations_llama2, methods, markers, colors, "Llama2-7b")

    # Add legend
    legend_elements = [mlines.Line2D([0], [0], marker=marker, color=color, label=method, markersize=14, linestyle='None')
                       for marker, color, method in zip(markers, colors, methods)]
    fig.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.15), loc='lower center', fontsize=16, ncol=4)

    # Adjust layout and save the plot
    plt.tight_layout()
    # fig.subplots_adjust(bottom=0.2)  # Increased bottom margin for legend
    plt.savefig('./figures/latency_vs_acceptance.jpg', dpi=600, bbox_inches='tight')
    plt.close()

def draw_motivation2():
    # Data
    methods = ["PLD", "LADE", "REST", "HD (Ours)"]
    # methods = ['HD (Ours)', "PLD", "REST", "LADE"]
    markers = ['o', 'o', 'o', '*']
    colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple']

    # Vicuna-7b data
    times_vicuna = [0.31, 0.01, 2.86,  2.17]
    performances_vicuna = [45.22, 43.92, 66.51, 75.21, ]
    accelerations_vicuna = [1.33, 1.17, 1.18, 1.51]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3), dpi=600)

    # Plot Vicuna-7b data
    plot_model(ax2, times_vicuna, performances_vicuna, accelerations_vicuna, methods, markers, colors, "Vicuna-7b")

    # legend_elements = [mlines.Line2D([0], [0], marker=marker, color=color, label=method, markersize=14, linestyle='None')
    #                    for marker, color, method in zip(markers, colors, methods)]
    # ax1.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.35), loc='lower center', fontsize=10, ncol=4)

    # Data for the two tasks (QA and Summarization)
    tasks = ['QA', 'RAG']
    decodings = ['PLD', 'LADE', 'REST', 'Ours']

    # Scores for each task and decoding method
    scores = {
        'PLD': [61.3, 71.85],
        'LADE': [67.48, 51.61],
        'REST': [79.62, 63.80],
        'Ours': [81.15, 77.04]
    }
    acceleration_ratio = {
        'PLD': ["1.05x", "1.41x"],
        'LADE': ["1.15x", "1.01x"],
        "REST": ["1.36x", "1.25x"],
        "Ours": ["1.39x", "1.51x"]
    }
    colors2 = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple']
    
    draw_task_bar(ax1, scores, acceleration_ratio, tasks, decodings, colors2)
    # Plot Llama2-7b data
    # plot_model(ax2, times_llama2, performances_llama2, accelerations_llama2, methods, markers, colors, "Llama2-7b")

    # Add legend
    legend_elements = [mlines.Line2D([0], [0], marker=marker, color=color, label=method, markersize=14, markeredgecolor='black', linestyle='None')
                       for marker, color, method in zip(markers, colors, methods)]
    fig.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.15), loc='lower center', fontsize=16, ncol=4)

    # Adjust layout and save the plot
    plt.tight_layout()
    # fig.subplots_adjust(bottom=0.2)  # Increased bottom margin for legend
    plt.savefig('./figures/motivation.jpg', dpi=600, bbox_inches='tight')
    plt.close()

def draw_overlap():
    def change_fnc(datapoint):
        result = []
        for d in datapoint:
            for _ in range(0,d[0]):
                if d[1] == -1:
                    result.append(False)
                elif d[0] == 1:
                    result.append(False)
                else:
                    result.append(True)
        return result
    
    model_name = "Llama-2-7b"
    file_pathes = [
        f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/{model_name}-float16-hierarchy-W.jsonl",
        f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/{model_name}-float16-hierarchy-S.jsonl",
        f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/{model_name}-float16-hierarchy-L.jsonl",
    ]

    results = []
    for jsonl_file in file_pathes:
        data = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        results.append(data)

    data = [[],[],[],[],[],[],[]]

    for r1, r2,r3 in zip(results[0], results[1], results[2]):
        if len(r1["choices"][0]['new_tokens']) == len(r2["choices"][0]['new_tokens']) and len(r2["choices"][0]['new_tokens']) == len(r3["choices"][0]['new_tokens']):
            datapoint1 = change_fnc(r1["choices"][0]["accept_lengths"])
            datapoint2 = change_fnc(r2["choices"][0]["accept_lengths"])
            datapoint3 = change_fnc(r3["choices"][0]["accept_lengths"])
            

            d12 = [x and y for x,y in zip(datapoint1, datapoint2)]
            d23 = [x and y for x,y in zip(datapoint2, datapoint3)]
            d13 = [x and y for x,y in zip(datapoint1, datapoint3)]
            d123 = [x and y for x,y in zip(d12, datapoint3)]

            data[0].append(sum(datapoint1)/len(datapoint1))
            data[1].append(sum(datapoint2)/len(datapoint2))
            data[2].append(sum(datapoint3)/len(datapoint3))
            data[3].append(sum(d12)/len(d12))
            data[4].append(sum(d23)/len(d23))
            data[5].append(sum(d13)/len(d13))
            data[6].append(sum(d123)/len(d123))
        else:
            continue
            
    data = [round(sum(d) * 100/len(d)) for d in data]
    data[3] = data[3] - data[6]
    data[4] = data[4] - data[6]
    data[5] = data[5] - data[6]
    data[0] = data[0] - data[3] - data[5] - data[6]
    data[1] = data[1] - data[3] - data[4] - data[6]
    data[2] = data[2] - data[4] - data[5] - data[6]

    positions = ['100', '010', '110', '001', '101', '011', '111']

    fig, ax = plt.subplots(figsize=(3,3), dpi=600)


    v = venn3(subsets= data, set_labels=(r"$\mathcal{D}_{c}$",r"$\mathcal{D}_{m}$",r"$\mathcal{D}_{s}$",), alpha=0.4, ax=ax, set_colors=("blue", "orange", "green"))
    venn3_circles(subsets=data, linestyle="dashed", linewidth=0.2, ax=ax)
    for d, position in zip(data, positions):
        v.get_label_by_id(position).set_text(str(d) + "%")
        v.get_label_by_id(position).set_fontsize(10)
    v.get_label_by_id('A').set_fontsize(12)
    v.get_label_by_id('B').set_fontsize(12)
    v.get_label_by_id('C').set_fontsize(12)

    # Adjust layout and save the plot
    plt.tight_layout()
    # ax.set_title("Accepted Token Overlap between drafting sources", fontsize=16)
    # fig.subplots_adjust(bottom=0.2)  # Increased bottom margin for legend

    plt.savefig('./figures/venn.jpg', dpi=600, bbox_inches='tight')

def draw_order():
    speedup = (1.72, 1.63, 1.15, 1.06, 1.67, 1.31, 1.22, 1.07, 1.07)

    order = (
        "CMS",
        "C",
        "M",
        "S",
        "CSM",
        "MCS",
        "MSC",
        "SCM",
        "SMC"
    )

    categories = {
        "CMS": 0,
        "C": 1, "M": 1, "S": 1,
        "CSM": 2, "MCS": 2, "MSC": 2, "SCM": 2, "SMC": 2
    }

    positions = {
        0: 1,    # CMS
        1: [2, 2.5, 3],  # C, M, S
        2: [4, 4.5, 5, 5.5, 6]   # CSM, MCS, MSC, SCM, SMC
    }

    order_positions = [positions[categories[o]] for o in order]
    flat_positions = [item if isinstance(item, int) else item.pop(0) for item in order_positions]

    weight_counts = {
        "Context-based\nSource": np.array([53.72, 58.09, 0, 0, 53.02, 18.03, 0, 0, 0]),
        "Model-based\nSource": np.array([9.16, 0, 31.51, 0, 0, 27.54, 29.12, 0, 0]),
        "Statistics-based\nSource": np.array([4.59, 0, 0, 53.84, 17.26, 4.29, 14.27, 53.55, 54.10])
    }

    width = 0.3

    bottom = np.zeros(9)

    fig, ax = plt.subplots(figsize=(14, 3), dpi=600)

    for label, weight_count in weight_counts.items():
        bars = ax.bar(flat_positions, weight_count, width, label=label, bottom=bottom)
        bottom += weight_count

    # Add speedup values on top of the bars
    for i, (pos, spd) in enumerate(zip(flat_positions, speedup)):
        total_height = sum(weight_counts[label][i] for label in weight_counts)
        if i == 0:
            ax.text(pos, total_height + 2, f'{spd:.2f}x', ha='center', va='bottom', fontsize=16, weight = 'bold')
        else:
            ax.text(pos, total_height + 2, f'{spd:.2f}x', ha='center', va='bottom', fontsize=14)


    ax.set_xticks(flat_positions)
    ax.set_xticklabels(order, fontsize=14)
    plt.ylim(0, 85)
    ax.tick_params(axis='both', which='major', labelsize=12)

    ax.set_xlabel('Draft sources', fontsize=14)
    ax.set_ylabel('Acceptance Ratio (%)', fontsize=14)

    ax.legend(bbox_to_anchor=(1.12, 0.1), loc='lower center', fontsize=14, ncol=1)
    

    category_boundaries = [1, 4]  # Positions where the categories change
    for boundary in category_boundaries:
        ax.axvline(x=(flat_positions[boundary] + flat_positions[boundary - 1]) / 2, color='grey', linestyle='--', linewidth=1)


    plt.tight_layout()
    fig.subplots_adjust(bottom=0.2)  # Increased bottom margin for legend

    # Ensure the directory exists before saving the plot
    import os
    if not os.path.exists('./figures'):
        os.makedirs('./figures')

    plt.savefig('./figures/order.jpg', dpi=600, bbox_inches='tight', pad_inches=0.2)
    plt.close()

def draw_step_len():
    dataset = "spec_bench"

    # Vicuna-7b
    vicuna_files = [
        "/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/vicuna-7b-v1.3-vanilla-float16-prompt-default.jsonl",
        "/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/vicuna-7b-v1.3-pld-float16.jsonl",
        "/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/vicuna-7b-v1.3-lade-5-win-7-guess-7-float16-prompt-default.jsonl",
        "/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/vicuna-7b-v1.3-rest-float16-prompt-default-temperature-0.0-top_p-0.jsonl",
        "/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/vicuna-7b-v1.3-float16-hierarchy-WSL.jsonl"
    ]

    # Llama-7b
    # llama_files = [
    #     f"/data/smcho/Spec-Bench/data/{dataset}/model_answer/Llama-2-7b-hierachy-level-5-win-7-guess-7-previous-2-order-WSL-float16-prompt-default.jsonl",
    #     f"/data/smcho/Spec-Bench/data/{dataset}/model_answer/Llama-2-7b-lade-5-win-7-guess-7-float16-prompt-default.jsonl",
    #     f"/data/smcho/Spec-Bench/data/{dataset}/model_answer/Llama-2-7b-pld-float16.jsonl",
    #     f"/data/smcho/Spec-Bench/data/{dataset}/model_answer/Llama-2-7b-rest-float16-prompt-default-temperature-0.0-top_k0-top_p-0.jsonl",
    # ]

    def process_files(file_paths):
        results = []
        for jsonl_file in file_paths:
            data = []
            with open(jsonl_file, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
            results.append(data)

        steps = [[], [], [], [], []]
        speeds = [[], [], [], [], []]

        for i, r in enumerate(results):
            for datapoint in r:
                # tokens = sum(datapoint["choices"][0]['new_tokens'])
                times = sum(datapoint["choices"][0]['wall_time'])
                tokens = sum(datapoint["choices"][0]["new_tokens"])
                # tokens = len(datapoint["choices"][0]['accept_lengths'])
                steps[i].append(tokens)
                speeds[i].append(times)
        return steps, speeds

    # Data
    names = ["AR", "PLD", "LADE", "REST", "HD"]
    colors = ['tab:gray', 'tab:blue', 'tab:red', 'tab:green', 'tab:purple']

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 6), dpi=600)
    fig, ax1 = plt.subplots(1, 1, figsize=(5, 3), dpi=600)
    
    def plot_data(ax, steps, speeds, title):
        legend_elements = []
        for step, speed, color, name in zip(steps, speeds, colors, names):
            scatter = ax.scatter(step, speed, c=color, alpha=0.3, s=5)
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label=name,
                                              markerfacecolor=color, markersize=10))
            
            # Calculate and plot trend line (linear fit)
            z = np.polyfit(step, speed, 1)
            p = np.poly1d(z)
            step_sorted = list(range(0,1200))
            ax.plot(step_sorted, p(step_sorted), c=color, linestyle='--', linewidth=1.5)

        ax.set_xlabel('# Tokens', fontsize=14)
        # if title == "Vicuna-7b":
        ax.set_ylabel('Time (sec)', fontsize=14)
        # ax.set_title(title, fontsize=22)
        return legend_elements

    # Plot Vicuna-7b data
    vicuna_steps, vicuna_speeds = process_files(vicuna_files)
    legend_elements = plot_data(ax1, vicuna_steps, vicuna_speeds, "Vicuna-7b")

    # Plot Llama-7b data
    # llama_steps, llama_speeds = process_files(llama_files)
    # plot_data(ax2, llama_steps, llama_speeds, "Llama-7b")
    ax1.set_xlim(0, 1024)
    ax1.set_ylim(0,15)
    # Create legend with only scatter points
    fig.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.07), loc='lower center', 
               fontsize=10, ncol=5)

    # Adjust layout and save the plot
    plt.tight_layout()
    # fig.subplots_adjust(bottom=0.2)  # Increased bottom margin for legend
    plt.savefig('./figures/tokens.jpg', dpi=600, bbox_inches='tight')
    plt.close()

def draw_contour():
    def z_function(x, y):
        length = 179
        return  (length / ( ((length  * (0.0327 + 0.00022342 + x/1000) / y))))
    # Create a grid of x and y values
    y = np.linspace(1.0, 3.0, 1000)
    # x = np.linspace(0.0001, 0.005, 1000)
    x = np.linspace(0.1, 3.0, 1000)
    X, Y = np.meshgrid(x, y)

    # Calculate Z values
    Z = z_function(X, Y)
    # Create the figure and axis objects
    fig, ax = plt.subplots(figsize=(12, 8))

    # Data
    methods = ['HD (Ours)']#, "L", "WL"]
    markers = ['*']#, 'x', 'x']
    colors = ['purple']#, 'red', 'red']

    # Vicuna-7b data
    times = [0.3234,] #11.5644, 3.8646]# 2.31]
    performances = [1.92,] #1.83, 2.03]# 2.21]
    accelerations = [1.71,] #1.18, 1.62] #1.84]

    # Create the filled contour plot
    contourf = ax.contourf(X, Y, Z, levels=40, cmap='viridis', alpha=0.7)

    # Add colorbar to the right of the plot
    cbar = fig.colorbar(contourf, ax=ax, label='Token/Sec')

    # Create contour lines for z values from 1.0 to 2.0 with interval 0.1
    levels = np.arange(20, 100, 2.5)
    contour_lines = ax.contour(X, Y, Z, levels=levels, colors='black', linewidths=0.5)
    # Add labels to the contour lines
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')

    for method, time, performance, marker, color, acceleration in zip(methods, times, performances, markers, colors, accelerations):
        s = 300 if method == 'HD (Ours)' else 250
        ax.scatter(time, performance, marker=marker, color=color, s=s, zorder=5)
        
        weight = 'bold' if method == 'HD (Ours)' else 'normal'
        fontsize = 16 if method == 'HD (Ours)' else 14
        
        # Emphasize coordinates
        coord_text = f"({time:.2f}, {performance:.2f})"
        if method == 'HD (Ours)':
            xytext = (10, 15)
            bbox_props = dict(boxstyle="round,pad=0.3", fc="yellow", ec="b", lw=2, alpha=0.7)
        else:
            xytext = (10, 15)
            bbox_props = dict(boxstyle="round,pad=0.3", fc="lightgreen", ec="g", lw=2, alpha=0.7)
        
        ax.annotate(coord_text, (time, performance), xytext=xytext, textcoords='offset points', 
                    fontsize=fontsize, weight=weight, bbox=bbox_props, zorder=6)
        
        # Add acceleration text
        ax.annotate(f"{acceleration:.2f}x", (time, performance), xytext=(10, -15), 
                    textcoords='offset points', fontsize=fontsize, weight=weight, zorder=6)

    # Set labels and title
    ax.set_xlabel('Draft Latency (ms)', fontsize=20)
    ax.set_ylabel('Mean Accepted Tokens', fontsize=20)
    # ax.set_title('Contour Plot of z = f(x, y) with Specific Contour Lines', fontsize=20)

    # Add gridlines
    ax.grid(True, linestyle='--', alpha=0.5,)

    # Adjust layout to prevent clipping of labels
    plt.tight_layout()

    plt.savefig('./figures/contour3.jpg', dpi=600, bbox_inches='tight')
    plt.close()

def draw_contour2():
    def z_function(x, y):
        length = 179
        return  (length / ( ((length  * (0.0327 + 0.00022342 + x/1000) / y) +  0.00144782)))
    # Create a grid of x and y values
    y = np.linspace(1.0, 3.0, 1000)
    # x = np.linspace(0.0001, 0.005, 1000)
    x = np.linspace(0.1, 1.0, 1000)
    X, Y = np.meshgrid(x, y)

    # Calculate Z values
    Z = z_function(X, Y)
    # Create the figure and axis objects
    fig, ax = plt.subplots(figsize=(12, 8))

    # Data
    methods = ['HD (Ours)', 'Test1']#, "L", "WL"]
    markers = ['*', 'x']#, 'x', 'x']
    colors = ['purple', 'red']#, 'red', 'red']

    # Vicuna-7b data
    times = [0.3234, 0.4457] #11.5644, 3.8646]# 2.31]
    performances = [1.92, 1.93] #1.83, 2.03]# 2.21]
    accelerations = [z_function(time, perform) for time, perform in zip(times, performances)]
    # accelerations = [1.71, 1.51] #1.18, 1.62] #1.84]

    # Create the filled contour plot
    contourf = ax.contourf(X, Y, Z, levels=40, cmap='viridis', alpha=0.7)

    # Add colorbar to the right of the plot
    cbar = fig.colorbar(contourf, ax=ax, label='Token/Sec')

    # Create contour lines for z values from 1.0 to 2.0 with interval 0.1
    levels = np.arange(20, 100, 2.5)
    contour_lines = ax.contour(X, Y, Z, levels=levels, colors='black', linewidths=0.5)
    # Add labels to the contour lines
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')

    for method, time, performance, marker, color, acceleration in zip(methods, times, performances, markers, colors, accelerations):
        s = 300 if method == 'HD (Ours)' else 250
        ax.scatter(time, performance, marker=marker, color=color, s=s, zorder=5)
        
        weight = 'bold' if method == 'HD (Ours)' else 'normal'
        fontsize = 16 if method == 'HD (Ours)' else 14
        
        # Emphasize coordinates
        coord_text = f"({time:.2f}, {performance:.2f})"
        if method == 'HD (Ours)':
            xytext = (10, 15)
            bbox_props = dict(boxstyle="round,pad=0.3", fc="yellow", ec="b", lw=2, alpha=0.7)
        else:
            xytext = (10, 15)
            bbox_props = dict(boxstyle="round,pad=0.3", fc="lightgreen", ec="g", lw=2, alpha=0.7)
        
        # ax.annotate(coord_text, (time, performance), xytext=xytext, textcoords='offset points', 
        #             fontsize=fontsize, weight=weight, bbox=bbox_props, zorder=6)
        
        # Add acceleration text
        ax.annotate(f"{acceleration:.2f}", (time, performance), xytext=(10, -15), 
                    textcoords='offset points', fontsize=fontsize, weight=weight, zorder=6)

    # Set labels and title
    ax.set_xlabel('Draft Latency (ms)', fontsize=20)
    ax.set_ylabel('Mean Accepted Tokens', fontsize=20)
    # ax.set_title('Contour Plot of z = f(x, y) with Specific Contour Lines', fontsize=20)

    # Add gridlines
    ax.grid(True, linestyle='--', alpha=0.5,)

    # Adjust layout to prevent clipping of labels
    plt.tight_layout()

    plt.savefig('./figures/contour_ver2.jpg', dpi=600, bbox_inches='tight')
    plt.close()
    
def draw_order2():

    def analysis_DB(result):
        draft, verify = [], []
        for datapoint in result:
            for data in datapoint["choices"][0]["accept_lengths"]:
                if len(data) == 5:
                    if len(data[1]) > 0:
                        draft.append(1)
                        if data[4] > -1:
                            verify.append(1)
                        else:
                            verify.append(0)
                    else:
                        draft.append(0)
                        verify.append(0)
                elif len(data) == 4:
                    if len(data[1]) > 0:
                        draft.append(1)
                        if data[0] > 1:
                            verify.append(1)
                        else:
                            verify.append(0)
                    else:
                        draft.append(0)
                        verify.append(0)
                else:
                    raise NotImplementedError
        return (draft, verify)

    # model_name = "Llama-2-7b"
    model_name = "vicuna-7b-v1.3"
    vicuna_files = [
        f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/{model_name}-float16-hierarchy.jsonl",
        f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/{model_name}-float16-hierarchy-W.jsonl",
        f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/{model_name}-float16-hierarchy-S.jsonl",
        f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/{model_name}-float16-hierarchy-L.jsonl",
        # f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/{model_name}-float16-hierarchy-WLS.jsonl",
        # f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/{model_name}-float16-hierarchy-SWL.jsonl",
        # f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/{model_name}-float16-hierarchy-SLW.jsonl",
        # f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/{model_name}-float16-hierarchy-LWS.jsonl",
        # f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/{model_name}-float16-hierarchy-LSW.jsonl",
        f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/{model_name}-float16-hierarchy-WS.jsonl",
        f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/{model_name}-float16-hierarchy-WL.jsonl",
        f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/{model_name}-float16-hierarchy-SL.jsonl"
    ]    
    
    acceptance_ratio = [79.52, 58.78, 45.83, 61.82, 75.6, 78.42, 60.85]
    results = [read_data(vicuna_file) for vicuna_file in vicuna_files]
    analysis = [analysis_DB(result) for result in results]
    
    order = (r"$(\mathcal{D}_{c},\mathcal{D}_{m},\mathcal{D}_{s})$",
             r"$\mathcal{D}_{c}$",r"$\mathcal{D}_{m}$",r"$\mathcal{D}_{s}$",
            #  r"$(\mathcal{D}_{c},\mathcal{D}_{s},\mathcal{D}_{m})$",
            #  r"$(\mathcal{D}_{m},\mathcal{D}_{c},\mathcal{D}_{s})$",
            #  r"$(\mathcal{D}_{m},\mathcal{D}_{s},\mathcal{D}_{c})$",
            #  r"$(\mathcal{D}_{s},\mathcal{D}_{c},\mathcal{D}_{m})$",
            #  r"$(\mathcal{D}_{s},\mathcal{D}_{m},\mathcal{D}_{c})$"             
             r"$(\mathcal{D}_{c},\mathcal{D}_{m})$",
             r"$(\mathcal{D}_{c},\mathcal{D}_{s})$",
             r"$(\mathcal{D}_{m},\mathcal{D}_{s})$"
             )

    categories = {
        r"$(\mathcal{D}_{c},\mathcal{D}_{m},\mathcal{D}_{s})$": 0,
        r"$\mathcal{D}_{c}$": 1, r"$\mathcal{D}_{m}$": 1, r"$\mathcal{D}_{s}$": 1,
        # r"$(\mathcal{D}_{c},\mathcal{D}_{s},\mathcal{D}_{m})$":2,
        # r"$(\mathcal{D}_{m},\mathcal{D}_{c},\mathcal{D}_{s})$":2,
        # r"$(\mathcal{D}_{m},\mathcal{D}_{s},\mathcal{D}_{c})$":2,
        # r"$(\mathcal{D}_{s},\mathcal{D}_{c},\mathcal{D}_{m})$":2,
        # r"$(\mathcal{D}_{s},\mathcal{D}_{m},\mathcal{D}_{c})$":2     
        r"$(\mathcal{D}_{c},\mathcal{D}_{m})$" : 2,
        r"$(\mathcal{D}_{c},\mathcal{D}_{s})$" : 2,
        r"$(\mathcal{D}_{m},\mathcal{D}_{s})$" : 2
    }

    positions = {
        0: 1,    # CMS
        1: [1.5, 2, 2.5],  # C, M, S
        2: [3,3.5,4,4.5,5]
    }

    order_positions = [positions[categories[o]] for o in order]
    flat_positions = [item if isinstance(item, int) else item.pop(0) for item in order_positions]

    verify_success = [sum(a[1])*100/len(a[1]) for a in analysis]
    draft_success = [sum(a[0])*100/len(a[0]) - b for a, b in zip(analysis, verify_success)]
    all_fail = [100 - a - b for a, b in zip(verify_success, draft_success)]

    weight_counts = {
        "Draft & Verify Success": np.array(verify_success),
        "Draft Success": np.array(draft_success),
        "Draft Fail": np.array(all_fail)
    }
    width = 0.2

    bottom = np.zeros(len(vicuna_files))

    fig, ax = plt.subplots(figsize=(10, 3), dpi=600)

    hatches = [None, None, None]
    edgecolors = [None, None, None]
    colors = ["#3D3B8E","#5B97EB", "lightgray"]
    # colors = ["#519872", "#A0DDE6", "lightgray"]
    add_bottoms = [0, 0, 0]

    for (label, weight_count), hatch, color, edgecolor, add_bottom in zip(weight_counts.items(), hatches, colors, edgecolors, add_bottoms):
        bars = ax.bar(flat_positions, weight_count, width, label=label, bottom=bottom + add_bottom, hatch=hatch, color=color, edgecolor=edgecolor)
        bottom += weight_count

    line = ax.plot(flat_positions, acceptance_ratio, color="#0D090A", marker='o', linestyle='-', linewidth=2, markersize=6, label='Acceptance Ratio')
    # Add data labels to the line plot
    for x, y in zip(flat_positions, acceptance_ratio):
        if y == 61.82:
            ax.annotate(f'{y:.2f}%', (x, y), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=10, color="#0D090A")
        else:
            ax.annotate(f'{y:.2f}%', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, color="#0D090A")


    ax.set_xticks(flat_positions)
    ax.set_xticklabels(order, fontsize=14)
    plt.ylim(0, 100)
    ax.tick_params(axis='both', which='major', labelsize=12)

    ax.set_xlabel('Exploited Database', fontsize=14)
    ax.set_ylabel('%', fontsize=14)


    draft_color = '#CC2936'
    draft_latency = [2.17, 0.02, 0.03, 12.51, 0.03, 9.49, 2.88]
    ax2 = ax.twinx()
    line = ax2.plot(flat_positions, draft_latency, marker='o', color=draft_color, linestyle='-', linewidth=2, markersize=6, label='Draft Latency')
    ax2.set_ylabel('Draft Latency (ms)', color=draft_color, fontsize=14)
    ax2.tick_params(axis='y', labelcolor=draft_color)
    ax2.set_ylim(0, 15)

    # Add data labels to the line plot
    for x, y in zip(flat_positions, draft_latency):
        if y == 9.49:
            ax2.annotate(f'{y:.2f}ms', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, color=draft_color)
        else:
            ax2.annotate(f'{y:.2f}ms', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, color=draft_color)



    ax.legend(bbox_to_anchor=(0.5,-0.4), loc='lower center', fontsize=10, ncol=5)

    category_boundaries = [1,4]  # Positions where the categories change
    for boundary in category_boundaries:
        ax.axvline(x=(flat_positions[boundary] + flat_positions[boundary - 1]) / 2, color='grey', linestyle='--', linewidth=1)


    plt.tight_layout()
    fig.subplots_adjust(bottom=0.2)  # Increased bottom margin for legend

    # Ensure the directory exists before saving the plot
    import os
    if not os.path.exists('./figures'):
        os.makedirs('./figures')

    plt.savefig('./figures/DB.jpg', dpi=600, bbox_inches='tight', pad_inches=0.2)
    plt.close()

def draw_area_plot():
    
        model_name = "vicuna-7b-v1.3"
        vicuna_files = [
            f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/{model_name}-float16-hierarchy-WSL.jsonl"
        ]
        
        data = read_data(vicuna_files[0])
        total_access = 0
        result = [
            {"Draft & Verify Success": 0, "Draft Success": 0, "Access Count": 0},
            {"Draft & Verify Success": 0, "Draft Success": 0, "Access Count": 0},
            {"Draft & Verify Success": 0, "Draft Success": 0, "Access Count": 0},
        ]
        for datapoint in data:
            steps = datapoint["choices"][0]["accept_lengths"]
            total_access += len(steps)
            for step in steps:
                access = step[1]
                verify = step[-1]
                a = set(access)
                if 2 in a:
                    result[0]["Access Count"] += 1
                    result[1]["Access Count"] += 1
                    result[2]["Access Count"] += 1
                    if 0 in a:
                        result[0]["Draft Success"] += 1
                    if 1 in a:
                        result[1]["Draft Success"] += 1
                    if 2 in a:
                        result[2]["Draft Success"] += 1
                elif 1 in a:
                    result[0]["Access Count"] += 1
                    result[1]["Access Count"] += 1
                    if 0 in a:
                        result[0]["Draft Success"] += 1
                    if 1 in a:
                        result[1]["Draft Success"] += 1
                elif 0 in a:
                    result[0]["Access Count"] += 1
                    if 0 in a:
                        result[0]["Draft Success"] += 1
                if verify >= 0:
                    result[access[verify]]["Draft & Verify Success"] += 1
        for i in result:
            for k,v in i.items():
                i[k] = v * 100 / total_access
            i["Draft Success"] -= i["Draft & Verify Success"]
            i["Draft Fail"] = i["Access Count"] - (i["Draft Success"] + i["Draft & Verify Success"])
            del i["Access Count"]
        names = [r"$\mathcal{D}_{c}$",r"$\mathcal{D}_{m}$",r"$\mathcal{D}_{s}$"]
        colors = ["#3D3B8E","#5B97EB", "lightgray"]
        flat_positions = [1, 1.5, 2]
        bottom = np.zeros(3)
        labels = ["Draft & Verify Success", "Draft Success", "Draft Fail"]
        width = 0.3

        fig, ax = plt.subplots(figsize=(6, 2.5), dpi=600)

        for i in range(3):
            label = labels[i]
            weight_count = np.array([r[label] for r in result])
            color = colors[i]
            bars = ax.bar(flat_positions, weight_count, width, label=label, bottom=bottom, color=color)
            # Add value labels on the bars
            for j, bar in enumerate(bars):
                if label == "Draft Fail" and j == 2:
                    pass
                else:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., bottom[j] + height/2,
                            f'{weight_count[j]:.1f}',
                            ha='center', va='center', fontsize=8, color='white', fontweight='bold')
            bottom += weight_count
            
        ax.set_xticks(flat_positions)
        ax.set_xticklabels(names, fontsize=14)
        plt.ylim(0, 100)
        ax.tick_params(axis='both', which='major', labelsize=12)

        ax.set_xlabel('Accessed Database per Each Step', fontsize=14)
        ax.set_ylabel('%', fontsize=14)

        ax.legend(bbox_to_anchor=(0.5,-0.5), loc='lower center', fontsize=10, ncol=5)

        plt.tight_layout()
        fig.subplots_adjust(bottom=0.2)  # Increased bottom margin for legend

        # Ensure the directory exists before saving the plot
        import os
        if not os.path.exists('./figures'):
            os.makedirs('./figures')

        plt.savefig('./figures/access_pattern_vicuna.jpg', dpi=600, bbox_inches='tight', pad_inches=0.2)
        plt.close()

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from tqdm import tqdm

def draw_pattern():
    def analysis_DB(result, tokenizer):
        keys = dict()
        generation_end = []
        output = []
        for datapoint in tqdm(result[:101]):  # Include result[0] to result[100]
            turn = datapoint["choices"][0]["turns"][0]
            text = tokenizer(turn)["input_ids"][1:]
            for i in range(len(text)-3):
                key = "/".join([str(t) for t in text[i:i+4]])
                if key in keys:
                    output.append(keys[key])
                else:
                    keys[key] = len(keys)
                    output.append(keys[key])
            generation_end.append(len(output))
        return output, generation_end

    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("../../models/Llama-2-7b-chat-hf")
    model_name = "Llama-2-7b"
    vicuna_files = [
        f"/mnt/sda/smcho/Hierarchy_Drafting/data/spec_bench/model_answer/{model_name}-float16-hierarchy.jsonl",
    ]    
    
    results = [read_data(vicuna_file) for vicuna_file in vicuna_files]
    output, generation_end = [analysis_DB(result, tokenizer) for result in results][0]

    # Create figure with custom layout
    fig = plt.figure(figsize=(12, 4), dpi=600)
    gs = GridSpec(3, 20, width_ratios=[1]*20)  # More granular control

    # Main plot
    ax_main = fig.add_subplot(gs[:, :15])
    ax_main.scatter(list(range(len(output))), output, marker="o", color="red", s=0.5)
    ax_main.set_xlabel('Generation Index', fontsize=14)
    ax_main.set_ylabel(r'$4$-gram Index', fontsize=14)
    ax_main.set_title('4-Gram Statistics for 100 Generations', fontsize=16)

    # Set major x-ticks to show generation indices
    major_ticks = np.arange(0, 101, 10)
    major_positions = [0] + [generation_end[i] for i in major_ticks[1:]]
    ax_main.set_xticks(major_positions)
    ax_main.set_xticklabels(major_ticks)

    # Set minor x-ticks for each generation index
    minor_ticks = np.arange(0, 101, 1)
    minor_positions = [0] + [generation_end[i] for i in range(len(minor_ticks))]
    ax_main.set_xticks(minor_positions, minor=True)
    
    # Customize minor tick appearance
    ax_main.tick_params(axis='x', which='minor', length=4, color='gray', width=0.5)
    
    # Add grid for minor ticks
    # ax_main.grid(which='minor', axis='x', linestyle=':', alpha=0.4)


    index = 79
    # Calculate the range for result[79]
    start_index = generation_end[index-1] if len(generation_end) > index-1 else 0
    end_index = generation_end[index] if len(generation_end) > index else len(output)

    # Enlargement plot
    ax_enlarge = fig.add_subplot(gs[1:, 16:])  # Smaller vertical size, closer horizontally
    ax_enlarge.scatter(list(range(start_index, end_index)), output[start_index:end_index], marker="o", color="blue", s=2)
    # ax_enlarge.set_title('79th Generation', fontsize=12)
    # ax_enlarge.set_xlabel('Generation Index', fontsize=10)
    # ax_enlarge.set_ylabel(r'$n$-gram Index', fontsize=10)

    # Set x-ticks for enlarged view
    enlarge_ticks = [index-1, index]
    enlarge_positions = [generation_end[index-1], generation_end[index]]
    ax_enlarge.set_xticks(enlarge_positions)
    ax_enlarge.set_xticklabels(enlarge_ticks)

    # Adjust limits and ticks for better visibility
    ax_enlarge.set_xlim(start_index, end_index)
    y_min, y_max = 19950, 20550
    ax_enlarge.set_ylim(y_min, y_max)
    ax_enlarge.tick_params(axis='both', which='major', labelsize=8)

    # Add zoom effect
    rect = plt.Rectangle((start_index, y_min), end_index - start_index, y_max - y_min, 
                         fill=False, ec="grey", lw=1.5)
    ax_main.add_patch(rect)

    # Connect the zoomed area to the enlargement plot
    mark_inset(ax_main, ax_enlarge, loc1=1, loc2=3, fc="none", ec="grey", lw=1.5)

    plt.tight_layout()
    plt.savefig('./figures/pattern_with_zoom_effect_expanded.jpg', dpi=600, bbox_inches='tight')
    plt.close()
    
def draw_temperature():
    data = {
        "AR" : [55.02, 55.34, 55.45, 54.79, 55.29, 55.03, 55.48, 54.92, 55.23, 54.7],
        "HD" : [80.91, 80.69, 79.42, 78.6, 79.07, 77.98, 77.86, 75.78, 75.07, 73.48],
        "LADE" : [65.17, 64.71, 64.53, 64.41, 64.2, 62.97, 62.85, 62.94, 62.01, 62.0],
        "REST": [65.13, 64.89, 65.14, 65.83, 65.8, 65.59, 65.57, 65.97, 65.39, 65.55]
    }

    colors = {
        "AR" : "tab:gray",
        "HD" : "tab:purple",
        "LADE" : "tab:red",
        "REST" : "tab:green"
    }
    
    fig, ax = plt.subplots(figsize=(6, 3), dpi=600)

    # Plot the data
    for key in data:
        ax.plot(range(1, 11), data[key], color=colors[key], label=key, marker='o')

    ax.set_xlabel('Temperature', fontsize=12)
    ax.set_ylabel('#Tokens/Sec', fontsize=12)

    # Set x-axis ticks and labels
    ax.set_xticks(range(1, 11))
    ax.set_xticklabels([f'{i/10:.1f}' for i in range(1, 11)])

    # Customize the plot
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(bbox_to_anchor=(0.5, -0.6), loc='lower center', fontsize=8, ncol=4)
    plt.tight_layout()
    plt.savefig('./figures/temperature.jpg', dpi=600, bbox_inches='tight', pad_inches=0.2)
    plt.close()

draw_temperature()
# draw_pattern()
# draw_area_plot()
# draw_motivation2()
# draw_motivation()
# draw_specbench()
# draw_overlap()
# draw_step_len()
# draw_order2()
# draw_contour2()