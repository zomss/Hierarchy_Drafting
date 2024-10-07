import os, json
from collections import Counter
from tqdm import tqdm

LEVEL=5
num_history = 10000
history_path = "/home/tjrals/Hierarchy_Drafting/data/OASST/vicuna-7b/default/train.json"

with open(history_path, 'r') as f: results = json.load(f)

# result = []
# with open(os.path.join(history_path, "history.jsonl"), 'r') as f:
#     for line in f:
#         results.append(json.loads(line))

history = Counter()
for d in tqdm(results[:num_history]):
    tokens = [str(token) for token in d["tokens"]]
    ["/".join(tokens[i:i+LEVEL]) for i in range(0, len(tokens)-LEVEL)]
    history += Counter(list(set(["/".join(tokens[i:i+LEVEL]) for i in range(0, len(tokens)-LEVEL)])))
os.makedirs("/home/tjrals/Hierarchy_Drafting/datastore/history/OASST/vicuna-7b-v1.3/default", exist_ok=True)
with open(os.path.join("/home/tjrals/Hierarchy_Drafting/datastore/history/OASST/vicuna-7b-v1.3/default", "train_{}_{}.json".format(LEVEL, num_history)), 'w') as f: json.dump(dict(history), f)