Vicuna_PATH=../../models/Llama-2-7b-chat-hf
# # Eagle_PATH=../models/EAGLE-Vicuna-7B-v1.3
# # Medusa_PATH=../models/medusa-vicuna-7b-v1.3
# # Hydra_PATH=../models/hydra-vicuna-7b-v1.3
# # Drafter_PATH=../models/vicuna-68m
# # Space_PATH=../models/vicuna-v1.3-7b-space
MODEL_NAME=Llama-2-7b

GPU_DEVICES=0

# bench_NAME="Alpaca" #["OASST", "GSM8K"]
torch_dtype="float16" # ["float32", "float64", "float16", "bfloat16"]

ORDER="WSL"
LEVEL=5
WINDOW=7
GUESS=7
PREVIOUS_TOKENS=2

prompt=default
datastore_PATH=./datastore
db=/db/datastore_chat_large.idx
history=/history/OASST/Llama-2-7b/${prompt}/train_5.json

bench_NAME="spec_bench"

# ORDER="W"
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} USE_LADE=1 python -m evaluation.inference_hierachy --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-${torch_dtype}-hierarchy-${ORDER} --level $LEVEL --window $WINDOW --guess $GUESS --previous_tokens $PREVIOUS_TOKENS --bench-name $bench_NAME --dtype $torch_dtype --history_file ${datastore_PATH}${history} --db_file=${datastore_PATH}${db} --do_WM --do_SM --do_LM --order $ORDER 
# ORDER="S"
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} USE_LADE=1 python -m evaluation.inference_hierachy --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-${torch_dtype}-hierarchy-${ORDER} --level $LEVEL --window $WINDOW --guess $GUESS --previous_tokens $PREVIOUS_TOKENS --bench-name $bench_NAME --dtype $torch_dtype --history_file ${datastore_PATH}${history} --db_file=${datastore_PATH}${db} --do_WM --do_SM --do_LM --order $ORDER 
# ORDER="L"
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} USE_LADE=1 python -m evaluation.inference_hierachy --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-${torch_dtype}-hierarchy-${ORDER} --level $LEVEL --window $WINDOW --guess $GUESS --previous_tokens $PREVIOUS_TOKENS --bench-name $bench_NAME --dtype $torch_dtype --history_file ${datastore_PATH}${history} --db_file=${datastore_PATH}${db} --do_WM --do_SM --do_LM --order $ORDER 
# ORDER="WS"
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} USE_LADE=1 python -m evaluation.inference_hierachy --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-${torch_dtype}-hierarchy-${ORDER} --level $LEVEL --window $WINDOW --guess $GUESS --previous_tokens $PREVIOUS_TOKENS --bench-name $bench_NAME --dtype $torch_dtype --history_file ${datastore_PATH}${history} --db_file=${datastore_PATH}${db} --do_WM --do_SM --do_LM --order $ORDER 
# ORDER="WL"
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} USE_LADE=1 python -m evaluation.inference_hierachy --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-${torch_dtype}-hierarchy-${ORDER} --level $LEVEL --window $WINDOW --guess $GUESS --previous_tokens $PREVIOUS_TOKENS --bench-name $bench_NAME --dtype $torch_dtype --history_file ${datastore_PATH}${history} --db_file=${datastore_PATH}${db} --do_WM --do_SM --do_LM --order $ORDER 
# ORDER="SL"
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} USE_LADE=1 python -m evaluation.inference_hierachy --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-${torch_dtype}-hierarchy-${ORDER} --level $LEVEL --window $WINDOW --guess $GUESS --previous_tokens $PREVIOUS_TOKENS --bench-name $bench_NAME --dtype $torch_dtype --history_file ${datastore_PATH}${history} --db_file=${datastore_PATH}${db} --do_WM --do_SM --do_LM --order $ORDER 


# ORDER=W
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} USE_LADE=1 python -m evaluation.inference_hierachy --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-${torch_dtype}-hierarchy-${ORDER} --level $LEVEL --window $WINDOW --guess $GUESS --previous_tokens $PREVIOUS_TOKENS --bench-name $bench_NAME --dtype $torch_dtype --history_file ${datastore_PATH}${history} --db_file=${datastore_PATH}${db} --do_WM --do_SM --do_LM --order $ORDER 
# ORDER=S
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} USE_LADE=1 python -m evaluation.inference_hierachy --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-${torch_dtype}-hierarchy-${ORDER} --level $LEVEL --window $WINDOW --guess $GUESS --previous_tokens $PREVIOUS_TOKENS --bench-name $bench_NAME --dtype $torch_dtype --history_file ${datastore_PATH}${history} --db_file=${datastore_PATH}${db} --do_WM --do_SM --do_LM --order $ORDER 
# ORDER=L
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} USE_LADE=1 python -m evaluation.inference_hierachy --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-${torch_dtype}-hierarchy-${ORDER} --level $LEVEL --window $WINDOW --guess $GUESS --previous_tokens $PREVIOUS_TOKENS --bench-name $bench_NAME --dtype $torch_dtype --history_file ${datastore_PATH}${history} --db_file=${datastore_PATH}${db} --do_WM --do_SM --do_LM --order $ORDER 
# ORDER=WS
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} USE_LADE=1 python -m evaluation.inference_hierachy --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-${torch_dtype}-hierarchy-${ORDER} --level $LEVEL --window $WINDOW --guess $GUESS --previous_tokens $PREVIOUS_TOKENS --bench-name $bench_NAME --dtype $torch_dtype --history_file ${datastore_PATH}${history} --db_file=${datastore_PATH}${db} --do_WM --do_SM --do_LM --order $ORDER 
# ORDER=WL
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} USE_LADE=1 python -m evaluation.inference_hierachy --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-${torch_dtype}-hierarchy-${ORDER} --level $LEVEL --window $WINDOW --guess $GUESS --previous_tokens $PREVIOUS_TOKENS --bench-name $bench_NAME --dtype $torch_dtype --history_file ${datastore_PATH}${history} --db_file=${datastore_PATH}${db} --do_WM --do_SM --do_LM --order $ORDER 
# ORDER=SW
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} USE_LADE=1 python -m evaluation.inference_hierachy --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-${torch_dtype}-hierarchy-${ORDER} --level $LEVEL --window $WINDOW --guess $GUESS --previous_tokens $PREVIOUS_TOKENS --bench-name $bench_NAME --dtype $torch_dtype --history_file ${datastore_PATH}${history} --db_file=${datastore_PATH}${db} --do_WM --do_SM --do_LM --order $ORDER 
# ORDER=SL
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} USE_LADE=1 python -m evaluation.inference_hierachy --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-${torch_dtype}-hierarchy-${ORDER} --level $LEVEL --window $WINDOW --guess $GUESS --previous_tokens $PREVIOUS_TOKENS --bench-name $bench_NAME --dtype $torch_dtype --history_file ${datastore_PATH}${history} --db_file=${datastore_PATH}${db} --do_WM --do_SM --do_LM --order $ORDER 
# ORDER=LW
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} USE_LADE=1 python -m evaluation.inference_hierachy --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-${torch_dtype}-hierarchy-${ORDER} --level $LEVEL --window $WINDOW --guess $GUESS --previous_tokens $PREVIOUS_TOKENS --bench-name $bench_NAME --dtype $torch_dtype --history_file ${datastore_PATH}${history} --db_file=${datastore_PATH}${db} --do_WM --do_SM --do_LM --order $ORDER 
# ORDER=LS
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} USE_LADE=1 python -m evaluation.inference_hierachy --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-${torch_dtype}-hierarchy-${ORDER} --level $LEVEL --window $WINDOW --guess $GUESS --previous_tokens $PREVIOUS_TOKENS --bench-name $bench_NAME --dtype $torch_dtype --history_file ${datastore_PATH}${history} --db_file=${datastore_PATH}${db} --do_WM --do_SM --do_LM --order $ORDER 
# ORDER=WSL
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} USE_LADE=1 python -m evaluation.inference_hierachy --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-${torch_dtype}-hierarchy-${ORDER} --level $LEVEL --window $WINDOW --guess $GUESS --previous_tokens $PREVIOUS_TOKENS --bench-name $bench_NAME --dtype $torch_dtype --history_file ${datastore_PATH}${history} --db_file=${datastore_PATH}${db} --do_WM --do_SM --do_LM --order $ORDER 

# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_baseline --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-vanilla-${torch_dtype}-prompt-${prompt} --bench-name $bench_NAME --dtype $torch_dtype
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_pld --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-pld-${torch_dtype} --bench-name $bench_NAME --dtype $torch_dtype 
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} RAYON_NUM_THREADS=6 python -m evaluation.inference_rest --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-rest-${torch_dtype}-prompt-${prompt} --datastore-path ${datastore_PATH}${db} --bench-name $bench_NAME --dtype $torch_dtype 
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} USE_LADE=1 python -m evaluation.inference_lookahead --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-lade-${LEVEL}-win-${WINDOW}-guess-${GUESS}-${torch_dtype}-prompt-${prompt} --level $LEVEL --window $WINDOW --guess $GUESS --bench-name $bench_NAME --dtype $torch_dtype 
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} USE_LADE=1 python -m evaluation.inference_hierachy --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-${torch_dtype}-hierarchy --level $LEVEL --window $WINDOW --guess $GUESS --previous_tokens $PREVIOUS_TOKENS --bench-name $bench_NAME --dtype $torch_dtype --history_file ${datastore_PATH}${history} --db_file=${datastore_PATH}${db} --do_WM --do_SM --do_LM --order $ORDER 

# Vicuna_PATH=../../models/Llama-2-13b-chat-hf
# MODEL_NAME=Llama-2-13b
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_baseline --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-vanilla-${torch_dtype}-prompt-${prompt} --bench-name $bench_NAME --dtype $torch_dtype
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_pld --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-pld-${torch_dtype} --bench-name $bench_NAME --dtype $torch_dtype
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} RAYON_NUM_THREADS=6 python -m evaluation.inference_rest --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-rest-${torch_dtype}-prompt-${prompt} --datastore-path ${datastore_PATH}${db} --bench-name $bench_NAME --dtype $torch_dtype
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} USE_LADE=1 python -m evaluation.inference_lookahead --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-lade-${LEVEL}-win-${WINDOW}-guess-${GUESS}-${torch_dtype}-prompt-${prompt} --level $LEVEL --window $WINDOW --guess $GUESS --bench-name $bench_NAME --dtype $torch_dtype
# ORDER="WSL"
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} USE_LADE=1 python -m evaluation.inference_hierachy --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-${torch_dtype}-hierarchy --level $LEVEL --window $WINDOW --guess $GUESS --previous_tokens $PREVIOUS_TOKENS --bench-name $bench_NAME --dtype $torch_dtype --history_file ${datastore_PATH}${history} --db_file=${datastore_PATH}${db} --do_WM --do_SM --do_LM --order $ORDER 

# TEMP=1.0
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_baseline --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-vanilla-${torch_dtype}-prompt-${prompt}-temp-${TEMP} --bench-name $bench_NAME --dtype $torch_dtype --temperature $TEMP
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} USE_LADE=1 python -m evaluation.inference_lookahead --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-lade-${LEVEL}-win-${WINDOW}-guess-${GUESS}-${torch_dtype}-prompt-${prompt}-temp-${TEMP}  --level $LEVEL --window $WINDOW --guess $GUESS --bench-name $bench_NAME --dtype $torch_dtype --temperature $TEMP --do_sample
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} RAYON_NUM_THREADS=6 python -m evaluation.inference_rest --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-rest-${torch_dtype}-prompt-${prompt} --datastore-path ${datastore_PATH}${db} --bench-name $bench_NAME --dtype $torch_dtype --temperature $TEMP
# ORDER="WSL"
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} USE_LADE=1 python -m evaluation.inference_hierachy --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-${torch_dtype}-hierarchy-temp-${TEMP}-${ORDER}  --level $LEVEL --window $WINDOW --guess $GUESS --previous_tokens $PREVIOUS_TOKENS --bench-name $bench_NAME --dtype $torch_dtype --history_file ${datastore_PATH}${history} --db_file=${datastore_PATH}${db} --do_WM --do_SM --do_LM --order $ORDER --do_sample --temperature $TEMP

# Vicuna_PATH=../../models/vicuna-13b-v1.3
# MODEL_NAME=vicuna-13b-v1.3
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_baseline --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-vanilla-${torch_dtype}-prompt-${prompt}-temp-${TEMP} --bench-name $bench_NAME --dtype $torch_dtype --temperature $TEMP
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} USE_LADE=1 python -m evaluation.inference_lookahead --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-lade-${LEVEL}-win-${WINDOW}-guess-${GUESS}-${torch_dtype}-prompt-${prompt}-temp-${TEMP}  --level $LEVEL --window $WINDOW --guess $GUESS --bench-name $bench_NAME --dtype $torch_dtype --temperature $TEMP --do_sample
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} RAYON_NUM_THREADS=6 python -m evaluation.inference_rest --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-rest-${torch_dtype}-prompt-${prompt} --datastore-path ${datastore_PATH}${db} --bench-name $bench_NAME --dtype $torch_dtype --temperature $TEMP
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} USE_LADE=1 python -m evaluation.inference_hierachy --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-${torch_dtype}-hierarchy-temp-${TEMP}  --level $LEVEL --window $WINDOW --guess $GUESS --previous_tokens $PREVIOUS_TOKENS --bench-name $bench_NAME --dtype $torch_dtype --history_file ${datastore_PATH}${history} --db_file=${datastore_PATH}${db} --do_WM --do_SM --do_LM --order $ORDER --do_sample --temperature $TEMP
