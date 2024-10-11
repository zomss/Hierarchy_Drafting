# Hierarchy Drafting for Speculative Decoding
Official Code Repository for the paper "Lossless Acceleration of Large Language Models with Hierarchical Drafting based on Temporal Locality in Speculative Decoding"

As the paper is under review, we anonymously released the code.

## Installation
The first step of installation is to create a conda environment as follows:
<pre>
<code>
$ conda create -n HD python=3.10
$ pip install -r requirements.txt</code>
</pre>

Then, we exploit Python Library, DraftRetriever designed by [REST](https://github.com/FasterDecoding/REST/tree/main?tab=readme-ov-file), for the statistics-dependent database. DraftRetriever can be installed in [here](https://github.com/FasterDecoding/REST/tree/main/DraftRetriever).

## Construct Model-dependent DB
As model-dependent DB is based on previously generated texts by target LLM, such texts can be generated as follows:
<pre>
<code>
$ python ./scripts/construct_model_DB --model_dir {save_path} --model {target LLM}</code>
</pre>

Currently, we only support Llama-2 and Vicuna-v1.3. We will update the code to support more models and provide the data for generated texts.

## Run Hierarchy Drafting
You can run the hierarchy drafting as follows:
<pre>
<code>
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} USE_LADE=1 python -m evaluation.inference_hierachy --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-${torch_dtype}-hierarchy --level $LEVEL --window $WINDOW --guess $GUESS --previous_tokens $PREVIOUS_TOKENS --bench-name $bench_NAME --dtype $torch_dtype --history_file {history_file} --db_file=${db_file} --do_WM --do_SM --do_LM --order $ORDER</code>
</pre>

Also, you can easily run Hierarchy Drafting and other baselines in the *run_llama.sh* and *run_vicuna.sh*.

## Acknowledgement
The implementation of Hierarchy Drafting is from [REST](https://github.com/FasterDecoding/REST/tree/main?tab=readme-ov-file), [LADE](https://github.com/hao-ai-lab/LookaheadDecoding), and [Spec-Bench](https://github.com/hemingkx/Spec-Bench?tab=readme-ov-file).

