# Launch Megatron-LM Multi-node Training

Here is a summary of launching steps for multi-node training using Megatron-LM. We use GPT model as an example because it is well officially supported by Megatron-LM.

## Setup

Nvidia provides [PyTorch NGC Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) and recommends to deploy Megatron-LM with Docker. 

<pre>
git checkout core_r0.12.0

docker pull nvcr.io/nvidia/pytorch:25.03-py3
# use the container version in the previous month of megatron release

docker run --gpus all --ipc=host -it --rm \
  -v </path/to/megatron>:/workspace/megatron \
  -v </path/to/dataset>:/workspace/dataset \
  -v </path/to/checkpoints>:/workspace/checkpoints \
  nvcr.io/nvidia/pytorch:25.03-py3
</pre>
Note: use `--net host` may cause errors in NCCL.

## Data Preprocessing

Training data should be placed in a loose json format, with one json containing a text sample per line. For example:
<pre>
{"src": "www.nvidia.com", "text": "The quick brown fox", "type": "Eng", "id": "0", "title": "First Part"}
{"src": "The Internet", "text": "jumps over the lazy dog", "type": "Eng", "id": "42", "title": "Second Part"}
</pre>

For Wikipedia dataset, use the Wikipedia data extraction process specified by Google research: "the recommended pre-processing is to download [the latest dump](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2), extract the text with [WikiExtractor.py](https://github.com/attardi/wikiextractor), and then apply any necessary cleanup to convert it into plain text." It's recomended to use `--json` to generate loose json format data and use nltk punctuation standardization for further process.

After that, Megatron-LM provides a script to preprocess the data.
<pre>
python tools/preprocess_data.py \
       --input my-corpus.json \
       --output-prefix my-gpt2 \
       --vocab-file gpt2-vocab.json \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file gpt2-merges.txt \
       --append-eod
</pre>
Here the output files are named `my-gpt2_text_document.bin` and `my-gpt2_text_document.idx`. The `--data-path` specified in later training is new filename without file extension.

## Pretraining

The `examples/gpt3/train_gpt3_175b_distributed.sh` script runs 175B parameter GPT-3 pretraining on 8 GPU in a single node, and can simply modify to run on multi-nodes by:
<pre>
DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --node_rank $NODE_RANK    # only this differs on each node
    --master_addr $MASTER_ADDR  # IP address of the node 0
    --master_port $MASTER_PORT
)
</pre>
These arguments are passed to `torchrun` which lauches the distributed training. It then automatically sets up environment variables `LOCAL_RANK`, `GROUP_RANK` and `RANK` (global rank) for each worker to identify itself.

The parallelism can be set by:
<pre>
MODEL_PARALLEL_ARGS=(
    # data parallelism
    --overlap-grad-reduce # overlapping of the gradient reduction with the backward pass

    # model parallelism
    --tensor-model-parallel-size 8  # the number of GPUs among which to split the model
    --sequence-parallel # requires TP, further splits across the same GPUs

    # pipeline parallelism
    --pipeline-model-parallel-size 16 # the number of stages to split the model

    --use-distributed-optimizer # distribute optimizer states across data parallel ranks like ZERO
)
</pre>

## LLaMA
Megatron-LM allows to convert LLaMA's Meta/HuggingFace checkpoints into Megatron format for inference/finetuning, but it does not support LLaMA training officially. [This post](https://zhuanlan.zhihu.com/p/668057319) gives a list of self-defined parameters to realize LLaMA-2 based on GPT script. In addition, Alibaba open-sources [Megatron-LLaMA](https://github.com/alibaba/Megatron-LLaMA) project with some modifications from the original Megatron-LM.