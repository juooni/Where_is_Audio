import subprocess
import os

command='torchrun --nproc_per_node 1 ./llama/example_text_completion.py     --ckpt_dir ./llama/llama-2-7b/     --tokenizer_path ./llama/tokenizer.model     --max_seq_len 512 --max_batch_size 6'
os.system(command)
