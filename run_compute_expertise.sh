#!/bin/bash

# concepts="wemb_exps/concept_list.basic_explicit.small.csv"
concepts="wemb_exps/concept_list.basic_implicit.small.csv"

echo "============================================="
echo "[Computing expertise with Qwen2.5-7B-Instruct on ${concepts}]"

python scripts/compute_expertise.py --model-name Qwen/Qwen2.5-7B-Instruct --root-dir wemb_exps --concepts ${concepts}
