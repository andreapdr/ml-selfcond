export CUDA_VISIBLE_DEVICES=0

# concepts="assets/wemb/concept_list.basic_implicit.csv"
concepts="assets/wemb/concept_list.basic_explicit.csv"

echo "============================================="
echo "[Computing responses with Qwen2.5-7B-Instruct on ${concepts}]"

python scripts/compute_responses.py --model-name-or-path Qwen/Qwen2.5-7B-Instruct --data-path assets/wemb --responses-path wemb_exps --concepts ${concepts} --inf-batch-size 8
