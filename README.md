# SEAT

The **SEAT** framework is used to address security issues in large language models (LLMs) during extended reasoning processes, and achieves dynamic safety alignment and effectively mitigates over-rejection phenomena through sequence-level risk estimation during testing.

---

## Preparation

You need to download the following models and configure the local paths in the corresponding `inc/models.py` and `inc/data.py` respectively.

* **meta-llama/Llama-2-7b-chat-hf**: (https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
* **vicuna-7b-v1.5**: (https://huggingface.co/lmsys/vicuna-7b-v1.5)
* **meta-llama/Llama-Guard-3-1B**: [https://huggingface.co/meta-llama/Llama-Guard-3-1B](https://huggingface.co/meta-llama/Llama-Guard-3-1B)
* **google/shieldgemma-2b**: [https://huggingface.co/google/shieldgemma-2b](https://huggingface.co/google/shieldgemma-2b)

---

## Structure

* `baseline/` : Stores the baseline code for test-time security alignment methods.
* `data/` : Stores jailbreak attack datasets and neutral datasets.
* `requirements.txt` : Project dependency library list.
* `srta.py` : The main code of the SRTA framework, responsible for starting the core process.

---

## Experiments

We recommend the following process for conducting experiments:

### 1. Execute the following code to run our SEAT experiment:

**Models**: Llama2_7B_hf/Vicuna_7B_Inst

**data**: GCGVicuna/GCGLlama2/PAIRvicuna/PAIRLlama2/DeepInception/ORbench/XSTest/AutoDANvicuna/AutoDANllama2

```bash
python seat.py --data GCGVicuna --policy Vicuna_7B_Inst --reward LlamaGuard3_1B --judge LlamaGuard3_1B --env PrefillAtk --save_dir outputs/GCG_Vicuna --budget 4
```

### 2. Execute the following code to measure the toxicity of the output text:

(Note: The policy model here must be consistent with the policy model used to measure ASR)

```bash
python measure_toxicity.py --pkl_dir /path/to/pkl_files --policy_model /path/to/policy_model --safety_model /path/to/google/shieldgemma-2b --output path/results.json
```

The following is an example of measuring toxicity code:

```bash
python measure_toxicity.py --pkl_dir /data/saffron/outputs/GCG_Vicuna/GCGVicuna~trade024tp0.8/data/saffron/models/Saffron-1-1B~PrefillAtk10~Vicuna_7B_Inst~LlamaGuard3_1B~LlamaGuard3_1B --policy_model path/Vicuna_7B_Inst --safety_model path/shieldgemma-2b --output path/results.json
```
### 3. Calculate the FPR metric：
```bash
python fpr.py
```

### 4. Calculate the LLM safety score.：
```bash
python llm_eval.py
```

### 5. Refer to the following instructions to perform the baseline experiments：
```bash
cd baseline/smooth-llm

python main.py \
    --results_dir ./results/vicuna_GCG \
    --target_model vicuna \
    --attack GCG \
    --attack_logfile /baseline/smooth-llm/data/GCG/AutoDAN_vicuna.json \
    --smoothllm_pert_type RandomSwapPerturbation \
    --smoothllm_pert_pct 10 \
    --smoothllm_num_copies 5
```

```bash
cd baseline/Gradient-Cuff

python main.py --model_id /path_to/vicuna-7b-v1.5 --user_query_path /baseline/Gradient-Cuff/attack_data/vicuna_AutoDAN.json --save_path outputs/vicuna_AutoDAN.json --batch_size 20 --device cuda:0
```
