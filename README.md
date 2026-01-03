# SRTA

The **SRTA** framework is used to address security issues in large language models (LLMs) during extended reasoning processes, and achieves dynamic safety alignment and effectively mitigates over-rejection phenomena through sequence-level risk estimation during testing.

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

### 1. Execute the following code to run our SRTA experiment:

**Models**: Llama2_7B_hf/Vicuna_7B_Inst

**data**: GCGVicuna/GCGLlama2/PAIRvicuna/PAIRLlama2/DeepInception/ORbench/XSTest/AutoDANvicuna/AutoDANllama2

```bash
python srta.py --data GCGVicuna --policy Vicuna_7B_Inst --reward LlamaGuard3_1B --judge LlamaGuard3_1B --env PrefillAtk --save_dir outputs/GCG_Vicuna --budget 4
