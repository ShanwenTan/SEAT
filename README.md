# SRTA

The **SRTA** framework is used to address security issues in large language models (LLMs) during extended reasoning processes, and achieves dynamic safety alignment and effectively mitigates over-rejection phenomena through sequence-level risk estimation during testing.

---

## Preparation

You need to download the following models and configure the local paths in the corresponding `inc/models.py` and `inc/data.py` respectively.

* **meta-llama/Meta-Llama-3-8B-Instruct**: [https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
* **mistralai/Mistral-7B-Instruct-v0.3**: [https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
* **ModelCloud/Llama3.2-1B-Instruct**: [https://huggingface.co/ModelCloud/Llama3.2-1B-Instruct](https://huggingface.co/ModelCloud/Llama3.2-1B-Instruct)
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

```bash
python srta.py --data (AdvBench/HarmfulHExPHI/JBBBehaviors) --policy (Llama3_8B_Inst/Llama3_2_1B_Instruct)
