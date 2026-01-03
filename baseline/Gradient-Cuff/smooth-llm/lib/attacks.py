import json
import pandas as pd

class Prompt:
    def __init__(self, full_prompt, perturbable_prompt, max_new_tokens):
        self.full_prompt = full_prompt
        self.perturbable_prompt = perturbable_prompt
        self.max_new_tokens = max_new_tokens

    def perturb(self, perturbation_fn):
        perturbed_prompt = perturbation_fn(self.perturbable_prompt)
        self.full_prompt = self.full_prompt.replace(
            self.perturbable_prompt,
            perturbed_prompt
        )
        self.perturbable_prompt = perturbed_prompt

class Attack:
    def __init__(self, logfile, target_model):
        self.logfile = logfile
        self.target_model = target_model


class GCG(Attack):
    """Greedy Coordinate Gradient attack.

    Title: Universal and Transferable Adversarial Attacks on 
                Aligned Language Models
    Authors: Andy Zou, Zifan Wang, J. Zico Kolter, Matt Fredrikson
    Paper: https://arxiv.org/abs/2307.15043
    """

    def __init__(self, logfile, target_model):
        super(GCG, self).__init__(logfile, target_model)

        with open(self.logfile, 'r') as f:
            log = json.load(f)

        # 仅保留 goal 和 target，移除 controls
        self.goals = log['goal']
        self.targets = log['target']

        # 基于 goal 和 target 生成提示（不再依赖 controls）
        self.prompts = [
            self.create_prompt(g, t)  # 仅传入 goal 和 target
            for (g, t) in zip(self.goals, self.targets)
        ]

    def create_prompt(self, goal, target, max_new_len=100):
        """仅使用 goal 和 target 创建提示（移除 control 依赖）"""

        # 保留基于 target 计算最大生成 token 数的逻辑
        max_new_tokens = max(
            len(self.target_model.tokenizer(target).input_ids) + 2,
            max_new_len
        )

        # 创建符合模型对话格式的完整提示（仅使用 goal 作为用户输入）
        conv_template = self.target_model.conv_template
        conv_template.append_message(
            conv_template.roles[0], goal  # 仅传入 goal，不拼接 control
        )
        conv_template.append_message(conv_template.roles[1], "")  # 预留助手回复位置
        prompt = conv_template.get_prompt()

        # 标准化提示格式（移除特殊标记）
        encoding = self.target_model.tokenizer(prompt)
        full_prompt = self.target_model.tokenizer.decode(
            encoding.input_ids
        ).replace('<s>', '').replace('</s>', '')

        # 清空对话模板，避免影响后续生成
        conv_template.messages = []

        # 可扰动部分仅为 goal 对应的文本（不再包含 control）
        start_index = full_prompt.find(goal)
        end_index = start_index + len(goal)  # 以 goal 结尾作为边界
        perturbable_prompt = full_prompt[start_index:end_index]

        return Prompt(
            full_prompt,
            perturbable_prompt,
            max_new_tokens
        )
class PAIR(Attack):

    """Prompt Automatic Iterative Refinement (PAIR) attack.

    Title: Jailbreaking Black Box Large Language Models in Twenty Queries
    Authors: Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, 
                George J. Pappas, Eric Wong
    Paper: https://arxiv.org/abs/2310.08419
    """

    def __init__(self, logfile, target_model):
        super(PAIR, self).__init__(logfile, target_model)

        df = pd.read_pickle(logfile)
        jailbreak_prompts = df['jailbreak_prompt'].to_list()
        
        self.prompts = [
            self.create_prompt(prompt)
            for prompt in jailbreak_prompts
        ]
        
    def create_prompt(self, prompt):

        conv_template = self.target_model.conv_template
        conv_template.append_message(conv_template.roles[0], prompt)
        conv_template.append_message(conv_template.roles[1], None)
        full_prompt = conv_template.get_prompt()

        # Clear the conv template
        conv_template.messages = []

        return Prompt(
            full_prompt,
            prompt,
            max_new_tokens=100
        )