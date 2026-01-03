from inc.utils import *
import pandas as pd
from datasets import load_dataset


class DataPair(NamedTuple):
    prompt: str
    output: str

class Data:
    @staticmethod
    def add_args(parser: ArgumentParser):
        pass

    @property
    def name(self) -> str:
        return type(self).__name__

    @property
    def len(self) -> int:
        return len(self.pairs)

    def __init__(self, args: Dict, pairs: List[Tuple[str, str]]):
        self.pairs = tuple(DataPair(prompt = prompt, output = output) for prompt, output in pairs)

DATA = dict()

@register_cls(DATA)
class HarmfulHExPHI(Data):
    def __init__(self, args: Dict):
        # 直接指定本地文件路径，移除远程下载逻辑
        data_path = '/data/saffron/shallow-vs-deep-safety-alignment-dataset/data/safety_bench/Harmful-HEx-PHI.jsonl'  # 替换为实际本地路径
        print('Using local dataset from', data_path)

        dataset = []
        with open(data_path, 'r') as fi:
            for line in fi:
                dataset.append(json.loads(line))
        super().__init__(args=args, pairs=[(chat[0]['content'], chat[1]['content']) for chat in dataset])


@register_cls(DATA)
class Ai2Refusals(Data):
    """
    从本地加载allenai/reward-bench数据集中的拒绝生成子数据集，并将其转换为模型训练或评估所需的格式
    """

    def __init__(self, args: Dict):
        # 直接指定本地文件路径（假设为JSONL格式）
        data_path = 'path/to/your/local/refusals_dataset.jsonl'  # 替换为实际本地路径

        # 读取本地JSONL文件
        dataset = []
        with open(data_path, 'r') as f:
            for line in f:
                dataset.append(json.loads(line))

        # 过滤数据（保持与远程加载相同的逻辑）
        filtered_dataset = [row for row in dataset if row['subset'].startswith('refusals-')]

        super().__init__(args=args, pairs=[(row['prompt'], row['rejected']) for row in filtered_dataset])


from inc.utils import *
from datasets import load_dataset  # 利用datasets库处理Parquet格式，更稳定

from datasets import load_dataset  # 确保导入正确
import os
from typing import Dict

@register_cls(DATA)
class AdvBench(Data):
    """加载本地JBB-Behaviors数据集（CSV格式）"""

    def __init__(self, args: Dict):
        # 硬编码本地CSV文件路径，使用上传文件的路径
        data_path = "/data/saffron/AdvBench/data/harmful_behaviors01.csv"
        print(f"加载本地AdvBench数据集: {data_path}")

        # 检查文件是否存在
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据集文件不存在，请检查路径: {data_path}")

        try:
            # 使用pandas读取CSV文件（修改为read_csv）
            df = pd.read_csv(data_path)
        except Exception as e:
            raise RuntimeError(f"读取CSV文件失败: {str(e)}")

        # 验证必要字段（根据实际CSV列名调整，这里假设为'goal'和'target'）
        required_columns = ["goal", "target"]
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"CSV文件缺少必要列: {missing}，请检查数据集格式")

        # 提取并过滤有效数据（去除空值）
        pairs = []
        for _, row in df.iterrows():
            prompt = str(row["goal"]).strip()
            output = str(row["target"]).strip()
            if prompt and output:  # 跳过空字符串
                pairs.append((prompt, output))

        print(f"成功加载{len(pairs)}条AdvBench数据（原始样本数：{len(df)}，过滤无效数据后）")
        super().__init__(args=args, pairs=pairs)

    @staticmethod
    def add_args(parser: ArgumentParser):
        pass  # 路径已硬编码，无需命令行参数


from inc.utils import *
import pandas as pd
import os

# 保持原有DataPair和Data类定义不变...

@register_cls(DATA)
class JBBBehaviors(Data):
    """加载本地JBB-Behaviors数据集（CSV格式）"""

    def __init__(self, args: Dict):
        # 硬编码本地CSV文件路径，使用上传文件的路径
        data_path = "/data/saffron/JBB-Behaviors/data/harmful-behaviors.csv"
        print(f"加载本地JBB-Behaviors数据集: {data_path}")

        # 检查文件是否存在
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据集文件不存在，请检查路径: {data_path}")

        try:
            # 使用pandas读取CSV文件（修改为read_csv）
            df = pd.read_csv(data_path)
        except Exception as e:
            raise RuntimeError(f"读取CSV文件失败: {str(e)}")

        # 验证必要字段（根据实际CSV列名调整，这里假设为'goal'和'target'）
        required_columns = ["Goal", "Target"]
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"CSV文件缺少必要列: {missing}，请检查数据集格式")

        # 提取并过滤有效数据（去除空值）
        pairs = []
        for _, row in df.iterrows():
            prompt = str(row["Goal"]).strip()
            output = str(row["Target"]).strip()
            if prompt and output:  # 跳过空字符串
                pairs.append((prompt, output))

        print(f"成功加载{len(pairs)}条JBB-Behaviors数据（原始样本数：{len(df)}，过滤无效数据后）")
        super().__init__(args=args, pairs=pairs)

    @staticmethod
    def add_args(parser: ArgumentParser):
        pass  # 路径已硬编码，无需命令行参数


@register_cls(DATA)
class JBBBehaviors_good(Data):
    """加载本地JBB-Behaviors数据集（CSV格式）"""

    def __init__(self, args: Dict):
        # 硬编码本地CSV文件路径，使用上传文件的路径
        data_path = "/data/saffron/JBB-Behaviors/data/benign-behaviors.csv"
        print(f"加载本地JBB-Behaviors数据集的良性样本: {data_path}")

        # 检查文件是否存在
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据集文件不存在，请检查路径: {data_path}")

        try:
            # 使用pandas读取CSV文件（修改为read_csv）
            df = pd.read_csv(data_path)
        except Exception as e:
            raise RuntimeError(f"读取CSV文件失败: {str(e)}")

        # 验证必要字段（根据实际CSV列名调整，这里假设为'goal'和'target'）
        required_columns = ["Goal", "Target"]
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"CSV文件缺少必要列: {missing}，请检查数据集格式")

        # 提取并过滤有效数据（去除空值）
        pairs = []
        for _, row in df.iterrows():
            prompt = str(row["Goal"]).strip()
            output = str(row["Target"]).strip()
            if prompt and output:  # 跳过空字符串
                pairs.append((prompt, output))

        print(f"成功加载{len(pairs)}条JBB-Behaviors数据（原始样本数：{len(df)}，过滤无效数据后）")
        super().__init__(args=args, pairs=pairs)

    @staticmethod
    def add_args(parser: ArgumentParser):
        pass  # 路径已硬编码，无需命令行参数

@register_cls(DATA)
class JBBBehaviors_GCG_1B(Data):
    """加载本地JBB-Behaviors数据集（CSV格式）"""

    def __init__(self, args: Dict):
        # 硬编码本地CSV文件路径，使用上传文件的路径
        data_path = "/data/saffron/new_gcg_JBB_Llama_1B.csv"
        print(f"加载本地JBB-Behaviors数据集: {data_path}")

        # 检查文件是否存在
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据集文件不存在，请检查路径: {data_path}")

        try:
            # 使用pandas读取CSV文件（修改为read_csv）
            df = pd.read_csv(data_path)
        except Exception as e:
            raise RuntimeError(f"读取CSV文件失败: {str(e)}")

        # 验证必要字段（根据实际CSV列名调整，这里假设为'goal'和'target'）
        required_columns = ["gcg_prompt", "Target"]
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"CSV文件缺少必要列: {missing}，请检查数据集格式")

        # 提取并过滤有效数据（去除空值）
        pairs = []
        for _, row in df.iterrows():
            prompt = str(row["gcg_prompt"]).strip()
            output = str(row["Target"]).strip()
            if prompt and output:  # 跳过空字符串
                pairs.append((prompt, output))

        print(f"成功加载{len(pairs)}条JBB-Behaviors数据（原始样本数：{len(df)}，过滤无效数据后）")
        super().__init__(args=args, pairs=pairs)

    @staticmethod
    def add_args(parser: ArgumentParser):
        pass  # 路径已硬编码，无需命令行参数

@register_cls(DATA)
class JBBBehaviors_GCG_8B(Data):
    """加载本地JBB-Behaviors数据集（CSV格式）"""

    def __init__(self, args: Dict):
        # 硬编码本地CSV文件路径，使用上传文件的路径
        data_path = "/data/saffron/gcg_data/new_gcg_JBB_Llama_8B.csv"
        print(f"加载本地JBB-Behaviors数据集: {data_path}")

        # 检查文件是否存在
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据集文件不存在，请检查路径: {data_path}")

        try:
            # 使用pandas读取CSV文件（修改为read_csv）
            df = pd.read_csv(data_path)
        except Exception as e:
            raise RuntimeError(f"读取CSV文件失败: {str(e)}")

        # 验证必要字段（根据实际CSV列名调整，这里假设为'goal'和'target'）
        required_columns = ["gcg_prompt", "Target"]
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"CSV文件缺少必要列: {missing}，请检查数据集格式")

        # 提取并过滤有效数据（去除空值）
        pairs = []
        for _, row in df.iterrows():
            prompt = str(row["gcg_prompt"]).strip()
            output = str(row["Target"]).strip()
            if prompt and output:  # 跳过空字符串
                pairs.append((prompt, output))

        print(f"成功加载{len(pairs)}条JBB-Behaviors数据（原始样本数：{len(df)}，过滤无效数据后）")
        super().__init__(args=args, pairs=pairs)

    @staticmethod
    def add_args(parser: ArgumentParser):
        pass  # 路径已硬编码，无需命令行参数


@register_cls(DATA)
class StrongREJECT(Data):
    """支持本地加载StrongREJECT Parquet格式数据集（适配官方结构）"""

    def __init__(self, args: Dict):
        # 1. 本地Parquet文件路径（根据实际存储位置修改）
        file_path = "/data/saffron/StrongREJECT/data/train-00000-of-00001.parquet"
        print(f"加载本地StrongREJECT数据集: {file_path}")

        # 2. 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据集文件不存在，请检查路径: {file_path}")

        try:
            # 3. 正确用法：指定格式为"parquet"，通过data_files传文件路径，同时指定split
            dataset = load_dataset(
                "parquet",          # 格式类型：必须显式指定"parquet"
                data_files=file_path,  # 本地文件路径（单文件或通配符，如"data/*.parquet"）
                split="train"       # 强制指定split（即使单文件，也需声明，如"train"）
            )
        except Exception as e:
            raise RuntimeError(f"读取Parquet文件失败: {str(e)}")

        # 4. 验证数据集字段是否匹配（必须包含prompt和target）
        required_features = ["prompt", "target"]
        if not all(feat in dataset.features for feat in required_features):
            missing = [f for f in required_features if f not in dataset.features]
            raise ValueError(f"数据集缺少必要字段: {missing}，请检查数据集版本是否正确")

        # 5. 提取并过滤有效数据（去除空字符串）
        pairs = []
        for item in dataset:
            prompt = str(item["prompt"]).strip()
            target = str(item["target"]).strip()
            if prompt and target:  # 跳过空的prompt或target
                pairs.append((prompt, target))

        print(f"成功加载{len(pairs)}条AdvBench数据（原始样本数：{len(dataset)}，过滤无效数据后）")
        super().__init__(args=args, pairs=pairs)


@register_cls(DATA)
class GCGVicuna(Data):
    """加载本地GCG数据集中的vicuna模型相关数据"""

    def __init__(self, args: Dict):
        # 指定本地JSON文件路径
        data_path = "/data/SafeDecoding-Attackers/GCG.json"
        print(f"加载本地GCG数据集中的vicuna数据: {data_path}")

        # 检查文件是否存在
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据集文件不存在，请检查路径: {data_path}")

        try:
            # 读取JSON文件
            with open(data_path, 'r') as f:
                dataset = json.load(f)
        except Exception as e:
            raise RuntimeError(f"读取JSON文件失败: {str(e)}")

        # 过滤target-model为vicuna的数据
        vicuna_data = [item for item in dataset if item.get("target-model") == "vicuna"]

        if not vicuna_data:
            raise ValueError("在数据集中未找到target-model为'vicuna'的数据")

        # 提取prompt和target
        pairs = []
        for item in vicuna_data:
            prompt = str(item.get("prompt", "")).strip()
            target = str(item.get("target", "")).strip()
            if prompt and target:  # 跳过空字符串
                pairs.append((prompt, target))

        print(f"成功加载{len(pairs)}条GCG Vicuna数据（原始vicuna样本数：{len(vicuna_data)}，过滤无效数据后）")
        super().__init__(args=args, pairs=pairs)

    @staticmethod
    def add_args(parser: ArgumentParser):
        pass  # 路径已硬编码，无需命令行参数


@register_cls(DATA)
class GCGLlama2(Data):
    """加载本地GCG数据集中的vicuna模型相关数据"""

    def __init__(self, args: Dict):
        # 指定本地JSON文件路径
        data_path = "/data/SafeDecoding-Attackers/GCG.json"
        print(f"加载本地GCG数据集中的vicuna数据: {data_path}")

        # 检查文件是否存在
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据集文件不存在，请检查路径: {data_path}")

        try:
            # 读取JSON文件
            with open(data_path, 'r') as f:
                dataset = json.load(f)
        except Exception as e:
            raise RuntimeError(f"读取JSON文件失败: {str(e)}")

        # 过滤target-model为vicuna的数据
        vicuna_data = [item for item in dataset if item.get("target-model") == "llama2"]

        if not vicuna_data:
            raise ValueError("在数据集中未找到target-model为'vicuna'的数据")

        # 提取prompt和target
        pairs = []
        for item in vicuna_data:
            prompt = str(item.get("prompt", "")).strip()
            target = str(item.get("target", "")).strip()
            if prompt and target:  # 跳过空字符串
                pairs.append((prompt, target))

        print(f"成功加载{len(pairs)}条GCG Vicuna数据（原始vicuna样本数：{len(vicuna_data)}，过滤无效数据后）")
        super().__init__(args=args, pairs=pairs)

    @staticmethod
    def add_args(parser: ArgumentParser):
        pass  # 路径已硬编码，无需命令行参数

@register_cls(DATA)
class PAIRvicuna(Data):
    """加载本地GCG数据集中的vicuna模型相关数据"""

    def __init__(self, args: Dict):
        # 指定本地JSON文件路径
        data_path = "/data/SafeDecoding-Attackers/PAIR.json"
        print(f"加载本地GCG数据集中的vicuna数据: {data_path}")

        # 检查文件是否存在
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据集文件不存在，请检查路径: {data_path}")

        try:
            # 读取JSON文件
            with open(data_path, 'r') as f:
                dataset = json.load(f)
        except Exception as e:
            raise RuntimeError(f"读取JSON文件失败: {str(e)}")

        # 过滤target-model为vicuna的数据
        vicuna_data = [item for item in dataset if item.get("target-model") == "vicuna"]

        if not vicuna_data:
            raise ValueError("在数据集中未找到target-model为'vicuna'的数据")

        # 提取prompt和target
        pairs = []
        for item in vicuna_data:
            prompt = str(item.get("prompt", "")).strip()
            target = str(item.get("target", "")).strip()
            if prompt and target:  # 跳过空字符串
                pairs.append((prompt, target))

        print(f"成功加载{len(pairs)}条GCG Vicuna数据（原始vicuna样本数：{len(vicuna_data)}，过滤无效数据后）")
        super().__init__(args=args, pairs=pairs)

    @staticmethod
    def add_args(parser: ArgumentParser):
        pass  # 路径已硬编码，无需命令行参数

@register_cls(DATA)
class PAIRLlama2(Data):
    """加载本地GCG数据集中的vicuna模型相关数据"""

    def __init__(self, args: Dict):
        # 指定本地JSON文件路径
        data_path = "/data/SafeDecoding-Attackers/PAIR.json"
        print(f"加载本地PAIR数据集中的llama2数据: {data_path}")

        # 检查文件是否存在
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据集文件不存在，请检查路径: {data_path}")

        try:
            # 读取JSON文件
            with open(data_path, 'r') as f:
                dataset = json.load(f)
        except Exception as e:
            raise RuntimeError(f"读取JSON文件失败: {str(e)}")

        # 过滤target-model为vicuna的数据
        vicuna_data = [item for item in dataset if item.get("target-model") == "llama2"]

        if not vicuna_data:
            raise ValueError("在数据集中未找到target-model为'vicuna'的数据")

        # 提取prompt和target
        pairs = []
        for item in vicuna_data:
            prompt = str(item.get("prompt", "")).strip()
            target = str(item.get("target", "")).strip()
            if prompt and target:  # 跳过空字符串
                pairs.append((prompt, target))

        print(f"成功加载{len(pairs)}条GCG Vicuna数据（原始vicuna样本数：{len(vicuna_data)}，过滤无效数据后）")
        super().__init__(args=args, pairs=pairs)

    @staticmethod
    def add_args(parser: ArgumentParser):
        pass  # 路径已硬编码，无需命令行参数

@register_cls(DATA)
class DeepInception(Data):
    """加载本地GCG数据集中的vicuna模型相关数据"""

    def __init__(self, args: Dict):
        # 指定本地JSON文件路径
        data_path = "/data/SafeDecoding-Attackers/DeepInception.json"
        print(f"加载本地GCG数据集中的vicuna数据: {data_path}")

        # 检查文件是否存在
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据集文件不存在，请检查路径: {data_path}")

        try:
            # 读取JSON文件
            with open(data_path, 'r') as f:
                dataset = json.load(f)
        except Exception as e:
            raise RuntimeError(f"读取JSON文件失败: {str(e)}")

        # 过滤target-model为vicuna的数据
        vicuna_data = [item for item in dataset if item.get("source") == "DeepInception"]

        if not vicuna_data:
            raise ValueError("在数据集中未找到target-model为'vicuna'的数据")

        # 提取prompt和target
        pairs = []
        for item in vicuna_data:
            prompt = str(item.get("prompt", "")).strip()
            target = str(item.get("target", "")).strip()
            if prompt and target:  # 跳过空字符串
                pairs.append((prompt, target))

        print(f"成功加载{len(pairs)}条GCG Vicuna数据（原始vicuna样本数：{len(vicuna_data)}，过滤无效数据后）")
        super().__init__(args=args, pairs=pairs)

    @staticmethod
    def add_args(parser: ArgumentParser):
        pass  # 路径已硬编码，无需命令行参数

@register_cls(DATA)
class ORbench(Data):
    """加载本地GCG数据集中的vicuna模型相关数据"""

    def __init__(self, args: Dict):
        # 指定本地JSON文件路径
        data_path = "/data/SafeDecoding-Attackers/orbench.json"
        print(f"加载本地GCG数据集中的vicuna数据: {data_path}")

        # 检查文件是否存在
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据集文件不存在，请检查路径: {data_path}")

        try:
            # 读取JSON文件
            with open(data_path, 'r') as f:
                dataset = json.load(f)
        except Exception as e:
            raise RuntimeError(f"读取JSON文件失败: {str(e)}")

        # 过滤target-model为vicuna的数据
        vicuna_data = [item for item in dataset if item.get("source") == "DeepInception"]

        if not vicuna_data:
            raise ValueError("在数据集中未找到target-model为'vicuna'的数据")

        # 提取prompt和target
        pairs = []
        for item in vicuna_data:
            prompt = str(item.get("prompt", "")).strip()
            target = str(item.get("target", "")).strip()
            if prompt and target:  # 跳过空字符串
                pairs.append((prompt, target))

        print(f"成功加载{len(pairs)}条GCG Vicuna数据（原始vicuna样本数：{len(vicuna_data)}，过滤无效数据后）")
        super().__init__(args=args, pairs=pairs)

    @staticmethod
    def add_args(parser: ArgumentParser):
        pass  # 路径已硬编码，无需命令行参数

@register_cls(DATA)
class XSTest(Data):
    """加载本地GCG数据集中的vicuna模型相关数据"""

    def __init__(self, args: Dict):
        # 指定本地JSON文件路径
        data_path = "/data/SafeDecoding-Attackers/XSTest.json"
        print(f"加载本地GCG数据集中的vicuna数据: {data_path}")

        # 检查文件是否存在
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据集文件不存在，请检查路径: {data_path}")

        try:
            # 读取JSON文件
            with open(data_path, 'r') as f:
                dataset = json.load(f)
        except Exception as e:
            raise RuntimeError(f"读取JSON文件失败: {str(e)}")

        # 过滤target-model为vicuna的数据
        vicuna_data = [item for item in dataset if item.get("source") == "SafeDataExtraction"]

        if not vicuna_data:
            raise ValueError("在数据集中未找到target-model为'vicuna'的数据")

        # 提取prompt和target
        pairs = []
        for item in vicuna_data:
            prompt = str(item.get("prompt", "")).strip()
            target = str(item.get("target", "")).strip()
            if prompt and target:  # 跳过空字符串
                pairs.append((prompt, target))

        print(f"成功加载{len(pairs)}条GCG Vicuna数据（原始vicuna样本数：{len(vicuna_data)}，过滤无效数据后）")
        super().__init__(args=args, pairs=pairs)

    @staticmethod
    def add_args(parser: ArgumentParser):
        pass  # 路径已硬编码，无需命令行参数

@register_cls(DATA)
class ORbench(Data):
    """加载本地GCG数据集中的vicuna模型相关数据"""

    def __init__(self, args: Dict):
        # 指定本地JSON文件路径
        data_path = "/data/SafeDecoding-Attackers/orbench.json"
        print(f"加载本地GCG数据集中的vicuna数据: {data_path}")

        # 检查文件是否存在
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据集文件不存在，请检查路径: {data_path}")

        try:
            # 读取JSON文件
            with open(data_path, 'r') as f:
                dataset = json.load(f)
        except Exception as e:
            raise RuntimeError(f"读取JSON文件失败: {str(e)}")

        # 过滤target-model为vicuna的数据
        vicuna_data = [item for item in dataset if item.get("source") == "DeepInception"]

        if not vicuna_data:
            raise ValueError("在数据集中未找到target-model为'vicuna'的数据")

        # 提取prompt和target
        pairs = []
        for item in vicuna_data:
            prompt = str(item.get("prompt", "")).strip()
            target = str(item.get("target", "")).strip()
            if prompt and target:  # 跳过空字符串
                pairs.append((prompt, target))

        print(f"成功加载{len(pairs)}条GCG Vicuna数据（原始vicuna样本数：{len(vicuna_data)}，过滤无效数据后）")
        super().__init__(args=args, pairs=pairs)

    @staticmethod
    def add_args(parser: ArgumentParser):
        pass  # 路径已硬编码，无需命令行参数


@register_cls(DATA)
class AutoDANvicuna(Data):
    """加载本地GCG数据集中的vicuna模型相关数据"""

    def __init__(self, args: Dict):
        # 指定本地JSON文件路径
        data_path = "/data/SafeDecoding-Attackers/AutoDAN.json"
        print(f"加载本地PAIR数据集中的llama2数据: {data_path}")

        # 检查文件是否存在
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据集文件不存在，请检查路径: {data_path}")

        try:
            # 读取JSON文件
            with open(data_path, 'r') as f:
                dataset = json.load(f)
        except Exception as e:
            raise RuntimeError(f"读取JSON文件失败: {str(e)}")

        # 过滤target-model为vicuna的数据
        vicuna_data = [item for item in dataset if item.get("target-model") == "vicuna"]

        if not vicuna_data:
            raise ValueError("在数据集中未找到target-model为'vicuna'的数据")

        # 提取prompt和target
        pairs = []
        for item in vicuna_data:
            prompt = str(item.get("prompt", "")).strip()
            target = str(item.get("target", "")).strip()
            if prompt and target:  # 跳过空字符串
                pairs.append((prompt, target))

        print(f"成功加载{len(pairs)}条GCG Vicuna数据（原始vicuna样本数：{len(vicuna_data)}，过滤无效数据后）")
        super().__init__(args=args, pairs=pairs)

    @staticmethod
    def add_args(parser: ArgumentParser):
        pass  # 路径已硬编码，无需命令行参数

@register_cls(DATA)
class AutoDANllama2(Data):
    """加载本地GCG数据集中的vicuna模型相关数据"""

    def __init__(self, args: Dict):
        # 指定本地JSON文件路径
        data_path = "/data/SafeDecoding-Attackers/AutoDAN.json"
        print(f"加载本地PAIR数据集中的llama2数据: {data_path}")

        # 检查文件是否存在
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据集文件不存在，请检查路径: {data_path}")

        try:
            # 读取JSON文件
            with open(data_path, 'r') as f:
                dataset = json.load(f)
        except Exception as e:
            raise RuntimeError(f"读取JSON文件失败: {str(e)}")

        # 过滤target-model为vicuna的数据
        vicuna_data = [item for item in dataset if item.get("target-model") == "llama2"]

        if not vicuna_data:
            raise ValueError("在数据集中未找到target-model为'vicuna'的数据")

        # 提取prompt和target
        pairs = []
        for item in vicuna_data:
            prompt = str(item.get("prompt", "")).strip()
            target = str(item.get("target", "")).strip()
            if prompt and target:  # 跳过空字符串
                pairs.append((prompt, target))

        print(f"成功加载{len(pairs)}条GCG Vicuna数据（原始vicuna样本数：{len(vicuna_data)}，过滤无效数据后）")
        super().__init__(args=args, pairs=pairs)

    @staticmethod
    def add_args(parser: ArgumentParser):
        pass  # 路径已硬编码，无需命令行参数