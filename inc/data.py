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

        data_path = 'path/Harmful-HEx-PHI.jsonl'  
        print('Using local dataset from', data_path)

        dataset = []
        with open(data_path, 'r') as fi:
            for line in fi:
                dataset.append(json.loads(line))
        super().__init__(args=args, pairs=[(chat[0]['content'], chat[1]['content']) for chat in dataset])


@register_cls(DATA)



from inc.utils import *
from datasets import load_dataset  

from datasets import load_dataset  
import os
from typing import Dict

@register_cls(DATA)
class AdvBench(Data):

    def __init__(self, args: Dict):

        data_path = /AdvBench/data/harmful_behaviors01.csv"


        if not os.path.exists(data_path):
            raise FileNotFoundError(f": {data_path}")

        try:

            df = pd.read_csv(data_path)
        except Exception as e:
            raise RuntimeError(f": {str(e)}")

       
        required_columns = ["goal", "target"]
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            raise ValueError(f" {missing}，")


        pairs = []
        for _, row in df.iterrows():
            prompt = str(row["goal"]).strip()
            output = str(row["target"]).strip()
            if prompt and output: 
                pairs.append((prompt, output))

        print(f"{len(pairs)}{len(df)}")
        super().__init__(args=args, pairs=pairs)

    @staticmethod
    def add_args(parser: ArgumentParser):
        pass  


from inc.utils import *
import pandas as pd
import os








@register_cls(DATA)
class GCGVicuna(Data):


    def __init__(self, args: Dict):

        data_path = "/data/SafeDecoding-Attackers/GCG.json"
        print(f": {data_path}")

       
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"{data_path}")

        try:

            with open(data_path, 'r') as f:
                dataset = json.load(f)
        except Exception as e:
            raise RuntimeError(f"{str(e)}")

        vicuna_data = [item for item in dataset if item.get("target-model") == "vicuna"]

        if not vicuna_data:
            raise ValueError("")


        pairs = []
        for item in vicuna_data:
            prompt = str(item.get("prompt", "")).strip()
            target = str(item.get("target", "")).strip()
            if prompt and target: 
                pairs.append((prompt, target))

        print(f"{len(pairs)}{len(vicuna_data)}，")
        super().__init__(args=args, pairs=pairs)

    @staticmethod
    def add_args(parser: ArgumentParser):
        pass 

@register_cls(DATA)
class GCGLlama2(Data):


    def __init__(self, args: Dict):
       
        data_path = "/data/SafeDecoding-Attackers/GCG.json"
        print(f": {data_path}")

     
        if not os.path.exists(data_path):
            raise FileNotFoundError(f" {data_path}")

        try:

            with open(data_path, 'r') as f:
                dataset = json.load(f)
        except Exception as e:
            raise RuntimeError(f": {str(e)}")

        vicuna_data = [item for item in dataset if item.get("target-model") == "llama2"]

        if not vicuna_data:
            raise ValueError("")


        pairs = []
        for item in vicuna_data:
            prompt = str(item.get("prompt", "")).strip()
            target = str(item.get("target", "")).strip()
            if prompt and target:  
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
