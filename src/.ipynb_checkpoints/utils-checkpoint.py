import pandas as pd

def load_csv(file_path):
    """加載 CSV 文件"""
    return pd.read_csv(file_path)

def save_json(data, file_path):
    """保存為 JSON 文件"""
    import json
    with open(file_path, 'w') as f:
        json.dump(data, f)

def load_json(file_path):
    """加載 JSON 文件"""
    import json
    with open(file_path, 'r') as f:
        return json.load(f)
