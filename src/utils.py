import json
import yaml
from typing import List

def save_json(data: List, path: str) -> None:
    with open(path, "w") as json_file:
        json.dump(data, json_file)  
        
def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def load_yaml_file(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data