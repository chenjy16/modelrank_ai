import os
import json
import pytz
import logging
import asyncio
import sys  # 添加sys模块导入
from datetime import datetime
from pathlib import Path
import huggingface_hub
from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError
from dotenv import load_dotenv
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# 确保加载环境变量
load_dotenv()

# 导入项目模块
from app.config.hf_config import HF_TOKEN, API
from app.utils.model_validation import ModelValidator

# 配置日志和其他设置
huggingface_hub.logging.set_verbosity_error()
huggingface_hub.utils.disable_progress_bars()

logging.basicConfig(
    level=logging.ERROR,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

validator = ModelValidator()

# 首先定义辅助函数
def read_json(repo_path, file):
    """读取JSON文件"""
    file_path = Path(repo_path) / file
    with open(file_path, "r") as f:
        return json.load(f)

def write_json(repo_path, file, content):
    """写入JSON文件"""
    file_path = Path(repo_path) / file
    with open(file_path, "w") as f:
        json.dump(content, f, indent=2)

def get_files(repo_path):
    """获取目录中的所有 JSON 文件"""
    path = Path(repo_path)
    if not path.exists():
        return []
    
    files = []
    for file_path in path.glob('**/*.json'):
        files.append(str(file_path.relative_to(repo_path)))
    
    return files

# 然后定义主函数
def main():
    # 使用更可靠的路径构建方式
    requests_path = Path(__file__).resolve().parent.parent.parent / "data" / "requests"
    
    # 确保目录存在
    if not requests_path.exists():
        os.makedirs(requests_path, exist_ok=True)
        print(f"Created directory: {requests_path}")
    
    # 不再使用日期范围，直接获取所有文件
    changed_files = get_files(requests_path)

    for file in tqdm(changed_files):
        try:
            request_data = read_json(requests_path, file)
        except FileNotFoundError as e:
            tqdm.write(f"文件 {file} 未找到")
            continue
        
        try:
            model_info = API.model_info(
                repo_id=request_data["model"],
                revision=request_data["revision"],
                token=HF_TOKEN
            )
        except (RepositoryNotFoundError, RevisionNotFoundError) as e:
            tqdm.write(f"Model info for {request_data['model']} not found")
            continue
        
        with logging_redirect_tqdm():
            new_model_size, error = asyncio.run(validator.get_model_size(
                model_info=model_info,
                precision=request_data["precision"],
                base_model=request_data["base_model"],
                revision=request_data["revision"]
            ))

        if error:
            tqdm.write(f"Error getting model size info for {request_data['model']}, {error}")
            continue
        
        old_model_size = request_data["params"]
        if old_model_size != new_model_size:
            if new_model_size > 100:
                tqdm.write(f"Model: {request_data['model']}, size is more 100B: {new_model_size}")
            
            tqdm.write(f"Model: {request_data['model']}, old size: {request_data['params']} new size: {new_model_size}")
            tqdm.write(f"Updating request file {file}")

            request_data["params"] = new_model_size
            write_json(requests_path, file, content=request_data)

if __name__ == "__main__":
    main()
