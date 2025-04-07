#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import asyncio
from pathlib import Path
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import HfApi
import datasets

# Add project root and backend directories to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "backend"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# HuggingFace configuration
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_ORGANIZATION = "open-llm-leaderboard"
CONTENTS_REPO = f"{HF_ORGANIZATION}/contents"

# Initialize HF API
api = HfApi(token=HF_TOKEN)

# Since we're having import issues with the backend services,
# let's implement the necessary functionality directly in this script
# instead of importing from backend.app.services.leaderboard

async def fetch_leaderboard_data():
    """Fetch leaderboard data from HuggingFace"""
    logger.info("Fetching leaderboard data...")
    
    try:
        # Disable progress bar
        datasets.disable_progress_bar()
        
        # Load dataset
        dataset = datasets.load_dataset(CONTENTS_REPO)["train"]
        
        # Convert to pandas DataFrame
        df = dataset.to_pandas()
        
        # Sort by average score
        df = df.sort_values(by="Average ⬆️", ascending=False)
        
        logger.info(f"Successfully retrieved {len(df)} model entries")
        return df
    except Exception as e:
        logger.error(f"Failed to fetch data: {str(e)}")
        return None

def format_model_name(row):
    """Format model name with links"""
    model_name = row["Model"]
    full_name = row["fullname"]
    
    if pd.isna(full_name) or full_name == "":
        return model_name
    
    return f"[{model_name}](https://huggingface.co/{full_name})"

async def generate_markdown_table(df):
    """Generate Markdown table"""
    if df is None or len(df) == 0:
        return "No data available"
    
    # Select columns to display
    columns = [
        "Model", "Average ⬆️", "#Params (B)", 
        "IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"
    ]
    
    # Ensure all columns exist
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column {col} does not exist, will be skipped")
            columns.remove(col)
    
    # Create a new DataFrame with only the columns we need
    display_df = df[columns].copy()
    
    # Format model names
    display_df["Model"] = df.apply(format_model_name, axis=1)
    
    # Rename columns for better display
    column_rename = {
        "Model": "Model",
        "Average ⬆️": "Average Score",
        "#Params (B)": "Parameters(B)",
        "IFEval": "IFEval",
        "BBH": "BBH",
        "MATH Lvl 5": "MATH",
        "GPQA": "GPQA",
        "MUSR": "MUSR",
        "MMLU-PRO": "MMLU-PRO"
    }
    
    display_df = display_df.rename(columns=column_rename)
    
    # Limit the number of rows to display
    top_models = display_df.head(20)
    
    # Generate Markdown table
    markdown_table = "| Rank | " + " | ".join(top_models.columns) + " |\n"
    markdown_table += "| --- | " + " | ".join(["---"] * len(top_models.columns)) + " |\n"
    
    for i, (_, row) in enumerate(top_models.iterrows(), 1):
        # Format numbers with appropriate decimal places
        formatted_row = []
        for col, value in row.items():
            if isinstance(value, (int, float)) and col != "Parameters(B)":
                formatted_row.append(f"{value:.2f}")
            elif col == "Parameters(B)" and isinstance(value, (int, float)):
                formatted_row.append(f"{value:.1f}")
            else:
                formatted_row.append(str(value))
        
        markdown_table += f"| {i} | " + " | ".join(formatted_row) + " |\n"
    
    return markdown_table

async def update_readme():
    """Update README file"""
    logger.info("Starting README update")
    
    # Get leaderboard data
    df = await fetch_leaderboard_data()
    
    if df is None:
        logger.error("Unable to fetch data, update failed")
        return False
    
    # Generate Markdown table
    table = await generate_markdown_table(df)
    
    # Read existing README
    readme_path = Path(__file__).parent.parent / "README.md"
    if not readme_path.exists():
        # 如果README不存在，创建一个新的
        content = "# ModelRank AI\n\nThis is an automatically updated open-source large language model leaderboard with data sourced from HuggingFace.\n\n## Project Description\n\nThis project automatically fetches the latest model evaluation data from HuggingFace daily via GitHub Actions and updates this README.\n\n"
    else:
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
    
    # 更新时间
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    # 检查是否已存在主排行榜部分
    if "## 🏆 ModelRank AI Leaderboard" in content:
        # 使用正则表达式替换现有的排行榜部分
        pattern = r"## 🏆 ModelRank AI Leaderboard\s*\n\s*\*Last updated:.*?\*\s*\n\s*\|.*?(?=\n\n|\Z)"
        replacement = f"## 🏆 ModelRank AI Leaderboard\n\n*Last updated: {now}*\n\n{table}"
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    else:
        # 如果不存在，添加到内容末尾
        content += f"\n\n## 🏆 ModelRank AI Leaderboard\n\n*Last updated: {now}*\n\n{table}\n\n[View Complete Online Leaderboard](https://chenjy16.github.io/modelrank_ai/)"
    
    # 检查是否已存在领域排行榜链接部分
    domain_links_section = "## Domain-Specific Leaderboards"
    emoji_domain_links_section = "## 🌐 Domain-Specific Leaderboards"
    
    # 删除带有emoji的重复部分（如果存在）
    if emoji_domain_links_section in content:
        pattern = r"## 🌐 Domain-Specific Leaderboards\s*\n.*?(?=\n\n## |\Z)"
        content = re.sub(pattern, "", content, flags=re.DOTALL)
        # 清理多余的空行
        content = re.sub(r'\n{3,}', '\n\n', content)
    
    # 检查是否需要添加领域排行榜链接部分
    if domain_links_section not in content:
        domain_links = f"""
## Domain-Specific Leaderboards

Domain-specific model leaderboards can be accessed via the following links:

- [Medical Domain Leaderboard](https://chenjy16.github.io/modelrank_ai/medical_leaderboard.html)
- [Legal Domain Leaderboard](https://chenjy16.github.io/modelrank_ai/legal_leaderboard.html)
- [Finance Domain Leaderboard](https://chenjy16.github.io/modelrank_ai/finance_leaderboard.html)
"""
        # 在适当位置添加领域排行榜链接
        if "## Evaluation Metrics Explanation" in content:
            content = content.replace("## Evaluation Metrics Explanation", f"{domain_links}\n\n## Evaluation Metrics Explanation")
        else:
            content += f"\n\n{domain_links}"
    
    # 写入更新后的README
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    logger.info(f"README updated at {readme_path}")
    return True

if __name__ == "__main__":
    # Check if HF_TOKEN exists
    if not HF_TOKEN:
        logger.error("HF_TOKEN environment variable not set, please set it before running")
        sys.exit(1)
    
    # Run main function
    success = asyncio.run(update_readme())
    
    if not success:
        sys.exit(1)
