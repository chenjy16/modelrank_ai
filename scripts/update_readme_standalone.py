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

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

# HuggingFace 配置
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_ORGANIZATION = "open-llm-leaderboard"
CONTENTS_REPO = f"{HF_ORGANIZATION}/contents"

# 初始化 HF API
api = HfApi(token=HF_TOKEN)

async def fetch_leaderboard_data():
    """从 HuggingFace 获取排行榜数据"""
    logger.info("正在获取排行榜数据...")
    
    try:
        # 禁用进度条
        datasets.disable_progress_bar()
        
        # 加载数据集
        dataset = datasets.load_dataset(CONTENTS_REPO)["train"]
        
        # 转换为 pandas DataFrame
        df = dataset.to_pandas()
        
        # 按平均分数排序
        df = df.sort_values(by="Average ⬆️", ascending=False)
        
        logger.info(f"成功获取 {len(df)} 条模型数据")
        return df
    except Exception as e:
        logger.error(f"获取数据失败: {str(e)}")
        return None

def format_model_name(row):
    """格式化模型名称，添加链接"""
    model_name = row["Model"]
    full_name = row["fullname"]
    
    if pd.isna(full_name) or full_name == "":
        return model_name
    
    return f"[{model_name}](https://huggingface.co/{full_name})"

async def generate_markdown_table(df):
    """生成 Markdown 表格"""
    if df is None or len(df) == 0:
        return "暂无数据"
    
    # 选择要显示的列
    columns = [
        "Model", "Average ⬆️", "#Params (B)", 
        "IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"
    ]
    
    # 确保所有列都存在
    for col in columns:
        if col not in df.columns:
            logger.warning(f"列 {col} 不存在，将被跳过")
            columns.remove(col)
    
    # 创建一个新的 DataFrame 只包含我们需要的列
    display_df = df[columns].copy()
    
    # 格式化模型名称
    display_df["Model"] = df.apply(format_model_name, axis=1)
    
    # 重命名列以便更好地显示
    column_rename = {
        "Model": "模型",
        "Average ⬆️": "平均分数",
        "#Params (B)": "参数量(B)",
        "IFEval": "IFEval",
        "BBH": "BBH",
        "MATH Lvl 5": "MATH",
        "GPQA": "GPQA",
        "MUSR": "MUSR",
        "MMLU-PRO": "MMLU-PRO"
    }
    
    display_df = display_df.rename(columns=column_rename)
    
    # 限制显示的行数
    top_models = display_df.head(20)
    
    # 生成 Markdown 表格
    markdown_table = "| 排名 | " + " | ".join(top_models.columns) + " |\n"
    markdown_table += "| --- | " + " | ".join(["---"] * len(top_models.columns)) + " |\n"
    
    for i, (_, row) in enumerate(top_models.iterrows(), 1):
        # 格式化数值，保留两位小数
        formatted_row = []
        for col, value in row.items():
            if isinstance(value, (int, float)) and col != "参数量(B)":
                formatted_row.append(f"{value:.2f}")
            elif col == "参数量(B)" and isinstance(value, (int, float)):
                formatted_row.append(f"{value:.1f}")
            else:
                formatted_row.append(str(value))
        
        markdown_table += f"| {i} | " + " | ".join(formatted_row) + " |\n"
    
    return markdown_table

async def update_readme():
    """更新 README 文件"""
    logger.info("开始更新 README 文件")
    
    # 获取排行榜数据
    df = await fetch_leaderboard_data()
    
    if df is None:
        logger.error("无法获取数据，更新失败")
        return False
    
    # 生成 Markdown 表格
    table = await generate_markdown_table(df)
    
    # 读取现有 README
    readme_path = Path(__file__).parent.parent / "README.md"
    if not readme_path.exists():
        # 如果 README 不存在，创建一个新的
        content = "# ModelRank AI\n\n这是一个自动更新的开源大语言模型排行榜，数据来源于HuggingFace的Open LLM Leaderboard。\n\n## 项目说明\n\n本项目通过GitHub Actions每天自动从HuggingFace获取最新的模型评测数据，并更新到此README中。\n\n"
    else:
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
    
    # 更新时间戳
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    # 检查 README 是否已有排行榜部分
    if "## 🏆 ModelRank AI 排行榜" in content:
        # 替换现有排行榜部分
        start_marker = "## 🏆 ModelRank AI 排行榜"
        end_marker = "## "  # 下一个标题开始
        
        start_idx = content.find(start_marker)
        end_idx = content.find(end_marker, start_idx + len(start_marker))
        
        if end_idx == -1:  # 如果是最后一个部分
            end_idx = len(content)
        
        new_section = f"{start_marker}\n\n*最后更新时间: {now}*\n\n{table}\n\n"
        content = content[:start_idx] + new_section + content[end_idx:]
    else:
        # 添加新的排行榜部分
        content += f"\n## 🏆 ModelRank AI 排行榜\n\n*最后更新时间: {now}*\n\n{table}\n\n"
    
    # 添加数据来源说明（如果不存在）
    if "## 数据来源" not in content:
        content += "\n## 数据来源\n\n数据来自HuggingFace的[Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)。\n\n"
    
    # 添加许可证说明（如果不存在）
    if "## 许可证" not in content:
        content += "\n## 许可证\n\n本项目基于MIT许可证开源。\n"
    
    # 写回 README
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    logger.info(f"README 更新成功，时间: {now}")
    return True

if __name__ == "__main__":
    # 检查 HF_TOKEN 是否存在
    if not HF_TOKEN:
        logger.error("未设置 HF_TOKEN 环境变量，请先设置后再运行")
        sys.exit(1)
    
    # 运行主函数
    success = asyncio.run(update_readme())
    
    if not success:
        sys.exit(1)