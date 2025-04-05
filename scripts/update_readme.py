#!/usr/bin/env python
import os
import sys
import json
import logging
import asyncio
# import pandas as pd  # 移除 pandas 依赖
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入后端服务
from backend.app.services.leaderboard import LeaderboardService
from backend.app.config.hf_config import HF_ORGANIZATION

async def generate_markdown_table(data, top_n=20):
    """生成Markdown格式的排行榜表格"""
    # 确保数据不为空
    if not data:
        return "No data available"
    
    # 排序并选择前N个模型
    sorted_data = sorted(data, key=lambda x: x.get('average', 0) or 0, reverse=True)[:top_n]
    
    # 构建Markdown表格
    header = "| Rank | Model | Average Score | Parameters | Type |\n"
    separator = "| ---- | ----- | ------------ | ---------- | ---- |\n"
    
    rows = ""
    for i, item in enumerate(sorted_data):
        try:
            model_name = item.get('model_name', 'Unknown')
            model_id = item.get('model_id', 'Unknown')
            model_link = f"https://huggingface.co/{model_id}"
            
            # 确保数值类型正确
            try:
                average = float(item.get('average', 0) or 0)
            except (ValueError, TypeError):
                average = 0.0
                
            try:
                params = float(item.get('params_billions', 0) or 0)
            except (ValueError, TypeError):
                params = 0.0
                
            model_type = item.get('model_type', 'Unknown')
            
            rows += f"| {i+1} | [{model_name}]({model_link}) | {average:.2f} | {params}B | {model_type} |\n"
        except Exception as e:
            logger.error(f"生成表格行时出错: {str(e)}")
            continue
    
    return header + separator + rows

async def update_readme():
    """更新README文件"""
    logger.info("开始更新README文件")
    
    # 初始化服务
    leaderboard_service = LeaderboardService()
    
    # 获取排行榜数据
    logger.info("获取排行榜数据")
    leaderboard_data = await leaderboard_service.get_formatted_leaderboard()
    
    # 生成Markdown表格
    logger.info("生成Markdown表格")
    table = await generate_markdown_table(leaderboard_data)
    
    # 读取现有README
    readme_path = Path(__file__).parent.parent / "README.md"
    if not readme_path.exists():
        # 如果README不存在，创建一个新的，使用新的项目名称
        content = f"# ModelRank AI\n\n"  # 修改为新的项目名称
    else:
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
    
    # 更新时间戳
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    # 检查README是否已有排行榜部分
    if "## 🏆 ModelRank AI 排行榜" in content:
        # 替换现有排行榜部分
        start_marker = "## 🏆 ModelRank AI 排行榜"
        end_marker = "##" # 下一个标题开始
        
        start_idx = content.find(start_marker)
        end_idx = content.find(end_marker, start_idx + len(start_marker))
        
        if end_idx == -1:  # 如果是最后一个部分
            end_idx = len(content)
        
        new_section = f"{start_marker}\n\n*最后更新时间: {now}*\n\n{table}\n\n"
        content = content[:start_idx] + new_section + content[end_idx:]
    else:
        # 添加新的排行榜部分
        content += f"\n\n## 🏆 ModelRank AI 排行榜\n\n*最后更新时间: {now}*\n\n{table}\n"
    
    # 写回README
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    logger.info(f"README更新成功，时间: {now}")

if __name__ == "__main__":
    asyncio.run(update_readme())