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
import json

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

def format_model_name_html(row):
    """格式化模型名称，添加HTML链接"""
    model_name = row["Model"]
    full_name = row["fullname"]
    
    if pd.isna(full_name) or full_name == "":
        return model_name
    
    return f'<a href="https://huggingface.co/{full_name}" target="_blank">{model_name}</a>'

async def generate_markdown_table(df, limit=20):
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
    if limit:
        top_models = display_df.head(limit)
    else:
        top_models = display_df
    
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

def generate_html_page(df, update_time):
    """生成完整的HTML页面"""
    # 创建HTML版本的DataFrame
    html_df = df.copy()
    
    # 添加排名列
    html_df.insert(0, '排名', range(1, len(html_df) + 1))
    
    # 格式化模型名称为HTML链接
    html_df["Model"] = html_df.apply(format_model_name_html, axis=1)
    
    # 重命名列
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
    
    # 选择要显示的列
    display_columns = ['排名'] + list(column_rename.keys())
    display_df = html_df[display_columns].copy()
    
    # 重命名列
    display_df = display_df.rename(columns=column_rename)
    
    # 格式化数值列
    for col in display_df.columns:
        if col not in ['排名', '模型']:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.2f}" if isinstance(x, (int, float)) and col != "参数量(B)" else 
                          f"{x:.1f}" if col == "参数量(B)" and isinstance(x, (int, float)) else x
            )
    
    # 生成HTML表格
    html_table = display_df.to_html(
        index=False,
        escape=False,
        classes="table table-striped table-hover table-bordered",
        table_id="leaderboard"
    )
    
    # 创建完整的HTML页面
    # 使用三引号字符串和 r 前缀来避免 f-string 解析问题
    js_code = r"""
    <script>
        $(document).ready(function() {
            $('#leaderboard').DataTable({
                "pageLength": 25,
                "order": [[0, "asc"]],
                "language": {
                    "search": "搜索:",
                    "lengthMenu": "显示 _MENU_ 条记录",
                    "info": "显示第 _START_ 至 _END_ 条记录，共 _TOTAL_ 条",
                    "infoEmpty": "没有记录",
                    "infoFiltered": "(从 _MAX_ 条记录过滤)",
                    "paginate": {
                        "first": "首页",
                        "last": "末页",
                        "next": "下一页",
                        "previous": "上一页"
                    }
                }
            });
        });
    </script>
    """
    
    # 然后在 HTML 内容中使用这个变量
    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ModelRank AI - 大语言模型排行榜</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.datatables.net/1.13.4/css/dataTables.bootstrap5.min.css">
    <style>
        body {{ 
            padding: 20px; 
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }}
        .container {{ max-width: 1400px; }}
        h1 {{ margin-bottom: 20px; color: #333; }}
        .table {{ width: 100%; }}
        .table th {{ 
            position: sticky; 
            top: 0; 
            background-color: #f8f9fa; 
            color: #333;
            font-weight: 600;
        }}
        .table td {{ vertical-align: middle; }}
        .footer {{ 
            margin-top: 30px; 
            padding-top: 10px; 
            border-top: 1px solid #eee; 
            color: #666;
            font-size: 0.9rem;
        }}
        .card {{ margin-bottom: 20px; }}
        .badge {{ font-size: 0.8rem; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="row mb-4">
            <div class="col-md-8">
                <h1>🏆 ModelRank AI - 大语言模型排行榜</h1>
                <p class="text-muted">最后更新时间: {update_time}</p>
            </div>
            <div class="col-md-4 text-end">
                <a href="https://github.com/chenjy16/modelrank_ai" class="btn btn-outline-dark">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-github" viewBox="0 0 16 16">
                        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
                    </svg>
                    GitHub 仓库
                </a>
            </div>
        </div>
        
        <div class="card">
            <div class="card-header bg-light">
                <div class="row">
                    <div class="col">
                        <h5 class="mb-0">排行榜数据</h5>
                    </div>
                    <div class="col-auto">
                        <span class="badge bg-primary">共 {len(display_df)} 个模型</span>
                    </div>
                </div>
            </div>
            <div class="card-body p-0">
                <div class="table-responsive">
                    {html_table}
                </div>
            </div>
        </div>
        
        <div class="row mt-4">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header bg-light">
                        <h5 class="mb-0">数据下载</h5>
                    </div>
                    <div class="card-body">
                        <p>您可以通过以下链接下载完整数据：</p>
                        <a href="leaderboard.json" class="btn btn-outline-primary me-2">JSON 格式</a>
                        <a href="leaderboard.csv" class="btn btn-outline-primary">CSV 格式</a>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header bg-light">
                        <h5 class="mb-0">关于项目</h5>
                    </div>
                    <div class="card-body">
                        <p>ModelRank AI 是一个自动更新的开源大语言模型排行榜，数据来源于HuggingFace。</p>
                        <p>本项目通过GitHub Actions每天自动从HuggingFace获取最新的模型评测数据。</p>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="footer text-center">
            <p>数据来源: <a href="https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard" target="_blank">HuggingFace Open LLM Leaderboard</a></p>
            <p>© {datetime.now().year} ModelRank AI - <a href="https://github.com/chenjy16/modelrank_ai/blob/main/LICENSE" target="_blank">MIT License</a></p>
        </div>
    </div>
    
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.4/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.4/js/dataTables.bootstrap5.min.js"></script>
    {js_code}
</body>
</html>
"""
    return html_content

async def update_readme():
    """更新 README 文件和 GitHub Pages"""
    logger.info("开始更新 README 文件和 GitHub Pages")
    
    # 获取排行榜数据
    df = await fetch_leaderboard_data()
    
    if df is None:
        logger.error("无法获取数据，更新失败")
        return False
    
    # 生成 Markdown 表格（仅显示前20个模型）
    table = await generate_markdown_table(df, limit=20)
    
    # 创建 GitHub Pages 目录
    docs_dir = Path(__file__).parent.parent / "docs"
    docs_dir.mkdir(exist_ok=True)
    
    # 更新时间
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    # 生成 HTML 页面
    html_content = generate_html_page(df, now)
    
    # 保存 HTML 页面
    index_path = docs_dir / "index.html"
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info(f"HTML 页面已保存到: {index_path}")
    
    # 保存 JSON 和 CSV 数据
    json_path = docs_dir / "leaderboard.json"
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)
    logger.info(f"JSON 数据已保存到: {json_path}")
    
    csv_path = docs_dir / "leaderboard.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"CSV 数据已保存到: {csv_path}")
    
    # 读取现有 README
    readme_path = Path(__file__).parent.parent / "README.md"
    if not readme_path.exists():
        # 如果 README 不存在，创建一个新的
        content = "# ModelRank AI\n\n这是一个自动更新的开源大语言模型排行榜，数据来源于HuggingFace的Open LLM Leaderboard。\n\n## 项目说明\n\n本项目通过GitHub Actions每天自动从HuggingFace获取最新的模型评测数据，并更新到此README中。\n\n"
    else:
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
    
    # 更新 README 中的排行榜部分
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
    
    # 添加完整数据链接（如果不存在）
    if "## 完整数据" not in content:
        content += "\n## 完整数据\n\n"
        content += "完整的排行榜数据可以通过以下方式查看：\n\n"
        content += "- [在线完整排行榜](https://chenjy16.github.io/modelrank_ai/)\n"
        content += "- [JSON 格式数据](https://chenjy16.github.io/modelrank_ai/leaderboard.json)\n"
        content += "- [CSV 格式数据](https://chenjy16.github.io/modelrank_ai/leaderboard.csv)\n\n"
    
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