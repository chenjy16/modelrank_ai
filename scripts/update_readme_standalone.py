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

def format_model_name_html(row):
    """Format model name with HTML links"""
    model_name = row["Model"]
    full_name = row["fullname"]
    
    if pd.isna(full_name) or full_name == "":
        return model_name
    
    return f'<a href="https://huggingface.co/{full_name}" target="_blank">{model_name}</a>'

async def generate_markdown_table(df, limit=20):
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
    if limit:
        top_models = display_df.head(limit)
    else:
        top_models = display_df
    
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

def generate_html_page(df, update_time):
    """Generate complete HTML page"""
    # Create HTML version of DataFrame
    html_df = df.copy()
    
    # Add rank column
    html_df.insert(0, 'Rank', range(1, len(html_df) + 1))
    
    # Format model names as HTML links
    html_df["Model"] = html_df.apply(format_model_name_html, axis=1)
    
    # Rename columns
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
    
    # Select columns to display
    display_columns = ['Rank'] + list(column_rename.keys())
    display_df = html_df[display_columns].copy()
    
    # Rename columns
    display_df = display_df.rename(columns=column_rename)
    
    # Format numeric columns
    for col in display_df.columns:
        if col not in ['Rank', 'Model']:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.2f}" if isinstance(x, (int, float)) and col != "Parameters(B)" else 
                          f"{x:.1f}" if col == "Parameters(B)" and isinstance(x, (int, float)) else x
            )
    
    # Generate HTML table
    html_table = display_df.to_html(
        index=False,
        escape=False,
        classes="table table-striped table-hover table-bordered",
        table_id="leaderboard"
    )
    
    # Create complete HTML page
    # Using triple quotes and r prefix to avoid f-string parsing issues
    js_code = r"""
    <script>
        $(document).ready(function() {
            $('#leaderboard').DataTable({
                "pageLength": 25,
                "order": [[0, "asc"]],
                "language": {
                    "search": "Search:",
                    "lengthMenu": "Show _MENU_ entries",
                    "info": "Showing _START_ to _END_ of _TOTAL_ entries",
                    "infoEmpty": "No entries available",
                    "infoFiltered": "(filtered from _MAX_ total entries)",
                    "paginate": {
                        "first": "First",
                        "last": "Last",
                        "next": "Next",
                        "previous": "Previous"
                    }
                }
            });
        });
    </script>
    """
    
    # Then use this variable in the HTML content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ModelRank AI - Large Language Model Leaderboard</title>
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
                <h1>🏆 ModelRank AI - Large Language Model Leaderboard</h1>
                <p class="text-muted">Last updated: {update_time}</p>
            </div>
            <div class="col-md-4 text-end">
                <a href="https://github.com/chenjy16/modelrank_ai" class="btn btn-outline-dark">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-github" viewBox="0 0 16 16">
                        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
                    </svg>
                    GitHub Repository
                </a>
            </div>
        </div>
        
        <div class="card">
            <div class="card-header bg-light">
                <div class="row">
                    <div class="col">
                        <h5 class="mb-0">Leaderboard Data</h5>
                    </div>
                    <div class="col-auto">
                        <span class="badge bg-primary">Total: {len(display_df)} models</span>
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
                        <h5 class="mb-0">Data Download</h5>
                    </div>
                    <div class="card-body">
                        <p>You can download the complete data via the following links:</p>
                        <a href="leaderboard.json" class="btn btn-outline-primary me-2">JSON Format</a>
                        <a href="leaderboard.csv" class="btn btn-outline-primary">CSV Format</a>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header bg-light">
                        <h5 class="mb-0">About the Project</h5>
                    </div>
                    <div class="card-body">
                        <p>ModelRank AI is an automatically updated open-source large language model leaderboard with data sourced from HuggingFace.</p>
                        <p>This project automatically fetches the latest model evaluation data from HuggingFace daily via GitHub Actions.</p>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="footer text-center">
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
    """Update README file and GitHub Pages"""
    logger.info("Starting to update README file and GitHub Pages")
    
    # Get leaderboard data
    df = await fetch_leaderboard_data()
    
    if df is None:
        logger.error("Unable to fetch data, update failed")
        return False
    
    # Generate Markdown table (showing only top 20 models)
    table = await generate_markdown_table(df, limit=20)
    
    # Create GitHub Pages directory
    docs_dir = Path(__file__).parent.parent / "docs"
    docs_dir.mkdir(exist_ok=True)
    
    # Update time
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    # Generate HTML page
    html_content = generate_html_page(df, now)
    
    # Save HTML page
    index_path = docs_dir / "index.html"
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info(f"HTML page saved to: {index_path}")
    
    # Save JSON and CSV data
    json_path = docs_dir / "leaderboard.json"
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)
    logger.info(f"JSON data saved to: {json_path}")
    
    csv_path = docs_dir / "leaderboard.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"CSV data saved to: {csv_path}")
    
    # Read existing README
    readme_path = Path(__file__).parent.parent / "README.md"
    if not readme_path.exists():
        # If README doesn't exist, create a new one
        content = "# ModelRank AI\n\nThis is an automatically updated open-source large language model leaderboard with data sourced from HuggingFace.\n\n## Project Description\n\nThis project automatically fetches the latest model evaluation data from HuggingFace daily via GitHub Actions and updates this README.\n\n"
    else:
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
    
    # Update the leaderboard section in README
    if "## 🏆 ModelRank AI Leaderboard" in content:
        # Replace existing leaderboard section
        start_marker = "## 🏆 ModelRank AI Leaderboard"
        end_marker = "## "  # Next section starts
        
        start_idx = content.find(start_marker)
        end_idx = content.find(end_marker, start_idx + len(start_marker))
        
        if end_idx == -1:  # If it's the last section
            end_idx = len(content)
        
        new_section = f"{start_marker}\n\n*Last updated: {now}*\n\n{table}\n\n"
        content = content[:start_idx] + new_section + content[end_idx:]
    else:
        # Add new leaderboard section
        content += f"\n## 🏆 ModelRank AI Leaderboard\n\n*Last updated: {now}*\n\n{table}\n\n"
    
    # 检查并替换专业领域排行榜部分
    domain_section_zh = "## 专业领域模型排行榜"
    domain_section_en = "## Domain-Specific Leaderboards"
    
    # 如果存在中文版本，替换为英文版本
    if domain_section_zh in content:
        start_idx = content.find(domain_section_zh)
        next_section_match = re.search(r"## [^#]", content[start_idx:])
        if next_section_match:
            end_idx = start_idx + next_section_match.start()
        else:
            end_idx = len(content)
        
        domain_links = f"{domain_section_en}\n\n"
        domain_links += "Domain-specific model leaderboards can be accessed via the following links:\n\n"
        domain_links += "- [Medical Domain Leaderboard](https://chenjy16.github.io/modelrank_ai/medical_leaderboard.html)\n"
        domain_links += "- [Legal Domain Leaderboard](https://chenjy16.github.io/modelrank_ai/legal_leaderboard.html)\n"
        domain_links += "- [Finance Domain Leaderboard](https://chenjy16.github.io/modelrank_ai/finance_leaderboard.html)\n\n"
        
        content = content[:start_idx] + domain_links + content[end_idx:]
    
    # 如果不存在任何版本，添加英文版本
    elif domain_section_en not in content:
        domain_links = f"\n{domain_section_en}\n\n"
        domain_links += "Domain-specific model leaderboards can be accessed via the following links:\n\n"
        domain_links += "- [Medical Domain Leaderboard](https://chenjy16.github.io/modelrank_ai/medical_leaderboard.html)\n"
        domain_links += "- [Legal Domain Leaderboard](https://chenjy16.github.io/modelrank_ai/legal_leaderboard.html)\n"
        domain_links += "- [Finance Domain Leaderboard](https://chenjy16.github.io/modelrank_ai/finance_leaderboard.html)\n\n"
        
        # 在 Complete Data 部分之前添加
        complete_data_idx = content.find("## Complete Data")
        if complete_data_idx != -1:
            content = content[:complete_data_idx] + domain_links + content[complete_data_idx:]
        else:
            content += domain_links
    
    # Add complete data links (if they don't exist)
    if "## Complete Data" not in content:
        content += "\n## Complete Data\n\n"
        content += "The complete leaderboard data can be viewed through the following methods:\n\n"
        content += "- [Online Complete Leaderboard](https://chenjy16.github.io/modelrank_ai/)\n"
        content += "- [JSON Format Data](https://chenjy16.github.io/modelrank_ai/leaderboard.json)\n"
        content += "- [CSV Format Data](https://chenjy16.github.io/modelrank_ai/leaderboard.csv)\n\n"
    
    # Add data source explanation (if it doesn't exist)
    if "## Data Source" not in content:
        content += "\n## Data Source\n\nData is sourced from HuggingFace.\n\n"
    
    # Add license information (if it doesn't exist)
    if "## License" not in content:
        content += "\n## License\n\n"
        license_section_en = "## License"
        license_section_zh = "## 许可证"
        
        if license_section_zh in content:
            start_idx = content.find(license_section_zh)
            next_section_match = re.search(r"## [^#]", content[start_idx:])
            if next_section_match:
                end_idx = start_idx + next_section_match.start()
            else:
                end_idx = len(content)
            
            license_content = f"{license_section_en}\n\n"
            license_content += "This project is open-sourced under the MIT License.\n\n"
            
            content = content[:start_idx] + license_content + content[end_idx:]
        
        content += "This project is open-sourced under the MIT License.\n"
    
    # Write back to README
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    logger.info(f"README updated successfully, time: {now}")
    
    # 初始化 domain_success 变量
    domain_success = True
    
    # 处理专业领域排行榜
    for domain in ["medical", "legal", "finance"]:
        try:
            # 获取领域数据
            domain_df = await fetch_domain_leaderboard_data(domain)
            
            if domain_df is not None and len(domain_df) > 0:
                # 生成领域 Markdown 表格
                domain_table = await generate_domain_markdown_table(domain_df, domain)
                
                # 更新 README 中的领域表格
                domain_success = await update_readme_with_domain(readme_path, domain, domain_table) and domain_success
                
                # 生成领域 HTML 页面
                domain_html = generate_domain_html_page(domain_df, domain, now)
                
                # 保存领域 HTML 页面
                domain_html_path = docs_dir / f"{domain}_leaderboard.html"
                with open(domain_html_path, "w", encoding="utf-8") as f:
                    f.write(domain_html)
                logger.info(f"Domain HTML page saved to: {domain_html_path}")
                
                # 保存领域数据文件
                domain_success = save_domain_data_files(domain_df, domain, docs_dir) and domain_success
            else:
                logger.warning(f"No data available for {domain} domain")
                domain_success = False
        except Exception as e:
            logger.error(f"Error processing {domain} domain: {str(e)}")
            domain_success = False
    
    return domain_success and True

async def fetch_domain_leaderboard_data(domain):
    """获取特定领域的模型评估数据"""
    logger.info(f"Fetching {domain} domain leaderboard data...")
    
    try:
        # 获取所有模型数据
        all_models_df = await fetch_leaderboard_data()
        if all_models_df is None:
            return None
        
        # 创建基础数据框架
        models_info = all_models_df[["fullname", "Model", "#Params (B)"]].copy()
        
        # 定义领域对应的评估指标（从主排行榜中选择相关指标）
        domain_metrics = {
            "medical": {
                "MMLU-PRO": 0.7,  # 权重
                "BBH": 0.3,       # 权重
            },
            "legal": {
                "MMLU-PRO": 0.6,  # 权重
                "BBH": 0.4,       # 权重
            },
            "finance": {
                "MATH": 0.5,      # 权重
                "MMLU-PRO": 0.5,  # 权重
            }
        }
        
        if domain not in domain_metrics:
            logger.error(f"Unknown domain: {domain}")
            return None
        
        # 获取该领域的评估指标
        metrics = domain_metrics[domain]
        
        # 计算领域得分
        domain_score = pd.Series(0, index=models_info.index)
        valid_metrics = 0
        
        for metric, weight in metrics.items():
            if metric in all_models_df.columns:
                # 将指标列添加到模型信息中
                models_info[metric] = all_models_df[metric]
                # 累加加权得分
                domain_score += all_models_df[metric].fillna(0) * weight
                valid_metrics += weight
                logger.info(f"Using {metric} for {domain} domain with weight {weight}")
        
        # 如果没有有效指标，返回None
        if valid_metrics == 0:
            logger.warning(f"No valid metrics found for {domain} domain")
            return None
        
        # 计算最终的领域平均分
        models_info["domain_average"] = domain_score / valid_metrics
        
        # 排序并过滤掉没有领域评分的模型
        models_info = models_info.dropna(subset=["domain_average"])
        models_info = models_info.sort_values("domain_average", ascending=False)
        
        if len(models_info) > 0:
            logger.info(f"✅ Successfully retrieved {domain} domain leaderboard data, total: {len(models_info)} models")
            return models_info
        else:
            logger.warning(f"No valid data found for {domain} domain")
            return None
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch {domain} domain leaderboard data: {str(e)}")
        return None

async def generate_domain_markdown_table(df, domain, limit=20):
    """生成专业领域排行榜 Markdown 表格"""
    if df is None or len(df) == 0:
        return "No data available for this domain"
    
    # 获取数据集列名
    dataset_columns = [col for col in df.columns if col not in ["fullname", "Model", "#Params (B)", "domain_average"]]
    
    # 选择要显示的列
    columns = ["Model", "domain_average", "#Params (B)"] + dataset_columns
    
    # 创建一个新的 DataFrame，只包含我们需要的列
    display_df = df[columns].copy()
    
    # 格式化模型名称
    display_df["Model"] = df.apply(format_model_name, axis=1)
    
    # 重命名列以便更好地显示
    column_rename = {
        "Model": "Model",
        "domain_average": "Average Score",
        "#Params (B)": "Parameters(B)"
    }
    
    # 添加数据集列的重命名
    dataset_display_names = {
        "medmcqa": "MedMCQA",
        "pubmedqa": "PubMedQA",
        "mmlu_medical_genetics": "MMLU-Medical",
        "mmlu_clinical_knowledge": "MMLU-Clinical",
        "lextreme": "LexTreme",
        "mmlu_jurisprudence": "MMLU-Law",
        "mmlu_professional_law": "MMLU-Prof Law",
        "finqa": "FinQA",
        "mmlu_econometrics": "MMLU-Econometrics",
        "mmlu_global_economics": "MMLU-Economics"
    }
    
    for dataset in dataset_columns:
        column_rename[dataset] = dataset_display_names.get(dataset, dataset.capitalize())
    
    display_df = display_df.rename(columns=column_rename)
    
    # 限制要显示的行数
    if limit and len(display_df) > limit:
        top_models = display_df.head(limit)
    else:
        top_models = display_df
    
    # 生成 Markdown 表格
    markdown_table = "| Rank | " + " | ".join(top_models.columns) + " |\n"
    markdown_table += "| --- | " + " | ".join(["---"] * len(top_models.columns)) + " |\n"
    
    for i, (_, row) in enumerate(top_models.iterrows(), 1):
        # 格式化数字，使用适当的小数位数
        formatted_row = []
        for col, value in row.items():
            if pd.isna(value):
                formatted_row.append("-")
            elif isinstance(value, (int, float)) and col != "Parameters(B)":
                formatted_row.append(f"{value:.2f}")
            elif col == "Parameters(B)" and isinstance(value, (int, float)):
                formatted_row.append(f"{value:.1f}")
            else:
                formatted_row.append(str(value))
        
        markdown_table += f"| {i} | " + " | ".join(formatted_row) + " |\n"
    
    return markdown_table

def generate_domain_html_page(df, domain, update_time):
    """生成专业领域排行榜 HTML 页面"""
    if df is None or len(df) == 0:
        return f"<p>No data available for {domain} domain</p>"
    
    # 创建 HTML 版本的 DataFrame
    html_df = df.copy()
    
    # 添加排名列
    html_df.insert(0, 'Rank', range(1, len(html_df) + 1))
    
    # 格式化模型名称为 HTML 链接
    html_df["Model"] = html_df.apply(format_model_name_html, axis=1)
    
    # 获取数据集列名
    dataset_columns = [col for col in df.columns if col not in ["fullname", "Model", "#Params (B)", "domain_average"]]
    
    # 选择要显示的列
    display_columns = ['Rank', 'Model', 'domain_average', '#Params (B)'] + dataset_columns
    display_df = html_df[display_columns].copy()
    
    # 重命名列
    column_rename = {
        "Model": "Model",
        "domain_average": "Average Score",
        "#Params (B)": "Parameters(B)"
    }
    
    # 添加数据集列的重命名
    dataset_display_names = {
        "medmcqa": "MedMCQA",
        "pubmedqa": "PubMedQA",
        "mmlu_medical_genetics": "MMLU-Medical",
        "mmlu_clinical_knowledge": "MMLU-Clinical",
        "lextreme": "LexTreme",
        "mmlu_jurisprudence": "MMLU-Law",
        "mmlu_professional_law": "MMLU-Prof Law",
        "finqa": "FinQA",
        "mmlu_econometrics": "MMLU-Econometrics",
        "mmlu_global_economics": "MMLU-Economics"
    }
    
    for dataset in dataset_columns:
        column_rename[dataset] = dataset_display_names.get(dataset, dataset.capitalize())
    
    display_df = display_df.rename(columns=column_rename)
    
    # 格式化数字列
    for col in display_df.columns:
        if col not in ['Rank', 'Model']:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.2f}" if isinstance(x, (int, float)) and not pd.isna(x) and col != "Parameters(B)" else 
                          f"{x:.1f}" if col == "Parameters(B)" and isinstance(x, (int, float)) and not pd.isna(x) else 
                          "-" if pd.isna(x) else x
            )
    
    # 生成 HTML 表格
    html_table = display_df.to_html(
        index=False,
        escape=False,
        classes="table table-striped table-hover table-bordered",
        table_id=f"{domain}_leaderboard",
        na_rep="-"
    )
    
    # 创建完整的 HTML 页面
    domain_titles = {
        "medical": "Medical Domain",
        "legal": "Legal Domain",
        "finance": "Finance Domain"
    }
    
    domain_title = domain_titles.get(domain, domain)
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ModelRank AI - {domain_title} Leaderboard</title>
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
                <h1>🏆 ModelRank AI - {domain_title} Leaderboard</h1>
                <p class="text-muted">Last updated: {update_time}</p>
            </div>
            <div class="col-md-4 text-end">
                <a href="index.html" class="btn btn-outline-primary me-2">Back to Main Leaderboard</a>
                <a href="https://github.com/chenjy16/modelrank_ai" class="btn btn-outline-dark">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-github" viewBox="0 0 16 16">
                        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
                    </svg>
                    GitHub Repository
                </a>
            </div>
        </div>
        
        <div class="card">
            <div class="card-header bg-light">
                <div class="row">
                    <div class="col">
                        <h5 class="mb-0">{domain_title} Leaderboard Data</h5>
                    </div>
                    <div class="col-auto">
                        <span class="badge bg-primary">Total: {len(display_df)} models</span>
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
                        <h5 class="mb-0">Data Download</h5>
                    </div>
                    <div class="card-body">
                        <p>You can download the complete data via the following links:</p>
                        <a href="{domain}_leaderboard.json" class="btn btn-outline-primary me-2">JSON Format</a>
                        <a href="{domain}_leaderboard.csv" class="btn btn-outline-primary">CSV Format</a>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header bg-light">
                        <h5 class="mb-0">About the Project</h5>
                    </div>
                    <div class="card-body">
                        <p>ModelRank AI is an automatically updated open-source large language model leaderboard with data sourced from HuggingFace.</p>
                        <p>This project automatically fetches the latest model evaluation data from HuggingFace daily via GitHub Actions.</p>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="footer text-center">
            <p>© {datetime.now().year} ModelRank AI - <a href="https://github.com/chenjy16/modelrank_ai/blob/main/LICENSE" target="_blank">MIT License</a></p>
        </div>
    </div>
    
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.4/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.4/js/dataTables.bootstrap5.min.js"></script>
    <script>
        $(document).ready(function() {{
            $('#{domain}_leaderboard').DataTable({{
                "pageLength": 25,
                "order": [[0, "asc"]],
                "language": {{
                    "search": "Search:",
                    "lengthMenu": "Show _MENU_ entries",
                    "info": "Showing _START_ to _END_ of _TOTAL_ entries",
                    "infoEmpty": "No entries available",
                    "infoFiltered": "(filtered from _MAX_ total entries)",
                    "paginate": {{
                        "first": "First",
                        "last": "Last",
                        "next": "Next",
                        "previous": "Previous"
                    }}
                }}
            }});
        }});
    </script>
</body>
</html>
"""
    return html_content

async def update_readme_with_domain(readme_path, domain, domain_table):
    """更新 README.md 文件中的专业领域排行榜部分"""
    try:
        # 读取现有的 README.md 文件
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # 定义领域标题映射 (修改为英文)
        domain_titles = {
            "medical": "🏥 Medical Domain Leaderboard",
            "legal": "⚖️ Legal Domain Leaderboard",
            "finance": "💰 Finance Domain Leaderboard"
        }
        
        domain_section_start = f"## {domain_titles.get(domain, f'Domain Leaderboard: {domain}')}"
        
        # 查找下一个章节的开始位置
        next_section_pattern = r"## [^#]"
        
        if domain_section_start in content:
            # 如果已经存在，则替换该部分
            start_idx = content.find(domain_section_start)
            
            # 查找下一个章节的开始
            matches = list(re.finditer(next_section_pattern, content[start_idx + len(domain_section_start):]))
            if matches:
                end_idx = start_idx + len(domain_section_start) + matches[0].start()
            else:
                end_idx = len(content)
            
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
            new_section = f"{domain_section_start}\n\n*Last updated: {now}*\n\n{domain_table}\n\n"
            content = content[:start_idx] + new_section + content[end_idx:]
        else:
            # 如果不存在，则在主排行榜后面添加
            complete_data_section = "## Complete Data"
            complete_data_idx = content.find(complete_data_section)
            
            if complete_data_idx != -1:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
                new_section = f"{domain_section_start}\n\n*Last updated: {now}*\n\n{domain_table}\n\n"
                content = content[:complete_data_idx] + new_section + content[complete_data_idx:]
            else:
                # 如果找不到 Complete Data 部分，则添加到文件末尾
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
                content += f"\n\n{domain_section_start}\n\n*Last updated: {now}*\n\n{domain_table}\n\n"
        
        # 写回 README.md 文件
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        logger.info(f"✅ Successfully updated {domain} domain leaderboard in README.md")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to update {domain} domain leaderboard in README.md: {str(e)}")
        return False

def save_domain_data_files(df, domain, docs_dir):
    """保存专业领域排行榜数据文件"""
    try:
        if df is None or len(df) == 0:
            logger.warning(f"No data available for {domain} domain export")
            return False
            
        # 导出为 JSON
        json_path = docs_dir / f"{domain}_leaderboard.json"
        df.to_json(json_path, orient="records", force_ascii=False, indent=2)
        logger.info(f"JSON data saved to: {json_path}")
        
        # 导出为 CSV
        csv_path = docs_dir / f"{domain}_leaderboard.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"CSV data saved to: {csv_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to save {domain} domain leaderboard data files: {str(e)}")
        return False


async def main():
    """Main function"""
    try:
        success = await update_readme()
        if success:
            logger.info("✅ Leaderboard update successful")
            sys.exit(0)
        else:
            logger.error("❌ Leaderboard update failed")
            sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
