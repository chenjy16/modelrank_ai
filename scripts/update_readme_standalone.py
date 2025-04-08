#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import logging
import asyncio
import pandas as pd
import re
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download


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

# --- Global Constants ---
# Define JS Code globally for reusability
DATATABLES_JS_CODE = r"""
<script>
    $(document).ready(function() {
        $('#leaderboard').DataTable({
            "pageLength": 25,
            "order": [[0, "asc"]], // Assuming Rank is the first column
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

async def fetch_leaderboard_data():
    """使用 HuggingFace API 直接获取排行榜数据"""
    logger.info("获取排行榜数据...")
    
    try:
        # 检查 token 是否存在
        if not HF_TOKEN:
            logger.error("环境变量中未找到 HF_TOKEN")
            return None
        
        # 获取仓库中的文件列表
        logger.info(f"正在获取 {CONTENTS_REPO} 仓库中的文件列表...")
        repo_files = api.list_repo_files(
            repo_id=CONTENTS_REPO,
            repo_type="dataset"
        )
        
        logger.info(f"在仓库中找到 {len(repo_files)} 个文件")
        
        # 查找数据文件（jsonl, csv, parquet 等）
        data_files = [f for f in repo_files if f.endswith('.jsonl') or f.endswith('.csv') or f.endswith('.parquet')]
        
        if not data_files:
            logger.error("仓库中未找到数据文件")
            return None
            
        logger.info(f"找到数据文件: {data_files}")
        
        # 下载第一个数据文件
        data_file = data_files[0]
        logger.info(f"正在下载文件: {data_file}")
        
        try:
            local_path = hf_hub_download(
                repo_id=CONTENTS_REPO,
                filename=data_file,
                repo_type="dataset",
                token=HF_TOKEN
            )
            
            logger.info(f"文件已下载到: {local_path}")
            
            # 根据文件类型加载数据
            if data_file.endswith('.jsonl'):
                rows = []
                with open(local_path, 'r') as f:
                    for line in f:
                        rows.append(json.loads(line))
                df = pd.DataFrame(rows)
            elif data_file.endswith('.csv'):
                df = pd.read_csv(local_path)
            elif data_file.endswith('.parquet'):
                df = pd.read_parquet(local_path)
            else:
                logger.error(f"不支持的文件格式: {data_file}")
                return None
                
            # 按平均分数排序
            if "Average ⬆️" in df.columns:
                df = df.sort_values(by="Average ⬆️", ascending=False)
                logger.info(f"成功获取并按 'Average ⬆️' 排序 {len(df)} 个模型条目")
            else:
                logger.warning("未找到 'Average ⬆️' 列。数据已获取但未按平均值排序。")
                # 如果没有平均值列，则按模型名称排序
                if "Model" in df.columns:
                    df = df.sort_values(by="Model", ascending=True)
            
            return df
            
        except Exception as download_error:
            logger.error(f"下载文件时出错: {str(download_error)}", exc_info=True)
            
            # 尝试列出可用的数据集以验证连接
            try:
                available_datasets = api.list_datasets(author=HF_ORGANIZATION, limit=5)
                logger.info(f"{HF_ORGANIZATION} 的可用数据集: {[ds.id for ds in available_datasets]}")
            except Exception as list_error:
                logger.error(f"列出可用数据集失败: {str(list_error)}")
            
            return None
            
    except Exception as e:
        logger.error(f"获取数据失败: {str(e)}", exc_info=True)
        
        # 添加更多诊断信息
        logger.error("诊断信息:")
        logger.error(f"- HF_TOKEN 可用: {bool(HF_TOKEN)}")
        logger.error(f"- 数据集路径: {CONTENTS_REPO}")
        
        return None

# 添加一个备用方法，使用本地数据
async def fetch_local_leaderboard_data():
    """从本地文件获取排行榜数据"""
    logger.info("尝试从本地文件获取排行榜数据...")
    
    try:
        # 检查本地数据文件
        local_data_path = Path(__file__).parent.parent / "docs" / "leaderboard.csv"
        
        if local_data_path.exists():
            logger.info(f"使用本地数据: {local_data_path}")
            df = pd.read_csv(local_data_path)
            
            # 按平均分数排序
            if "Average ⬆️" in df.columns:
                df = df.sort_values(by="Average ⬆️", ascending=False)
                
            logger.info(f"成功从本地文件获取 {len(df)} 个模型条目")
            return df
        else:
            logger.error(f"本地数据文件不存在: {local_data_path}")
            return None
    except Exception as e:
        logger.error(f"从本地文件获取数据失败: {str(e)}")
        return None

def format_model_name(row):
    """Format model name with links"""
    model_name = row.get("Model", "") # Use .get for safety
    full_name = row.get("fullname", "") # Use .get for safety

    if pd.isna(full_name) or full_name == "":
        return model_name

    # Ensure model_name and full_name are strings before formatting
    model_name_str = str(model_name)
    full_name_str = str(full_name)

    return f"[{model_name_str}](https://huggingface.co/{full_name_str})"

def format_model_name_html(row):
    """Format model name with HTML links"""
    model_name = row.get("Model", "") # Use .get for safety
    full_name = row.get("fullname", "") # Use .get for safety

    if pd.isna(full_name) or full_name == "":
        return str(model_name) # Ensure return is string

    # Ensure model_name and full_name are strings before formatting
    model_name_str = str(model_name)
    full_name_str = str(full_name)

    return f'<a href="https://huggingface.co/{full_name_str}" target="_blank">{model_name_str}</a>'

async def generate_markdown_table(df, limit=20):
    """Generate Markdown table"""
    if df is None or df.empty: # Check if DataFrame is empty
        return "No data available"

    # Select columns to display
    columns_to_try = [
        "Model", "Average ⬆️", "#Params (B)",
        "IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"
    ]

    # Filter for columns that actually exist in the DataFrame
    columns = [col for col in columns_to_try if col in df.columns]
    if not columns:
        logger.warning("No displayable columns found in the DataFrame.")
        return "No displayable data columns found."
    if "Model" not in columns:
        logger.warning("'Model' column missing, cannot generate table properly.")
        return "Essential 'Model' column is missing."


    # Create a new DataFrame with only the columns we need
    display_df = df[columns].copy()

    # Format model names
    if 'fullname' in df.columns:
        display_df["Model"] = df.apply(format_model_name, axis=1)
    else:
        logger.warning("'fullname' column missing, model names will not be links.")
        display_df["Model"] = df["Model"] # Keep original model name

    # Rename columns for better display
    column_rename = {
        "Model": "Model",
        "Average ⬆️": "Avg", # Shorter name
        "#Params (B)": "Params(B)", # Shorter name
        "IFEval": "IFEval",
        "BBH": "BBH",
        "MATH Lvl 5": "MATH",
        "GPQA": "GPQA",
        "MUSR": "MUSR",
        "MMLU-PRO": "MMLU-PRO"
    }

    # Apply renaming only for columns that exist in display_df
    display_df = display_df.rename(columns={k: v for k, v in column_rename.items() if k in display_df.columns})

    # Limit the number of rows to display
    if limit and len(display_df) > limit:
        top_models = display_df.head(limit)
    else:
        top_models = display_df

    # Get the final column names after renaming
    final_columns = top_models.columns.tolist()

    # Generate Markdown table header
    markdown_table = "| Rank | " + " | ".join(final_columns) + " |\n"
    markdown_table += "| --- | " + " | ".join(["---"] * len(final_columns)) + " |\n"

    # Generate Markdown table rows
    for i, (_, row) in enumerate(top_models.iterrows(), 1):
        formatted_row = []
        for col in final_columns: # Iterate using final column names
            value = row[col]
            if pd.isna(value):
                formatted_row.append("-")
            # Use renamed column names for formatting logic
            elif isinstance(value, (int, float)) and col != "Params(B)":
                formatted_row.append(f"{value:.2f}")
            elif col == "Params(B)" and isinstance(value, (int, float)):
                # Handle potential integer params cleanly
                 formatted_row.append(f"{value:.1f}" if value % 1 != 0 else f"{int(value)}")
            else:
                formatted_row.append(str(value))

        markdown_table += f"| {i} | " + " | ".join(formatted_row) + " |\n"

    return markdown_table

# --- NEW FUNCTION: generate_html_page (for index.html) ---
def generate_html_page(df, update_time):
    """Generate the main leaderboard HTML page (index.html)"""
    if df is None or df.empty:
        return "<p>No data available</p>"

    # Create HTML version DataFrame
    html_df = df.copy()

    # Add Rank column
    html_df.insert(0, 'Rank', range(1, len(html_df) + 1))

    # Format model names with HTML links if 'fullname' exists
    if 'fullname' in html_df.columns:
        html_df["Model"] = html_df.apply(format_model_name_html, axis=1)
    else:
        logger.warning("'fullname' column missing for HTML page, model names will not be links.")
        # Ensure Model column is string
        html_df["Model"] = html_df["Model"].astype(str)


    # Select and order columns for display
    columns_to_try = [
        "Rank", "Model", "Average ⬆️", "#Params (B)",
        "IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"
    ]
    display_columns = [col for col in columns_to_try if col in html_df.columns]

    if not display_columns:
         logger.error("No displayable columns found for HTML page.")
         return "<p>Error: No data columns to display.</p>"

    display_df = html_df[display_columns].copy()

    # Rename columns for display
    column_rename = {
        "Rank": "Rank",
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
    display_df = display_df.rename(columns={k: v for k, v in column_rename.items() if k in display_df.columns})

    # Format numeric columns
    final_columns = display_df.columns.tolist()
    for col in final_columns:
        if col not in ['Rank', 'Model']:
            # Apply formatting, handling potential non-numeric gracefully
            try:
                 display_df[col] = display_df[col].apply(
                    lambda x: f"{float(x):.2f}" if pd.notna(x) and isinstance(x, (int, float)) and col != "Parameters(B)" else
                              (f"{float(x):.1f}" if float(x) % 1 != 0 else f"{int(float(x))}") if pd.notna(x) and col == "Parameters(B)" and isinstance(x, (int, float)) else
                              x if pd.notna(x) else "-" # Handle NaN/None gracefully
                 )
            except (ValueError, TypeError):
                 logger.warning(f"Could not format column '{col}' as numeric. Keeping original values.")
                 display_df[col] = display_df[col].astype(str) # Ensure it's string if formatting fails


    # Generate HTML table
    html_table = display_df.to_html(
        index=False,
        escape=False, # Allow HTML links in 'Model' column
        classes="table table-striped table-hover table-bordered",
        table_id="leaderboard",
        na_rep="-" # Representation for NaN values
    )

    # Create the full HTML page content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ModelRank AI - Open LLM Leaderboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.datatables.net/1.13.4/css/dataTables.bootstrap5.min.css">
    <style>
        body {{
            padding: 20px;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }}
        .container {{ max-width: 1600px; }} /* Wider container */
        h1 {{ margin-bottom: 20px; color: #333; }}
        .table {{ width: 100%; }}
        .table th {{
            position: sticky;
            top: 0;
            background-color: #f8f9fa;
            color: #333;
            font-weight: 600;
            white-space: nowrap; /* Prevent header wrapping */
        }}
        .table td {{ vertical-align: middle; white-space: nowrap; }} /* Prevent cell wrapping */
        .footer {{
            margin-top: 30px;
            padding-top: 10px;
            border-top: 1px solid #eee;
            color: #666;
            font-size: 0.9rem;
        }}
        .card {{ margin-bottom: 20px; }}
        .badge {{ font-size: 0.8rem; }}
        .table-responsive {{ overflow-x: auto; }} /* Ensure horizontal scroll */
    </style>
</head>
<body>
    <div class="container">
        <div class="row mb-4 align-items-center">
            <div class="col-md-8">
                <h1>🏆 ModelRank AI - Open LLM Leaderboard</h1>
                <p class="text-muted mb-0">Last updated: {update_time}</p>
            </div>
            <div class="col-md-4 text-end">
                 <a href="https://github.com/chenjy16/modelrank_ai" class="btn btn-outline-dark" target="_blank">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-github" viewBox="0 0 16 16">
                        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
                    </svg> GitHub
                </a>
            </div>
        </div>

        <div class="row mb-3">
            <div class="col">
                 <div class="card">
                     <div class="card-header bg-light">
                         <h5 class="mb-0">Domain-Specific Leaderboards</h5>
                     </div>
                     <div class="card-body">
                         <a href="medical_leaderboard.html" class="btn btn-outline-primary me-2 mb-2">🏥 Medical</a>
                         <a href="legal_leaderboard.html" class="btn btn-outline-primary me-2 mb-2">⚖️ Legal</a>
                         <a href="finance_leaderboard.html" class="btn btn-outline-primary mb-2">💰 Finance</a>
                     </div>
                 </div>
            </div>
             <div class="col">
                 <div class="card">
                     <div class="card-header bg-light">
                         <h5 class="mb-0">Data Download</h5>
                     </div>
                     <div class="card-body">
                         <a href="leaderboard.json" class="btn btn-outline-secondary me-2 mb-2" download>JSON</a>
                         <a href="leaderboard.csv" class="btn btn-outline-secondary mb-2" download>CSV</a>
                     </div>
                 </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header bg-light">
                <div class="row">
                    <div class="col">
                        <h5 class="mb-0">Main Leaderboard</h5>
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

        <div class="footer text-center">
            <p>© {datetime.now().year} ModelRank AI - <a href="https://github.com/chenjy16/modelrank_ai/blob/main/LICENSE" target="_blank">MIT License</a></p>
             <p>Data sourced from <a href="https://huggingface.co/spaces/{CONTENTS_REPO}" target="_blank">HuggingFace Open LLM Leaderboard</a>.</p>
        </div>
    </div>

    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.4/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.4/js/dataTables.bootstrap5.min.js"></script>
    {DATATABLES_JS_CODE}
</body>
</html>""" # Use the global JS code constant

    return html_content

# --- CORRECTED: update_readme_with_domain ---
async def update_readme_with_domain(readme_path, domain, table_md, update_time):
    """Update README with a specific domain's leaderboard section."""
    logger.info(f"Updating README for {domain} domain section...")
    try:
        # Read existing README
        if not readme_path.exists():
             logger.error(f"README file not found at {readme_path} during domain update.")
             return False # Cannot update if file doesn't exist

        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Define domain titles and markers
        domain_titles = {
            "medical": "🏥 Medical Domain Leaderboard",
            "legal": "⚖️ Legal Domain Leaderboard",
            "finance": "💰 Finance Domain Leaderboard"
        }
        domain_title = domain_titles.get(domain, f"{domain.capitalize()} Domain Leaderboard")
        start_marker = f"## {domain_title}"
        end_marker = "\n## " # Start of next H2

        # Prepare the new section content
        new_section = f"{start_marker}\n\n*Top 10 models shown. Last updated: {update_time}*\n\n{table_md}\n" # table_md includes the link now

        start_idx = content.find(start_marker)
        if start_idx != -1:
            logger.info(f"Found existing section for {domain}, replacing.")
            # Find end of the section
            end_idx = content.find(end_marker, start_idx + len(start_marker))
            if end_idx == -1:
                 logger.info(f"{domain} section seems to be the last one.")
                 end_idx = len(content)
            else:
                 # Adjust end_idx to be right before the next section marker
                 end_idx_adjusted = content.rfind('\n', start_idx, end_idx) + 1
                 if end_idx_adjusted == 0: # Fallback if rfind fails
                    end_idx_adjusted = end_idx
                 end_idx = end_idx_adjusted


            content = content[:start_idx] + new_section + content[end_idx:]
        else:
            logger.info(f"Section for {domain} not found, adding it after Domain-Specific Links.")
            # Try to insert after the general domain links section
            links_title = "🌐 Domain-Specific Leaderboards"
            links_start_marker = f"## {links_title}"
            links_start_idx = content.find(links_start_marker)

            if links_start_idx != -1:
                 # Find the end of the links section
                 links_end_idx = content.find(end_marker, links_start_idx + len(links_start_marker))
                 if links_end_idx == -1:
                      insert_pos = len(content) # Append if links section is last
                 else:
                      insert_pos = links_end_idx
                 content = content[:insert_pos] + f"\n{new_section}" + content[insert_pos:]
            else:
                 logger.warning("Could not find 'Domain-Specific Leaderboards' section to insert after. Appending domain section to end.")
                 content += f"\n{new_section}"

        # Write back to README
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"✅ Successfully updated README for {domain} domain.")
        return True # Indicate success

    except Exception as e:
        logger.error(f"❌ Failed to update README for {domain} domain: {str(e)}", exc_info=True)
        return False # Indicate failure


# --- Main execution block ---
if __name__ == "__main__":
    # Ensure event loop exists for platforms like Windows
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # No running event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    logger.info("Script started.")
    success = loop.run_until_complete(update_readme())
    # success = asyncio.run(update_readme()) # Simpler call if loop handling isn't needed

    if success:
        logger.info("✅ Script finished successfully.")
        sys.exit(0)
    else:
        logger.error("❌ Script finished with errors.")
        sys.exit(1)