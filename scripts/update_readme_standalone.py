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
from huggingface_hub import HfApi
import datasets
# Removed duplicate json import

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
    """Fetch leaderboard data from HuggingFace"""
    logger.info("Fetching leaderboard data...")

    try:
        # Disable progress bar
        datasets.disable_progress_bar()

        # Load dataset
        dataset = datasets.load_dataset(CONTENTS_REPO, split="train") # Specify split

        # Convert to pandas DataFrame
        df = dataset.to_pandas()

        # Ensure 'Average ⬆️' exists before sorting
        if "Average ⬆️" in df.columns:
            # Sort by average score
            df = df.sort_values(by="Average ⬆️", ascending=False)
            logger.info(f"Successfully retrieved and sorted {len(df)} model entries by 'Average ⬆️'")
        else:
            logger.warning("'Average ⬆️' column not found. Data retrieved but not sorted by average.")
            # Fallback: Sort by model name if average is missing
            if "Model" in df.columns:
                 df = df.sort_values(by="Model", ascending=True)

        return df
    except Exception as e:
        logger.error(f"Failed to fetch data: {str(e)}", exc_info=True) # Add exc_info for traceback
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

# --- NEW FUNCTION: save_domain_data_files ---
def save_domain_data_files(df, domain, output_dir):
    """Save domain-specific data to JSON and CSV files."""
    if df is None or df.empty:
        logger.warning(f"No data to save for domain: {domain}")
        return False # Indicate failure

    json_path = output_dir / f"{domain}_leaderboard.json"
    csv_path = output_dir / f"{domain}_leaderboard.csv"

    try:
        # Save JSON
        # Ensure 'Rank' exists before saving, if not, add it
        if 'Rank' not in df.columns:
             df.insert(0, 'Rank', range(1, len(df) + 1))
        df.to_json(json_path, orient="records", force_ascii=False, indent=2)
        logger.info(f"Domain JSON data saved to: {json_path}")

        # Save CSV
        df.to_csv(csv_path, index=False)
        logger.info(f"Domain CSV data saved to: {csv_path}")
        return True # Indicate success
    except Exception as e:
        logger.error(f"Failed to save data files for domain {domain}: {str(e)}", exc_info=True)
        return False # Indicate failure


async def update_readme():
    """Update README file and GitHub Pages"""
    logger.info("Starting to update README file and GitHub Pages")

    # Get leaderboard data
    df = await fetch_leaderboard_data()

    if df is None:
        logger.error("Unable to fetch data, update failed")
        return False

    # Create GitHub Pages directory if it doesn't exist
    # Assumes script is in a subdirectory like 'scripts' or 'src'
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    docs_dir = project_root / "docs"
    docs_dir.mkdir(exist_ok=True)
    logger.info(f"Docs directory ensured at: {docs_dir}")

    # Update time
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    # --- Main Leaderboard Section ---
    # Generate Markdown table (showing only top 20 models)
    table = await generate_markdown_table(df, limit=20)

    # Generate main HTML page (index.html)
    html_content = generate_html_page(df, now)

    # Save main HTML page
    index_path = docs_dir / "index.html"
    try:
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"Main HTML page saved to: {index_path}")
    except Exception as e:
         logger.error(f"Failed to save main HTML page: {e}", exc_info=True)
         return False # Stop if essential file saving fails


    # Save main JSON and CSV data
    json_path = docs_dir / "leaderboard.json"
    csv_path = docs_dir / "leaderboard.csv"
    try:
        df.to_json(json_path, orient="records", force_ascii=False, indent=2)
        logger.info(f"Main JSON data saved to: {json_path}")
        df.to_csv(csv_path, index=False)
        logger.info(f"Main CSV data saved to: {csv_path}")
    except Exception as e:
         logger.error(f"Failed to save main data files: {e}", exc_info=True)
         # Decide if this is critical - maybe continue updating README? Let's continue for now.


    # --- README Update Section ---
    readme_path = project_root / "README.md"
    logger.info(f"Attempting to update README at: {readme_path}")
    try:
        if not readme_path.exists():
            logger.warning(f"README.md not found at {readme_path}. Creating a new one.")
            content = f"# ModelRank AI\n\nThis is an automatically updated open-source large language model leaderboard.\n\n*Initial content generated on {now}*\n\n"
        else:
            with open(readme_path, "r", encoding="utf-8") as f:
                content = f.read()

        # --- Update Main Leaderboard in README ---
        main_lb_title = "🏆 ModelRank AI Leaderboard"
        start_marker_main = f"## {main_lb_title}"
        end_marker_main = "\n## " # Start of the next H2 section
        new_section_main = f"{start_marker_main}\n\n*Last updated: {now}*\n\n{table}\n\n[View Complete Online Leaderboard](https://chenjy16.github.io/modelrank_ai/)\n"

        start_idx_main = content.find(start_marker_main)
        if start_idx_main != -1:
            logger.info("Found existing main leaderboard section in README.")
            # Find the end of the section (next H2 or end of file)
            end_idx_main = content.find(end_marker_main, start_idx_main + len(start_marker_main))
            if end_idx_main == -1:
                logger.info("Main leaderboard seems to be the last section.")
                end_idx_main = len(content)
            else:
                 # Adjust end_idx to be right before the next section marker
                 end_idx_main = content.rfind('\n', start_idx_main, end_idx_main) + 1 # Keep the newline before next section
                 if end_idx_main == 0: # If rfind fails, fallback
                    end_idx_main = content.find(end_marker_main, start_idx_main + len(start_marker_main))

            content = content[:start_idx_main] + new_section_main + content[end_idx_main:]
            logger.info("Main leaderboard section updated.")
        else:
            logger.info("Main leaderboard section not found, appending to README.")
            content += f"\n{new_section_main}" # Append if not found

        # --- Ensure Domain-Specific Links Section ---
        domain_links_title = "🌐 Domain-Specific Leaderboards"
        start_marker_domain_links = f"## {domain_links_title}"
        end_marker_domain_links = "\n## "
        domain_links_content = (
            f"{start_marker_domain_links}\n\n"
            "Explore leaderboards focused on specific professional areas:\n\n"
            "- [🏥 Medical Domain Leaderboard](https://chenjy16.github.io/modelrank_ai/medical_leaderboard.html)\n"
            "- [⚖️ Legal Domain Leaderboard](https://chenjy16.github.io/modelrank_ai/legal_leaderboard.html)\n"
            "- [💰 Finance Domain Leaderboard](https://chenjy16.github.io/modelrank_ai/finance_leaderboard.html)\n"
        )

        start_idx_domain_links = content.find(start_marker_domain_links)
        if start_idx_domain_links == -1:
             logger.info("Domain-Specific Links section not found, adding it.")
             # Try inserting before "Data Source" or "License", otherwise append
             insert_before = ["## Data Source", "## License"]
             insert_pos = -1
             for marker in insert_before:
                 pos = content.find(marker)
                 if pos != -1:
                     insert_pos = pos
                     break
             if insert_pos != -1:
                 content = content[:insert_pos] + f"\n{domain_links_content}\n" + content[insert_pos:]
             else:
                 content += f"\n{domain_links_content}\n"
        # No replacement needed for this section, just ensure it exists

        # --- Ensure Other Sections (Data Source, License, etc.) ---
        sections_to_ensure = {
            "Data Source": "Data is sourced from the [HuggingFace Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard).",
            "License": "This project is open-sourced under the [MIT License](LICENSE)."
        }
        for title, text in sections_to_ensure.items():
            section_marker = f"## {title}"
            if section_marker not in content:
                logger.info(f"'{title}' section not found, appending.")
                content += f"\n{section_marker}\n\n{text}\n"

        # Write back updated content to README
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"README update attempted, check {readme_path} for results.")

    except Exception as e:
        logger.error(f"Failed to update README.md: {str(e)}", exc_info=True)
        # README update failure might not be critical, proceed to domain updates

    # --- Domain-Specific Leaderboards Processing ---
    overall_success = True # Track success across all domains
    for domain in ["medical", "legal", "finance"]:
        domain_success_flag = False # Track success for this specific domain iteration
        try:
            logger.info(f"--- Processing {domain.capitalize()} Domain ---")
            # Fetch domain data
            domain_df = await fetch_domain_leaderboard_data(domain, df) # Pass main df for efficiency

            if domain_df is not None and not domain_df.empty:
                # Generate domain Markdown table (top 10 for README)
                domain_table_md = await generate_domain_markdown_table(domain_df, domain, limit=10)

                # Update README with domain table
                readme_update_ok = await update_readme_with_domain(readme_path, domain, domain_table_md, now)
                if not readme_update_ok:
                     logger.warning(f"Failed to update README for {domain} domain section.")
                     # Continue processing other parts for this domain

                # Generate domain HTML page
                domain_html = generate_domain_html_page(domain_df, domain, now)

                # Save domain HTML page
                domain_html_path = docs_dir / f"{domain}_leaderboard.html"
                try:
                    with open(domain_html_path, "w", encoding="utf-8") as f:
                        f.write(domain_html)
                    logger.info(f"Domain HTML page saved: {domain_html_path}")
                except Exception as e:
                    logger.error(f"Failed to save HTML page for {domain}: {e}", exc_info=True)
                    overall_success = False # Consider HTML saving important
                    continue # Skip saving data files if HTML fails? Or try anyway? Let's try saving data.

                # Save domain data files (JSON, CSV)
                data_save_ok = save_domain_data_files(domain_df, domain, docs_dir)
                if not data_save_ok:
                    logger.warning(f"Failed to save data files for {domain} domain.")
                    overall_success = False # Data saving failure is also important
                else:
                    domain_success_flag = True # Mark domain as successfully processed if HTML and data saved

            else:
                logger.warning(f"No data retrieved or generated for {domain} domain. Skipping updates for this domain.")
                # Consider if this should mark overall success as False? Depends on requirements.
                # Let's assume skipping an empty domain is not a failure of the overall process.

        except Exception as e:
            logger.error(f"❌ Unexpected error processing {domain} domain: {str(e)}", exc_info=True)
            overall_success = False # Any exception during domain processing marks failure

        if not domain_success_flag and (domain_df is not None and not domain_df.empty):
             overall_success = False # If we had data but failed saving HTML/data, mark overall failure.

    logger.info(f"--- Update process finished. Overall Success: {overall_success} ---")
    return overall_success

async def fetch_domain_leaderboard_data(domain, all_models_df):
    """获取特定领域的模型评估数据 (using pre-fetched main df)"""
    logger.info(f"Calculating {domain} domain leaderboard data...")

    if all_models_df is None or all_models_df.empty:
        logger.error("Main model DataFrame is empty, cannot calculate domain scores.")
        return None

    try:
        # Base columns needed
        base_columns = ["fullname", "Model", "#Params (B)"]
        if not all(col in all_models_df.columns for col in base_columns):
             missing = [col for col in base_columns if col not in all_models_df.columns]
             logger.error(f"Missing essential base columns: {missing}. Cannot calculate domain scores.")
             return None

        # Create base DataFrame
        models_info = all_models_df[base_columns].copy()

        # Define domain metrics and weights (Ensure these column names exist in all_models_df)
        # Using a simple weighted average approach based on potentially relevant benchmarks
        domain_metrics = {
            "medical": {"MMLU-PRO": 0.5, "BBH": 0.3, "GPQA": 0.2}, # Example weights
            "legal": {"MMLU-PRO": 0.6, "BBH": 0.4}, # Example weights
            "finance": {"MATH Lvl 5": 0.5, "MMLU-PRO": 0.3, "BBH": 0.2} # Example weights
        }

        if domain not in domain_metrics:
            logger.error(f"Unknown domain specified: {domain}")
            return None

        metrics_weights = domain_metrics[domain]
        domain_score = pd.Series(0.0, index=models_info.index) # Initialize with float
        total_weight = 0.0

        # Calculate weighted score
        used_metrics = []
        for metric, weight in metrics_weights.items():
            if metric in all_models_df.columns:
                # Convert metric column to numeric, coerce errors to NaN
                metric_values = pd.to_numeric(all_models_df[metric], errors='coerce')
                # Fill NaN with 0 *before* multiplying by weight
                domain_score += metric_values.fillna(0) * weight
                total_weight += weight
                models_info[metric] = metric_values # Add the (potentially NaNed) metric column
                used_metrics.append(metric)
                logger.info(f"Using metric '{metric}' for {domain} domain with weight {weight}.")
            else:
                logger.warning(f"Metric '{metric}' defined for {domain} not found in main leaderboard data.")

        if total_weight == 0:
            logger.warning(f"No valid metrics found or used for {domain} domain calculation. Cannot compute average.")
            return None # Cannot divide by zero

        # Calculate final average score, handle potential division by zero
        models_info["domain_average"] = domain_score / total_weight

        # Filter out models where the average could not be calculated (e.g., all metrics were NaN)
        # Also filter where domain_average is 0 if that implies no valid scores were found.
        # Let's keep 0 scores for now, but filter NaN.
        models_info = models_info.dropna(subset=["domain_average"])

        # Sort by the calculated domain average score
        models_info = models_info.sort_values("domain_average", ascending=False)

        # Select final columns for the domain DataFrame
        final_domain_cols = base_columns + ["domain_average"] + used_metrics
        models_info = models_info[final_domain_cols]


        if not models_info.empty:
            logger.info(f"✅ Successfully calculated {domain} domain leaderboard data, total: {len(models_info)} models")
            return models_info
        else:
            logger.warning(f"No models with valid scores found for {domain} domain after calculation.")
            return None

    except Exception as e:
        logger.error(f"❌ Failed to calculate {domain} domain leaderboard data: {str(e)}", exc_info=True)
        return None


async def generate_domain_markdown_table(df, domain, limit=10): # Reduced limit for README
    """Generate professional domain leaderboard Markdown table"""
    if df is None or df.empty:
        return f"No data available for the {domain} domain."

    # Identify dataset/metric columns used for this domain
    # These are columns other than the standard ones and domain_average
    standard_cols = ["fullname", "Model", "#Params (B)", "domain_average", "Rank"] # Include Rank if added before call
    dataset_columns = [col for col in df.columns if col not in standard_cols]

    # Columns to display in the table
    columns_to_display = ["Model", "domain_average", "#Params (B)"] + dataset_columns
    # Ensure selected columns exist
    columns = [col for col in columns_to_display if col in df.columns]
    if "Model" not in columns:
         return f"Error: 'Model' column missing in {domain} data."

    # Create display DataFrame
    display_df = df[columns].copy()

    # Format model name (handle missing fullname)
    if 'fullname' in df.columns:
        display_df["Model"] = df.apply(format_model_name, axis=1)
    else:
        display_df["Model"] = df["Model"]


    # Rename columns for display
    column_rename = {
        "Model": "Model",
        "domain_average": "Avg Score", # Shorter
        "#Params (B)": "Params(B)",
        # Add specific dataset display names if needed, otherwise keep original
        # Example: "mmlu_professional_law": "MMLU-ProfLaw"
    }
    # Apply renaming only for existing columns
    display_df = display_df.rename(columns={k: v for k, v in column_rename.items() if k in display_df.columns})


    # Limit rows
    if limit and len(display_df) > limit:
        top_models = display_df.head(limit)
    else:
        top_models = display_df

    # Get final column names after renaming
    final_columns = top_models.columns.tolist()

    # Generate Markdown table
    markdown_table = "| Rank | " + " | ".join(final_columns) + " |\n"
    markdown_table += "| --- | " + " | ".join(["---"] * len(final_columns)) + " |\n"

    for i, (_, row) in enumerate(top_models.iterrows(), 1):
        formatted_row = []
        for col in final_columns: # Use final column names
            value = row[col]
            if pd.isna(value):
                formatted_row.append("-")
             # Use renamed column names for formatting logic
            elif isinstance(value, (int, float)) and col not in ["Params(B)", "Model"]: # Check type and name
                formatted_row.append(f"{value:.2f}")
            elif col == "Params(B)" and isinstance(value, (int, float)):
                 formatted_row.append(f"{value:.1f}" if value % 1 != 0 else f"{int(value)}")
            else:
                formatted_row.append(str(value))
        markdown_table += f"| {i} | " + " | ".join(formatted_row) + " |\n"

    # Add link to the full domain page
    markdown_table += f"\n[View Full {domain.capitalize()} Leaderboard](https://chenjy16.github.io/modelrank_ai/{domain}_leaderboard.html)"

    return markdown_table


def generate_domain_html_page(df, domain, update_time):
    """Generate professional domain leaderboard HTML page"""
    if df is None or df.empty:
        return f"<p>No data available for {domain} domain</p>"

    # Create HTML version DataFrame
    html_df = df.copy()

    # Add Rank column if it doesn't exist
    if 'Rank' not in html_df.columns:
        html_df.insert(0, 'Rank', range(1, len(html_df) + 1))

    # Format model names with HTML links
    if 'fullname' in html_df.columns:
        html_df["Model"] = html_df.apply(format_model_name_html, axis=1)
    else:
        html_df["Model"] = html_df["Model"].astype(str)


    # Identify dataset/metric columns
    standard_cols = ["fullname", "Model", "#Params (B)", "domain_average", "Rank"]
    dataset_columns = [col for col in df.columns if col not in standard_cols]

    # Select and order columns for display
    display_columns_order = ['Rank', 'Model', 'domain_average', '#Params (B)'] + dataset_columns
    display_columns = [col for col in display_columns_order if col in html_df.columns]

    if not display_columns or 'Model' not in display_columns:
        logger.error(f"Essential columns missing for {domain} HTML page.")
        return f"<p>Error displaying {domain} data.</p>"

    display_df = html_df[display_columns].copy()

    # Rename columns
    column_rename = {
        "Rank": "Rank",
        "Model": "Model",
        "domain_average": "Average Score",
        "#Params (B)": "Parameters(B)"
        # Add specific renames for dataset columns if desired
        # e.g., "mmlu_professional_law": "MMLU Prof Law"
    }
    display_df = display_df.rename(columns={k: v for k, v in column_rename.items() if k in display_df.columns})


    # Format numeric columns
    final_columns = display_df.columns.tolist()
    for col in final_columns:
        if col not in ['Rank', 'Model']:
             try:
                  display_df[col] = display_df[col].apply(
                     lambda x: f"{float(x):.2f}" if pd.notna(x) and isinstance(x, (int, float)) and col != "Parameters(B)" else
                               (f"{float(x):.1f}" if float(x) % 1 != 0 else f"{int(float(x))}") if pd.notna(x) and col == "Parameters(B)" and isinstance(x, (int, float)) else
                               x if pd.notna(x) else "-"
                  )
             except (ValueError, TypeError):
                  logger.warning(f"Could not format column '{col}' as numeric in {domain} HTML. Keeping original.")
                  display_df[col] = display_df[col].astype(str)


    # Generate HTML table
    html_table = display_df.to_html(
        index=False,
        escape=False,
        classes="table table-striped table-hover table-bordered",
        table_id="leaderboard",
        na_rep="-"
    )

    # Domain display name
    domain_display_names = {
        "medical": "Medical",
        "legal": "Legal",
        "finance": "Finance"
    }
    domain_display = domain_display_names.get(domain, domain.capitalize())
    domain_icon = {"medical": "🏥", "legal": "⚖️", "finance": "💰"}.get(domain, "🏆")


    # Other leaderboards links (conditional)
    other_links = ""
    if domain != "medical":
        other_links += '<a href="medical_leaderboard.html" class="btn btn-outline-primary me-2 mb-2">🏥 Medical</a>\n'
    if domain != "legal":
        other_links += '<a href="legal_leaderboard.html" class="btn btn-outline-primary me-2 mb-2">⚖️ Legal</a>\n'
    if domain != "finance":
        other_links += '<a href="finance_leaderboard.html" class="btn btn-outline-primary mb-2">💰 Finance</a>\n'


    # Create full HTML page
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ModelRank AI - {domain_display} Domain Leaderboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.datatables.net/1.13.4/css/dataTables.bootstrap5.min.css">
    <style>
        body {{ padding: 20px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }}
        .container {{ max-width: 1400px; }}
        h1 {{ margin-bottom: 20px; color: #333; }}
        .table {{ width: 100%; }}
        .table th {{ position: sticky; top: 0; background-color: #f8f9fa; color: #333; font-weight: 600; white-space: nowrap; }}
        .table td {{ vertical-align: middle; white-space: nowrap; }}
        .footer {{ margin-top: 30px; padding-top: 10px; border-top: 1px solid #eee; color: #666; font-size: 0.9rem; }}
        .card {{ margin-bottom: 20px; }}
        .badge {{ font-size: 0.8rem; }}
        .table-responsive {{ overflow-x: auto; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="row mb-4 align-items-center">
            <div class="col-md-8">
                <h1>{domain_icon} ModelRank AI - {domain_display} Domain Leaderboard</h1>
                <p class="text-muted mb-0">Last updated: {update_time}</p>
            </div>
            <div class="col-md-4 text-end">
                <a href="index.html" class="btn btn-outline-secondary me-2">Back to Main Leaderboard</a>
                 <a href="https://github.com/chenjy16/modelrank_ai" class="btn btn-outline-dark" target="_blank">
                     <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-github" viewBox="0 0 16 16">
                         <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
                     </svg> GitHub
                 </a>
            </div>
        </div>

        <div class="card">
            <div class="card-header bg-light">
                <div class="row">
                    <div class="col">
                        <h5 class="mb-0">{domain_display} Domain Leaderboard</h5>
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
                    <div class="card-header bg-light"><h5 class="mb-0">Data Download</h5></div>
                    <div class="card-body">
                        <p>Download {domain_display} leaderboard data:</p>
                        <a href="{domain}_leaderboard.json" class="btn btn-outline-secondary me-2" download>JSON</a>
                        <a href="{domain}_leaderboard.csv" class="btn btn-outline-secondary" download>CSV</a>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header bg-light"><h5 class="mb-0">Other Leaderboards</h5></div>
                    <div class="card-body">
                        <p>View other leaderboards:</p>
                        <a href="index.html" class="btn btn-outline-secondary me-2 mb-2">Main Leaderboard</a>
                        {other_links}
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