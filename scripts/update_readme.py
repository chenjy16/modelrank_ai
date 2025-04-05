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
        # If README doesn't exist, create a new one
        content = "# ModelRank AI\n\nThis is an automatically updated open-source large language model leaderboard with data sourced from HuggingFace.\n\n## Project Description\n\nThis project automatically fetches the latest model evaluation data from HuggingFace daily via GitHub Actions and updates this README.\n\n"
    else:
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
    
    # Update timestamp
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    # Check if README already has a leaderboard section
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
    
    # Add data source explanation (if it doesn't exist)
    if "## Data Source" not in content:
        content += "\n## Data Source\n\nData is sourced from HuggingFace.\n\n"
    
    # Add license information (if it doesn't exist)
    if "## License" not in content:
        content += "\n## License\n\nThis project is open-sourced under the MIT License.\n"
    
    # Write back to README
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    logger.info(f"README updated successfully, time: {now}")
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
