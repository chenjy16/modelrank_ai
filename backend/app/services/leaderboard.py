from app.core.cache import cache_config
from datetime import datetime
from typing import List, Dict, Any
import datasets
from fastapi import HTTPException
import logging
from app.config.base import HF_ORGANIZATION
from app.core.formatting import LogFormatter

logger = logging.getLogger(__name__)

class LeaderboardService:
    def __init__(self):
        pass
        
    async def fetch_raw_data(self) -> List[Dict[str, Any]]:
        """Fetch raw leaderboard data from HuggingFace dataset"""
        try:
            logger.info(LogFormatter.section("FETCHING LEADERBOARD DATA"))
            logger.info(LogFormatter.info(f"Loading dataset from {HF_ORGANIZATION}/contents"))
            
            try:
                dataset = datasets.load_dataset(
                    f"{HF_ORGANIZATION}/contents",
                    cache_dir=cache_config.get_cache_path("datasets")
                )["train"]
            except Exception as e:
                logger.error(LogFormatter.error(f"Failed to load dataset: {e}", e))
                # 在脚本环境中返回空列表而不是抛出HTTP异常
                if 'fastapi' not in sys.modules:
                    return []
                raise HTTPException(status_code=500, detail=str(e))
            
            df = dataset.to_pandas()
            data = df.to_dict('records')
            
            stats = {
                "Total_Entries": len(data),
                "Dataset_Size": f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f}MB"
            }
            for line in LogFormatter.stats(stats, "Dataset Statistics"):
                logger.info(line)
                
            return data
            
        except Exception as e:
            logger.error(LogFormatter.error("Failed to fetch leaderboard data", e))
            # 在脚本环境中返回空列表而不是抛出HTTP异常
            if 'fastapi' not in sys.modules:
                return []
            raise HTTPException(status_code=500, detail=str(e))

    async def get_formatted_data(self) -> List[Dict[str, Any]]:
        """Get formatted leaderboard data"""
        try:
            logger.info(LogFormatter.section("FORMATTING LEADERBOARD DATA"))
            
            raw_data = await self.fetch_raw_data()
            formatted_data = []
            type_counts = {}
            error_count = 0
            
            # Initialize progress tracking
            total_items = len(raw_data)
            logger.info(LogFormatter.info(f"Processing {total_items:,} entries..."))
            
            for i, item in enumerate(raw_data, 1):
                try:
                    formatted_item = await self.transform_data(item)
                    formatted_data.append(formatted_item)
                    
                    # Count model types
                    model_type = formatted_item["model"]["type"]
                    type_counts[model_type] = type_counts.get(model_type, 0) + 1
                    
                except Exception as e:
                    error_count += 1
                    logger.error(LogFormatter.error(f"Failed to format entry {i}/{total_items}", e))
                    continue
                
                # Log progress every 10%
                if i % max(1, total_items // 10) == 0:
                    progress = (i / total_items) * 100
                    logger.info(LogFormatter.info(f"Progress: {LogFormatter.progress_bar(i, total_items)}"))
            
            # Log final statistics
            stats = {
                "Total_Processed": total_items,
                "Successful": len(formatted_data),
                "Failed": error_count
            }
            logger.info(LogFormatter.section("PROCESSING SUMMARY"))
            for line in LogFormatter.stats(stats, "Processing Statistics"):
                logger.info(line)
            
            # Log model type distribution
            type_stats = {f"Type_{k}": v for k, v in type_counts.items()}
            logger.info(LogFormatter.subsection("MODEL TYPE DISTRIBUTION"))
            for line in LogFormatter.stats(type_stats):
                logger.info(line)
                
            return formatted_data
            
        except Exception as e:
            logger.error(LogFormatter.error("Failed to format leaderboard data", e))
            raise HTTPException(status_code=500, detail=str(e))

    async def transform_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform raw data into the format expected by the frontend"""
        try:
            # Extract model name for logging
            model_name = data.get("fullname", "Unknown")
            logger.debug(LogFormatter.info(f"Transforming data for model: {model_name}"))
            
            # Create unique ID combining model name, precision, sha and chat template status
            unique_id = f"{data.get('fullname', 'Unknown')}_{data.get('Precision', 'Unknown')}_{data.get('Model sha', 'Unknown')}_{str(data.get('Chat Template', False))}"
            
            evaluations = {
                "ifeval": {
                    "name": "IFEval",
                    "value": data.get("IFEval Raw", 0),
                    "normalized_score": data.get("IFEval", 0)
                },
                "bbh": {
                    "name": "BBH",
                    "value": data.get("BBH Raw", 0),
                    "normalized_score": data.get("BBH", 0)
                },
                "math": {
                    "name": "MATH Level 5",
                    "value": data.get("MATH Lvl 5 Raw", 0),
                    "normalized_score": data.get("MATH Lvl 5", 0)
                },
                "gpqa": {
                    "name": "GPQA",
                    "value": data.get("GPQA Raw", 0),
                    "normalized_score": data.get("GPQA", 0)
                },
                "musr": {
                    "name": "MUSR",
                    "value": data.get("MUSR Raw", 0),
                    "normalized_score": data.get("MUSR", 0)
                },
                "mmlu_pro": {
                    "name": "MMLU-PRO",
                    "value": data.get("MMLU-PRO Raw", 0),
                    "normalized_score": data.get("MMLU-PRO", 0)
                }
            }

            features = {
                "is_not_available_on_hub": data.get("Available on the hub", False),
                "is_merged": data.get("Merged", False),
                "is_moe": data.get("MoE", False),
                "is_flagged": data.get("Flagged", False),
                "is_official_provider": data.get("Official Providers", False)
            }

            metadata = {
                "upload_date": data.get("Upload To Hub Date"),
                "submission_date": data.get("Submission Date"),
                "generation": data.get("Generation"),
                "base_model": data.get("Base Model"),
                "hub_license": data.get("Hub License"),
                "hub_hearts": data.get("Hub ❤️"),
                "params_billions": data.get("#Params (B)"),
                "co2_cost": data.get("CO₂ cost (kg)", 0)
            }

            # Clean model type by removing emojis if present
            original_type = data.get("Type", "")
            model_type = original_type.lower().strip()
            
            # Remove emojis and parentheses
            if "(" in model_type:
                model_type = model_type.split("(")[0].strip()
            model_type = ''.join(c for c in model_type if not c in '🔶🟢🟩💬🤝🌸 ')
                
            # Map old model types to new ones
            model_type_mapping = {
                "fine-tuned": "fined-tuned-on-domain-specific-dataset",
                "fine tuned": "fined-tuned-on-domain-specific-dataset",
                "finetuned": "fined-tuned-on-domain-specific-dataset",
                "fine_tuned": "fined-tuned-on-domain-specific-dataset",
                "ft": "fined-tuned-on-domain-specific-dataset",
                "finetuning": "fined-tuned-on-domain-specific-dataset",
                "fine tuning": "fined-tuned-on-domain-specific-dataset",
                "fine-tuning": "fined-tuned-on-domain-specific-dataset"
            }

            mapped_type = model_type_mapping.get(model_type.lower().strip(), model_type)
            
            if mapped_type != model_type:
                logger.debug(LogFormatter.info(f"Model type mapped: {original_type} -> {mapped_type}"))
            
            transformed_data = {
                "id": unique_id,
                "model": {
                    "name": data.get("fullname"),
                    "sha": data.get("Model sha"),
                    "precision": data.get("Precision"),
                    "type": mapped_type,
                    "weight_type": data.get("Weight type"),
                    "architecture": data.get("Architecture"),
                    "average_score": data.get("Average ⬆️"),
                    "has_chat_template": data.get("Chat Template", False)
                },
                "evaluations": evaluations,
                "features": features,
                "metadata": metadata
            }
            
            logger.debug(LogFormatter.success(f"Successfully transformed data for {model_name}"))
            return transformed_data
            
        except Exception as e:
            logger.error(LogFormatter.error(f"Failed to transform data for {data.get('fullname', 'Unknown')}", e))
            raise

    # 在现有的LeaderboardService类中添加或修改get_formatted_leaderboard方法
    # 确保它可以在没有HTTP请求上下文的情况下工作
    
    async def get_formatted_leaderboard(self) -> List[Dict[str, Any]]:
        """获取格式化的排行榜数据，适用于脚本环境"""
        try:
            # 获取原始数据
            raw_data = await self.fetch_raw_data()
            
            # 处理数据
            formatted_data = []
            for item in raw_data:
                # 提取模型信息
                model_id = item.get('eval_name', '')
                parts = model_id.split('/')
                provider = parts[0] if len(parts) > 0 else ''
                model_name = parts[1] if len(parts) > 1 else model_id
                
                # 计算平均分
                benchmarks = ['mmlu', 'arc_challenge', 'hellaswag', 'truthfulqa_mc', 'winogrande', 'gsm8k']
                scores = [item.get(b, 0) for b in benchmarks if b in item and item[b] is not None]
                average = sum(scores) / len(scores) if scores else 0
                
                # 构建格式化数据
                formatted_item = {
                    'model_id': model_id,
                    'model_name': model_name,
                    'provider': provider,
                    'average': average,
                    'params_billions': item.get('params_billions', 0),
                    'model_type': item.get('Type', 'Unknown')
                }
                
                # 添加各项基准测试分数
                for benchmark in benchmarks:
                    if benchmark in item and item[benchmark] is not None:
                        formatted_item[benchmark] = item[benchmark]
                
                formatted_data.append(formatted_item)
            
            # 按平均分排序
            formatted_data.sort(key=lambda x: x['average'], reverse=True)
            
            return formatted_data
            
        except Exception as e:
            logger.error(f"获取格式化排行榜数据失败: {str(e)}")
            return []