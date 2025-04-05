from datetime import datetime, timezone
from typing import Dict, Any, List, Set, Tuple, Optional
import json
import logging
import asyncio
from pathlib import Path
import aiohttp
from huggingface_hub import HfApi
import tempfile
import os

from app.services.hf_service import HuggingFaceService
from app.config import HF_TOKEN
from app.config.hf_config import HF_ORGANIZATION
from app.core.cache import cache_config
from app.core.formatting import LogFormatter

logger = logging.getLogger(__name__)

class VoteService(HuggingFaceService):
    _instance: Optional['VoteService'] = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VoteService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_init_done'):
            super().__init__()
            self.votes_file = cache_config.votes_file
            self.votes_to_upload: List[Dict[str, Any]] = []
            self.vote_check_set: Set[Tuple[str, str, str, str]] = set()
            self._votes_by_model: Dict[str, List[Dict[str, Any]]] = {}
            self._votes_by_user: Dict[str, List[Dict[str, Any]]] = {}
            self._last_sync = None
            self._sync_interval = 300  # 5 minutes
            self._total_votes = 0
            self._last_vote_timestamp = None
            self._max_retries = 3
            self._retry_delay = 1  # seconds
            self.hf_api = HfApi(token=HF_TOKEN)
            self._init_done = True

    async def initialize(self):
        """Initialize the vote service"""
        if self._initialized:
            await self._check_for_new_votes()
            return
        
        try:
            logger.info(LogFormatter.section("VOTE SERVICE INITIALIZATION"))
            
            # Ensure votes directory exists
            self.votes_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Load remote votes
            remote_votes = await self._fetch_remote_votes()
            if remote_votes:
                logger.info(LogFormatter.info(f"Loaded {len(remote_votes)} votes from hub"))
                
                # Save to local file
                with open(self.votes_file, 'w') as f:
                    for vote in remote_votes:
                        json.dump(vote, f)
                        f.write('\n')
                
                # Load into memory
                await self._load_existing_votes()
            else:
                logger.warning(LogFormatter.warning("No votes found on hub"))
            
            self._initialized = True
            self._last_sync = datetime.now(timezone.utc)
            
            # Final summary
            stats = {
                "Total_Votes": self._total_votes,
                "Last_Sync": self._last_sync.strftime("%Y-%m-%d %H:%M:%S UTC")
            }
            logger.info(LogFormatter.section("INITIALIZATION COMPLETE"))
            for line in LogFormatter.stats(stats):
                logger.info(line)
            
        except Exception as e:
            logger.error(LogFormatter.error("Initialization failed", e))
            raise

    async def _fetch_remote_votes(self) -> List[Dict[str, Any]]:
        """Fetch votes from HF hub"""
        url = f"https://huggingface.co/datasets/{HF_ORGANIZATION}/votes/raw/main/votes_data.jsonl"
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        votes = []
                        async for line in response.content:
                            if line.strip():
                                try:
                                    vote = json.loads(line.decode())
                                    votes.append(vote)
                                except json.JSONDecodeError:
                                    continue
                        return votes
                    else:
                        logger.error(f"Failed to get remote votes: HTTP {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error fetching remote votes: {str(e)}")
            return []

    async def _check_for_new_votes(self):
        """Check for new votes on the hub and sync if needed"""
        try:
            remote_votes = await self._fetch_remote_votes()
            if len(remote_votes) != self._total_votes:
                logger.info(f"Vote count changed: Local ({self._total_votes}) ≠ Remote ({len(remote_votes)})")
                # Save to local file
                with open(self.votes_file, 'w') as f:
                    for vote in remote_votes:
                        json.dump(vote, f)
                        f.write('\n')
                
                # Reload into memory
                await self._load_existing_votes()
            else:
                logger.info("Votes are in sync")
                
        except Exception as e:
            logger.error(f"Error checking for new votes: {str(e)}")

    async def _sync_with_hub(self):
        """Sync votes with HuggingFace hub"""
        try:
            logger.info(LogFormatter.section("VOTE SYNC"))
            
            # Get current remote votes
            remote_votes = await self._fetch_remote_votes()
            logger.info(LogFormatter.info(f"Loaded {len(remote_votes)} votes from hub"))
            
            # If we have pending votes to upload
            if self.votes_to_upload:
                logger.info(LogFormatter.info(f"Adding {len(self.votes_to_upload)} pending votes..."))
                
                # Add new votes to remote votes
                remote_votes.extend(self.votes_to_upload)
                
                # Create temporary file with all votes
                with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as temp_file:
                    for vote in remote_votes:
                        json.dump(vote, temp_file)
                        temp_file.write('\n')
                    temp_path = temp_file.name
                
                try:
                    # Upload JSONL file directly
                    self.hf_api.upload_file(
                        path_or_fileobj=temp_path,
                        path_in_repo="votes_data.jsonl",
                        repo_id=f"{HF_ORGANIZATION}/votes",
                        repo_type="dataset",
                        commit_message=f"Update votes: +{len(self.votes_to_upload)} new votes",
                        token=self.token
                    )
                    
                    # Clear pending votes only if upload succeeded
                    self.votes_to_upload.clear()
                    logger.info(LogFormatter.success("Pending votes uploaded successfully"))
                    
                except Exception as e:
                    logger.error(LogFormatter.error("Failed to upload votes to hub", e))
                    raise
                finally:
                    # Clean up temp file
                    os.unlink(temp_path)
            
            # Update local state
            with open(self.votes_file, 'w') as f:
                for vote in remote_votes:
                    json.dump(vote, f)
                    f.write('\n')
            
            # Reload votes in memory
            await self._load_existing_votes()
            logger.info(LogFormatter.success("Sync completed successfully"))

            self._last_sync = datetime.now(timezone.utc)
            
        except Exception as e:
            logger.error(LogFormatter.error("Sync failed", e))
            raise

    async def _load_existing_votes(self):
        """Load existing votes from file"""
        if not self.votes_file.exists():
            logger.warning(LogFormatter.warning("No votes file found"))
            return

        try:
            logger.info(LogFormatter.section("LOADING VOTES"))
            
            # Clear existing data structures
            self.vote_check_set.clear()
            self._votes_by_model.clear()
            self._votes_by_user.clear()
            
            vote_count = 0
            latest_timestamp = None
            
            with open(self.votes_file, "r") as f:
                for line in f:
                    try:
                        vote = json.loads(line.strip())
                        vote_count += 1
                        
                        # Track latest timestamp
                        try:
                            vote_timestamp = datetime.fromisoformat(vote["timestamp"].replace("Z", "+00:00"))
                            if not latest_timestamp or vote_timestamp > latest_timestamp:
                                latest_timestamp = vote_timestamp
                            vote["timestamp"] = vote_timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
                        except (KeyError, ValueError) as e:
                            logger.warning(LogFormatter.warning(f"Invalid timestamp in vote: {str(e)}"))
                            continue
                        
                        if vote_count % 1000 == 0:
                            logger.info(LogFormatter.info(f"Processed {vote_count:,} votes..."))
                        
                        self._add_vote_to_memory(vote)
                        
                    except json.JSONDecodeError as e:
                        logger.error(LogFormatter.error("Vote parsing failed", e))
                        continue
                    except Exception as e:
                        logger.error(LogFormatter.error("Vote processing failed", e))
                        continue
            
            self._total_votes = vote_count
            self._last_vote_timestamp = latest_timestamp
            
            # Final summary
            stats = {
                "Total_Votes": vote_count,
                "Latest_Vote": latest_timestamp.strftime("%Y-%m-%d %H:%M:%S UTC") if latest_timestamp else "None",
                "Unique_Models": len(self._votes_by_model),
                "Unique_Users": len(self._votes_by_user)
            }
            
            logger.info(LogFormatter.section("VOTE SUMMARY"))
            for line in LogFormatter.stats(stats):
                logger.info(line)
            
        except Exception as e:
            logger.error(LogFormatter.error("Failed to load votes", e))
            raise

    def _add_vote_to_memory(self, vote: Dict[str, Any]):
        """Add vote to memory structures"""
        try:
            # Create a unique identifier tuple that includes precision
            check_tuple = (
                vote["model"],
                vote.get("revision", "main"),
                vote["username"],
                vote.get("precision", "unknown")
            )
            
            # Skip if we already have this vote
            if check_tuple in self.vote_check_set:
                return
                
            self.vote_check_set.add(check_tuple)
            
            # Update model votes
            if vote["model"] not in self._votes_by_model:
                self._votes_by_model[vote["model"]] = []
            self._votes_by_model[vote["model"]].append(vote)
            
            # Update user votes
            if vote["username"] not in self._votes_by_user:
                self._votes_by_user[vote["username"]] = []
            self._votes_by_user[vote["username"]].append(vote)
            
        except KeyError as e:
            logger.error(LogFormatter.error("Malformed vote data, missing key", str(e)))
        except Exception as e:
            logger.error(LogFormatter.error("Error adding vote to memory", str(e)))

    async def get_user_votes(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all votes from a specific user"""
        logger.info(LogFormatter.info(f"Fetching votes for user: {user_id}"))
        
        # Check if we need to refresh votes
        if (datetime.now(timezone.utc) - self._last_sync).total_seconds() > self._sync_interval:
            logger.info(LogFormatter.info("Cache expired, refreshing votes..."))
            await self._check_for_new_votes()
            
        votes = self._votes_by_user.get(user_id, [])
        logger.info(LogFormatter.success(f"Found {len(votes):,} votes"))
        return votes

    async def get_model_votes(self, model_id: str) -> Dict[str, Any]:
        """Get all votes for a specific model"""
        logger.info(LogFormatter.info(f"Fetching votes for model: {model_id}"))
        
        # Check if we need to refresh votes
        if (datetime.now(timezone.utc) - self._last_sync).total_seconds() > self._sync_interval:
            logger.info(LogFormatter.info("Cache expired, refreshing votes..."))
            await self._check_for_new_votes()
        
        votes = self._votes_by_model.get(model_id, [])
        
        # Group votes by revision and precision
        votes_by_config = {}
        for vote in votes:
            revision = vote.get("revision", "main")
            precision = vote.get("precision", "unknown")
            config_key = f"{revision}_{precision}"
            if config_key not in votes_by_config:
                votes_by_config[config_key] = {
                    "revision": revision,
                    "precision": precision,
                    "count": 0
                }
            votes_by_config[config_key]["count"] += 1
        
        stats = {
            "Total_Votes": len(votes),
            **{f"Config_{k}": v["count"] for k, v in votes_by_config.items()}
        }
        
        logger.info(LogFormatter.section("VOTE STATISTICS"))
        for line in LogFormatter.stats(stats):
            logger.info(line)
        
        return {
            "total_votes": len(votes),
            "votes_by_config": votes_by_config,
            "votes": votes
        }

    async def _get_model_revision(self, model_id: str) -> str:
        """Get current revision of a model with retries"""
        logger.info(f"Getting revision for model: {model_id}")
        for attempt in range(self._max_retries):
            try:
                model_info = await asyncio.to_thread(self.hf_api.model_info, model_id)
                logger.info(f"Successfully got revision {model_info.sha} for model {model_id}")
                return model_info.sha
            except Exception as e:
                logger.error(f"Error getting model revision for {model_id} (attempt {attempt + 1}): {str(e)}")
                if attempt < self._max_retries - 1:
                    retry_delay = self._retry_delay * (attempt + 1)
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.warning(f"Using 'main' as fallback revision for {model_id} after {self._max_retries} failed attempts")
                    return "main"

    async def add_vote(self, model_id: str, user_id: str, vote_type: str, vote_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Add a vote for a model"""
        try:
            self._log_repo_operation("add", f"{HF_ORGANIZATION}/votes", f"Adding {vote_type} vote for {model_id} by {user_id}")
            logger.info(LogFormatter.section("NEW VOTE"))
            stats = {
                "Model": model_id,
                "User": user_id,
                "Type": vote_type,
                "Config": vote_data or {}
            }
            for line in LogFormatter.tree(stats, "Vote Details"):
                logger.info(line)
            
            # Use provided configuration or fallback to model info
            precision = None
            revision = None
            
            if vote_data:
                precision = vote_data.get("precision")
                revision = vote_data.get("revision")
            
            # If any info is missing, try to get it from model info
            if not all([precision, revision]):
                try:
                    model_info = await asyncio.to_thread(self.hf_api.model_info, model_id)
                    model_card_data = model_info.cardData if hasattr(model_info, 'cardData') else {}
                    
                    if not precision:
                        precision = model_card_data.get("precision", "unknown")
                    if not revision:
                        revision = model_info.sha
                except Exception as e:
                    logger.warning(LogFormatter.warning(f"Failed to get model info: {str(e)}. Using default values."))
                    precision = precision or "unknown"
                    revision = revision or "main"
            
            # Check if vote already exists with this configuration
            check_tuple = (model_id, revision, user_id, precision)
            
            if check_tuple in self.vote_check_set:
                raise ValueError(f"Vote already recorded for this model configuration (precision: {precision}, revision: {revision[:7] if revision else 'unknown'})")

            vote = {
                "model": model_id,
                "revision": revision,
                "username": user_id,
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "vote_type": vote_type,
                "precision": precision
            }

            # Update local storage
            with open(self.votes_file, "a") as f:
                f.write(json.dumps(vote) + "\n")
            
            self._add_vote_to_memory(vote)
            self.votes_to_upload.append(vote)
            
            stats = {
                "Status": "Success",
                "Queue_Size": len(self.votes_to_upload),
                "Model_Config": {
                    "Precision": precision,
                    "Revision": revision[:7] if revision else "unknown"
                }
            }
            for line in LogFormatter.stats(stats):
                logger.info(line)
            
            # Force immediate sync
            logger.info(LogFormatter.info("Forcing immediate sync with hub"))
            await self._sync_with_hub()
            
            return {"status": "success", "message": "Vote added successfully"}
            
        except Exception as e:
            logger.error(LogFormatter.error("Failed to add vote", e))
            raise