import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Union
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bs4 import BeautifulSoup
import logging
from dotenv import load_dotenv
import sys
import codecs
import os
from jsonformer import Jsonformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('jsonformer_agent.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class JsonformerPricingAgent:
    """Jsonformer-powered agent for parsing pricing pages into structured schema"""
    
    def __init__(self, model_name="mistralai/Mixtral-8x7B-v0.1"):
        """Initialize the model and tokenizer"""
        self.start_time = None
        try:
            logger.info(f"🚀 Initializing JsonformerPricingAgent with model: {model_name}")
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            logger.info("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_8bit=True
            )
            
            device_info = next(self.model.parameters()).device
            logger.info(f"✅ Model loaded successfully on device: {device_info}")
            logger.info(f"Model size: {sum(p.numel() for p in self.model.parameters())/1e6:.2f}M parameters")
            
            # Load the JSON schema
            self.schema = self._get_schema_template()
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}", exc_info=True)
            raise

    def _get_schema_template(self) -> Dict:
        """Get the JSON schema for pricing data"""
        return {
            "type": "object",
            "properties": {
                "service": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "url": {"type": "string"},
                        "logo_url": {"type": "string"},
                        "description": {"type": "string"},
                        "category": {
                            "type": "object",
                            "properties": {
                                "primary": {"type": "string"},
                                "secondary": {"type": "string"},
                                "tags": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            }
                        },
                        "business_model": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "description": {"type": "string"}
                            }
                        }
                    }
                },
                "plans": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "slug": {"type": "string"},
                            "description": {"type": "string"},
                            "highlight_features": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "is_popular": {"type": "boolean"},
                            "pricing": {
                                "type": "object",
                                "properties": {
                                    "base": {
                                        "type": "object",
                                        "properties": {
                                            "amount": {"type": "number"},
                                            "period": {"type": "string"},
                                            "currency": {"type": "string"},
                                            "is_per_user": {"type": "boolean"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

    def parse_content(self, raw_content: str, url: str) -> Dict:
        """Parse raw content into structured schema using Jsonformer"""
        self.start_time = time.time()
        
        logger.info(f"Starting content parsing for URL: {url}")
        logger.info(f"Raw content length: {len(raw_content)} characters")
        
        try:
            # Clean content
            cleaned_content = self._clean_content(raw_content)
            logger.info(f"Cleaned content length: {len(cleaned_content)} characters")
            
            # Create prompt
            prompt = f"""Extract pricing information from this webpage content into structured JSON format.
            Follow these rules:
            1. Extract all pricing plans and their features
            2. Convert prices to numbers (e.g., "$20" becomes 20)
            3. Create URL-friendly slugs for plan names
            4. Mark the most popular/featured plan
            5. Include all feature lists
            6. Extract numerical limits
            
            Content to analyze:
            {cleaned_content}
            """
            
            # Create Jsonformer instance
            jsonformer = Jsonformer(
                model=self.model,
                tokenizer=self.tokenizer,
                json_schema=self.schema,
                prompt=prompt
            )
            
            # Generate structured JSON
            parsed_data = jsonformer()
            
            # Add metadata
            parsed_data = self._add_metadata(parsed_data, url)
            
            total_time = time.time() - self.start_time
            logger.info(f"✅ Successfully parsed content in {total_time:.2f}s")
            return parsed_data
            
        except Exception as e:
            logger.error(f"❌ Process failed: {str(e)}", exc_info=True)
            raise

    def _clean_content(self, content: str) -> str:
        """Clean and prepare content for parsing"""
        try:
            soup = BeautifulSoup(content, 'html.parser')
            for element in soup(['script', 'style', 'meta', 'link']):
                element.decompose()
            text = soup.get_text(separator=' ', strip=True)
            return ' '.join(text.split())
        except Exception as e:
            logger.warning(f"Error cleaning content: {str(e)}")
            return content

    def _add_metadata(self, parsed_data: Dict, url: str) -> Dict:
        """Add metadata to the parsed data"""
        execution_time = time.time() - self.start_time
        
        parsed_data["agent_metadata"] = {
            "agent_name": "jsonformer-mixtral-8x7b",
            "agent_version": "1.0",
            "execution_time_seconds": round(execution_time, 2),
            "fallback_used": False,
            "comments": ""
        }
        
        if "service" not in parsed_data:
            parsed_data["service"] = {}
        parsed_data["service"]["url"] = url
        
        return parsed_data

def main():
    """Example usage"""
    logger.info("🚀 Starting JsonformerPricingAgent example")
    
    try:
        agent = JsonformerPricingAgent()
        
        content_path = Path("raw_content_storage/raw_content_airtable_20241217_021032.txt")
        if content_path.exists():
            logger.info(f"Reading content from: {content_path}")
            with open(content_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info("Processing content...")
            parsed_data = agent.parse_content(
                content,
                url="https://airtable.com/pricing"
            )
            
            output_path = Path("parsed_content_storage/parsed_jsonformer_airtable_20241217_021126.json")
            output_path.parent.mkdir(exist_ok=True)
            
            logger.info(f"Saving results to: {output_path}")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(parsed_data, f, indent=2)
                
            logger.info("✅ Process completed successfully")
            
    except Exception as e:
        logger.error(f"❌ Process failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 