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

# Force UTF-8 encoding for stdout/stderr
if sys.stdout.encoding != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Configure detailed logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llama_agent.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)  # Use UTF-8 encoded stdout
    ]
)
logger = logging.getLogger(__name__)

class LlamaPricingAgent:
    """Llama-powered agent for parsing pricing pages into structured schema"""
    
    def __init__(self, model_name="meta-llama/Llama-3.2-1B"):
        """Initialize the Llama model and tokenizer"""
        self.start_time = None
        try:
            logger.info(f"🚀 Initializing LlamaPricingAgent with model: {model_name}")
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            logger.info("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            device_info = next(self.model.parameters()).device
            logger.info(f"✅ Model loaded successfully on device: {device_info}")
            logger.info(f"Model size: {sum(p.numel() for p in self.model.parameters())/1e6:.2f}M parameters")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}", exc_info=True)
            raise

    def _create_system_prompt(self) -> str:
        """Create the system prompt for the Llama model"""
        return """You are a specialized pricing data extraction agent. Your task is to analyze pricing page content 
        and extract structured data according to our exact schema. You must follow these rules:

        1. Always return a complete JSON object matching our schema exactly
        2. For missing information, use null instead of omitting fields
        3. Convert all prices to numbers without currency symbols
        4. Detect pricing models accurately (fixed, usage-based, tiered, hybrid)
        5. Extract all feature lists and categorize them appropriately
        6. Include all limits and restrictions
        7. Create URL-friendly slugs for plan names (lowercase, hyphens)
        8. Identify popular/featured plans
        9. Format dates in ISO8601 format
        10. Use proper null values for missing data
        11. Use proper boolean values (true/false)
        12. Ensure all arrays exist even if empty

        Important: Return only valid JSON with properly quoted properties and no comments.
        """

    def _get_schema_template(self) -> Dict:
        """Get the empty schema template"""
        return {
            "service": {
                "name": None,
                "url": None,
                "logo_url": None,
                "description": None,
                "category": {
                    "primary": None,
                    "secondary": None,
                    "tags": []
                },
                "business_model": {
                    "type": [],
                    "description": None
                }
            },
            "pricing_metadata": {
                "last_updated": None,
                "currency": None,
                "regions": [],
                "billing_cycles": {
                    "available": [],
                    "default": None
                },
                "custom_pricing_available": None,
                "free_tier_available": None,
                "versioning": {
                    "current": None,
                    "history": []
                }
            },
            "plans": [
                {
                    "name": None,
                    "slug": None,
                    "description": None,
                    "highlight_features": [],
                    "is_popular": None,
                    "pricing": {
                        "base": {
                            "amount": None,
                            "amount_range": {
                                "min": None,
                                "max": None
                            },
                            "period": None,
                            "currency": None,
                            "is_per_user": None
                        },
                        "usage_based": [
                            {
                                "name": None,
                                "type": None,
                                "unit": None,
                                "tiers": [
                                    {
                                        "range": {
                                            "min": None,
                                            "max": None
                                        },
                                        "unit_price": None,
                                        "flat_fee": None
                                    }
                                ]
                            }
                        ]
                    },
                    "limits": {
                        "users": {
                            "min": None,
                            "max": None,
                            "description": None
                        },
                        "storage": {
                            "amount": None,
                            "unit": None,
                            "description": None
                        },
                        "api": {
                            "requests": {
                                "rate": None,
                                "period": None,
                                "quota": None,
                                "description": None
                            }
                        },
                        "compute": {
                            "vcpu": None,
                            "memory": None,
                            "unit": None
                        },
                        "other_limits": []
                    },
                    "features": {
                        "categories": [
                            {
                                "name": None,
                                "features": [
                                    {
                                        "name": None,
                                        "description": None,
                                        "included": None,
                                        "limit": None
                                    }
                                ]
                            }
                        ]
                    }
                }
            ],
            "discounts": [
                {
                    "type": None,
                    "amount": None,
                    "description": None,
                    "conditions": None,
                    "valid_until": None
                }
            ],
            "enterprise": {
                "available": None,
                "contact_sales": None,
                "minimum_seats": None,
                "custom_features": []
            },
            "ml_metadata": {
                "embeddings": {
                    "model": None,
                    "version": None,
                    "vectors": []
                },
                "confidence_scores": {
                    "pricing_accuracy": None,
                    "feature_accuracy": None
                },
                "last_validated": None
            },
            "agent_metadata": {
                "agent_name": None,
                "agent_version": None,
                "execution_time_seconds": None,
                "fallback_used": None,
                "comments": None
            }
        }

    def _create_extraction_prompt(self, content: str) -> str:
        """Create the extraction prompt with the content and schema"""
        schema_template = self._get_schema_template()
        
        return f"""You are a JSON extraction expert. Extract pricing information from the provided content into valid JSON format.
        
        Content to process:
        {content}
        
        Return ONLY a valid JSON object matching this schema (no other text):
        {json.dumps(schema_template, indent=2)}
        """

    def _generate_response(self, prompt: str, max_length: int = 4096) -> Dict:
        """Generate response from Llama model with improved parameters"""
        try:
            logger.info("Starting response generation...")
            
            # Encode the prompt
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
            input_length = len(inputs['input_ids'][0])
            logger.debug(f"Input length: {input_length} tokens")
            
            inputs = inputs.to(self.model.device)
            
            # Generate with more conservative parameters
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=3072,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            # Decode complete response
            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.debug(f"Generated text length: {len(full_text)}")
            logger.debug("First 500 chars of response: " + full_text[:500])
            
            # Extract and fix JSON
            try:
                # Find the first { and last } in the text
                start_idx = full_text.find('{')
                end_idx = full_text.rfind('}')
                
                if start_idx == -1 or end_idx == -1:
                    logger.error("No JSON structure found in response")
                    raise ValueError("No JSON structure found in response")
                    
                # Extract just the JSON part
                potential_json = full_text[start_idx:end_idx + 1].strip()
                
                # Log the extracted JSON for debugging
                logger.debug(f"Extracted JSON (first 500 chars): {potential_json[:500]}")
                
                # Clean up common issues
                potential_json = (
                    potential_json
                    .replace("'", '"')
                    .replace('None', 'null')
                    .replace('True', 'true')
                    .replace('False', 'false')
                    .replace('\n', '')
                    .replace('\t', '')
                    .replace('  ', ' ')
                    .replace('[ ', '[')
                    .replace(' ]', ']')
                    .replace(' :', ':')
                    .replace(': ', ':')
                    .replace(' ,', ',')
                    .replace(', ', ',')
                    .replace(',}', '}')
                    .replace(',]', ']')
                )
                
                # Ensure balanced braces
                open_count = potential_json.count('{')
                close_count = potential_json.count('}')
                if open_count > close_count:
                    logger.warning(f"Unbalanced braces detected: {open_count} opening vs {close_count} closing")
                    potential_json += '}' * (open_count - close_count)
                
                try:
                    # Try to parse the JSON
                    parsed_json = json.loads(potential_json)
                    logger.info("Successfully parsed JSON response")
                    
                    # Validate required structure
                    if not all(key in parsed_json for key in ['service', 'pricing_metadata', 'plans']):
                        raise ValueError("Missing required top-level keys in JSON response")
                    
                    return parsed_json
                    
                except json.JSONDecodeError as je:
                    logger.error(f"JSON parsing failed: {str(je)}")
                    logger.error(f"Problem at position {je.pos}, line {je.lineno}, column {je.colno}")
                    logger.error(f"Context: {potential_json[max(0, je.pos-50):min(len(potential_json), je.pos+50)]}")
                    raise
                
            except Exception as e:
                logger.error(f"Error processing response: {str(e)}")
                logger.error(f"Full response text: {full_text}")
                raise ValueError(f"Failed to generate valid JSON response: {str(e)}")
                
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
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
            "agent_name": "meta-llama/Llama-3.2-1B",
            "agent_version": "1.0",
            "execution_time_seconds": round(execution_time, 2),
            "fallback_used": False,
            "comments": ""
        }
        
        parsed_data["service"]["url"] = url
        parsed_data["pricing_metadata"]["last_updated"] = datetime.now().isoformat()
        parsed_data["pricing_metadata"]["versioning"] = {
            "current": "v1.0",
            "history": []
        }
        
        return parsed_data

    def parse_content(self, raw_content: str, url: str) -> Dict:
        """Parse raw content into structured schema"""
        self.start_time = time.time()
        max_retries = 3
        
        logger.info(f"Starting content parsing for URL: {url}")
        logger.info(f"Raw content length: {len(raw_content)} characters")
        
        for attempt in range(max_retries):
            try:
                logger.info(f"🔄 Attempt {attempt + 1}/{max_retries}")
                
                # Clean and truncate content
                cleaned_content = self._clean_content(raw_content)
                logger.info(f"Cleaned content length: {len(cleaned_content)} characters")
                
                if len(cleaned_content) > 4000:
                    logger.warning("⚠️ Content truncated to 4000 characters")
                    cleaned_content = cleaned_content[:4000] + "..."
                
                # Create and process prompt
                prompt = self._create_extraction_prompt(cleaned_content)
                logger.debug(f"Generated prompt length: {len(prompt)} characters")
                
                parsed_data = self._generate_response(prompt)
                
                # Validate structure
                required_keys = ["service", "pricing_metadata", "plans"]
                missing_keys = [key for key in required_keys if key not in parsed_data]
                if missing_keys:
                    raise ValueError(f"Missing required keys: {missing_keys}")
                
                # Add metadata
                parsed_data = self._add_metadata(parsed_data, url)
                
                total_time = time.time() - self.start_time
                logger.info(f"✅ Successfully parsed content in {total_time:.2f}s")
                return parsed_data
                
            except Exception as e:
                logger.warning(f"❌ Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error("❌ All attempts failed", exc_info=True)
                    raise
                logger.info(f"Waiting {2} seconds before next attempt...")
                time.sleep(2)

def main():
    """Example usage"""
    logger.info("🚀 Starting LlamaPricingAgent example")
    
    try:
        agent = LlamaPricingAgent()
        
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
            
            output_path = Path("parsed_content_storage/parsed_llama_airtable_20241217_021126.json")
            output_path.parent.mkdir(exist_ok=True)
            
            logger.info(f"Saving results to: {output_path}")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(parsed_data, f, indent=2)
                
            logger.info("✅ Process completed successfully")
            
    except Exception as e:
        logger.error(f"❌ Process failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
