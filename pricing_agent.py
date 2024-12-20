import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Union
from pathlib import Path
from openai import OpenAI
from bs4 import BeautifulSoup
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class PricingAgent:
    """OpenAI-powered agent for parsing pricing pages into structured schema"""
    
    def __init__(self, model="gpt-4-turbo-preview"):
        self.model = model
        self.start_time = None
        
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the OpenAI agent"""
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

    def parse_content(self, raw_content: str, url: str) -> Dict:
        """Parse raw content into structured schema"""
        self.start_time = time.time()
        
        try:
            # Clean the content first
            cleaned_content = self._clean_content(raw_content)
            
            # Create messages for the API call
            messages = [
                {"role": "system", "content": self._create_system_prompt()},
                {"role": "user", "content": self._create_extraction_prompt(cleaned_content)}
            ]
            
            # Make the API call using updated client
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=4000,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            try:
                parsed_data = json.loads(response.choices[0].message.content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {str(e)}")
                logger.error(f"Response content: {response.choices[0].message.content}")
                raise
            
            # Add metadata
            parsed_data = self._add_metadata(parsed_data, url)
            
            return parsed_data
            
        except Exception as e:
            logger.error(f"Error parsing content: {str(e)}")
            raise
        
    def _clean_content(self, content: str) -> str:
        """Clean and prepare content for parsing"""
        try:
            # Use BeautifulSoup to clean HTML if present
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove script and style elements
            for element in soup(['script', 'style', 'meta', 'link']):
                element.decompose()
                
            # Get text content
            text = soup.get_text(separator=' ', strip=True)
            
            # Basic text cleaning
            text = ' '.join(text.split())
            
            return text
            
        except Exception as e:
            logger.warning(f"Error cleaning content, using raw content: {str(e)}")
            return content
            
    def _add_metadata(self, parsed_data: Dict, url: str) -> Dict:
        """Add metadata to the parsed data"""
        execution_time = time.time() - self.start_time
        
        parsed_data["agent_metadata"] = {
            "agent_name": self.model,
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
        
    def _validate_schema_compliance(self, data: Dict, schema_path: str = "schema.json") -> None:
        """Validate that the parsed data complies with our schema"""
        # Load schema
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        
        def validate_against_schema(data: Dict, schema: Dict, path: str = "") -> None:
            """Recursively validate data against schema"""
            for key, value in schema.items():
                current_path = f"{path}.{key}" if path else key
                
                # Check required field exists
                if key not in data:
                    raise ValueError(f"Missing required field: {current_path}")
                
                # Validate type
                if isinstance(value, dict):
                    if not isinstance(data[key], dict):
                        raise ValueError(f"Invalid type for {current_path}: expected dict")
                    validate_against_schema(data[key], value, current_path)
                elif isinstance(value, list):
                    if not isinstance(data[key], list):
                        raise ValueError(f"Invalid type for {current_path}: expected list")
                    # Could add more specific list validation here
            
        try:
            validate_against_schema(data, schema)
        except Exception as e:
            logger.error(f"Schema validation failed: {str(e)}")
            raise

    def _create_extraction_prompt(self, content: str) -> str:
        """Create the extraction prompt with the content and schema"""
        schema_template = self._get_schema_template()
        
        return f"""Analyze this pricing page content and extract all information into a JSON object matching our schema exactly.
        
        Use this schema template (fill in all fields, use null for missing values):
        {json.dumps(schema_template, indent=2)}

        Important Notes:
        1. Follow the exact schema structure
        2. Use null for missing values, don't omit fields
        3. For prices, use numbers without currency symbols (e.g., 10.0 instead of "$10")
        4. For dates, use ISO8601 format (YYYY-MM-DDTHH:mm:ssZ)
        5. For slugs, convert plan names to lowercase, replace spaces with hyphens
        6. For arrays, use empty array [] if no values available
        7. For boolean values, use true/false (lowercase)
        8. For pricing ranges, include both min and max when available
        9. For feature categories, group similar features together
        10. For usage-based pricing, include all tier information

        Specific Field Guidelines:
        - service.name: The company/product name
        - service.category.primary: Main category (e.g., "Infrastructure", "Developer Tools")
        - service.business_model.type: Array of ["fixed", "usage_based", "tiered", "hybrid"]
        - pricing_metadata.currency: Three-letter currency code (e.g., "USD")
        - plans[].slug: URL-friendly version of plan name (e.g., "enterprise-plus")
        - plans[].pricing.base.amount: Numeric value without currency symbol
        - plans[].limits: Include all applicable limits with proper units
        - discounts[].amount: Use percentage or fixed amount (e.g., "20%" or 100)

        Here's the content to analyze:

        {content}

        Return only the valid JSON object matching our schema exactly. Do not include any explanatory text or comments.
        Ensure all JSON properties are properly quoted and all syntax is valid.
        """

def main():
    # Example usage
    agent = PricingAgent()
    
    # Load sample content
    content_path = Path("raw_content_storage/raw_content_airtable_20241217_012911.txt")
    with open(content_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Parse content
    try:
        parsed_data = agent.parse_content(
            content,
            url="https://airtable.com/pricing"
        )
        
        # Save parsed data
        output_path = Path("parsed_content_storage/parsed_airtable_20241217_012911.json")
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(parsed_data, f, indent=2)
            
        logger.info(f"Successfully parsed and saved data to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to parse content: {str(e)}")

if __name__ == "__main__":
    main() 