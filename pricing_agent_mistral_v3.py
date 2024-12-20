import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Union
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bs4 import BeautifulSoup
import logging
import sys
import codecs
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mistral_v3_agent.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MistralV3PricingAgent:
    """Mistral-7B-v0.3-powered agent for parsing pricing pages into structured schema"""
    
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.3"):
        """Initialize the model and tokenizer"""
        self.start_time = None
        try:
            logger.info(f"🚀 Initializing MistralV3PricingAgent with model: {model_name}")
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
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}", exc_info=True)
            raise

    def extract_pricing_info(self, content: str) -> str:
        """First pass: Extract pricing information in a readable format"""
        prompt = f"""Extract detailed pricing information from the content in this specific format:

SERVICE INFORMATION
Name: [exact service name]
URL: [pricing page URL]
Logo URL: [if available]
Description: [brief service description]
Category: [primary category] / [secondary category if any]
Tags: [comma-separated list of relevant tags]
Business Model: [type: fixed/usage_based/tiered/hybrid] - [brief description]

PRICING METADATA
Currency: [e.g., USD]
Available Regions: [list regions]
Billing Cycles: [list available cycles]
Default Billing: [default cycle]
Custom Pricing: [Available/Not Available]
Free Tier: [Available/Not Available]

PLANS
[For each plan, include:]
Plan Name: [name]
Slug: [URL-friendly version of name]
Description: [plan description]
Popular: [Yes/No]
Base Price: [amount] [currency] per [period]
Per User: [Yes/No]

Key Features:
- [feature 1]
- [feature 2]
...

Limits:
- Users: [min-max if specified]
- Storage: [amount + unit]
- API Requests: [rate/quota if specified]
- Other: [any other numerical limits]

ENTERPRISE OFFERING
Available: [Yes/No]
Contact Sales Required: [Yes/No]
Minimum Seats: [if specified]
Custom Features:
- [feature 1]
- [feature 2]

DISCOUNTS
[For each discount:]
Type: [type]
Amount: [amount]
Description: [description]
Conditions: [conditions]
Valid Until: [date if specified]

Content to analyze:
{content}

Extract ALL available information, being as specific as possible with numbers, limits, and features."""

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=4096,  # Increased for more detailed output
                temperature=0.1,
                do_sample=True,
                top_p=0.95,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"First pass failed: {str(e)}")
            raise

    def structure_pricing_data(self, extracted_info: str) -> Dict:
        """Second pass: Convert extracted info into structured JSON"""
        schema = self._get_schema_template()
        
        prompt = f"""Convert this EXTRACTED pricing information into JSON following the exact schema.
Focus on including ALL extracted information in the correct format:

1. From SERVICE INFORMATION section:
- Use exact name, URL, logo, and description
- Include all tags from the Tags field
- Set business_model.type based on the Business Model field

2. From PRICING METADATA section:
- Set currency, regions, and billing cycles exactly as listed
- Convert Yes/No values to true/false for custom_pricing and free_tier
- Use current timestamp for last_updated

3. From PLANS section:
- Create an entry for each plan including Free, Team, Business, and Enterprise
- Include all features listed under "Key Features" in highlight_features
- Set is_popular based on "Popular: Yes/No"
- Structure pricing.base correctly with amount, period, and is_per_user
- Include all limits under the limits object
- Convert all numerical values (storage, API quotas, etc.) to numbers

4. From ENTERPRISE OFFERING section:
- Set enterprise.available and contact_sales based on the information
- Include all listed Custom Features

SCHEMA:
{json.dumps(schema, indent=2)}

EXTRACTED INFORMATION:
{extracted_info}

Return only valid JSON matching the schema exactly. Ensure:
1. All numbers are parsed as numbers, not strings
2. All booleans are true/false, not strings
3. Arrays are properly formatted
4. Missing values are null
5. Dates are in ISO8601 format"""

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=4096,
                temperature=0.1,
                do_sample=True,
                top_p=0.95,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            json_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Save the model's JSON response for debugging
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_path = Path(f"debug_output/json_response_mistralv3_{timestamp}.txt")
            with open(debug_path, 'w', encoding='utf-8') as f:
                f.write(json_response)
            logger.info(f"JSON response saved to: {debug_path}")
            
            # Parse and validate JSON
            try:
                data = json.loads(json_response)
                return data
            except json.JSONDecodeError:
                # Try to extract JSON from response
                return self._extract_json(json_response)
            
        except Exception as e:
            logger.error(f"Second pass failed: {str(e)}")
            raise

    def parse_content(self, raw_content: str, url: str) -> Dict:
        """Main parsing function that coordinates both passes"""
        self.start_time = time.time()
        
        try:
            # Clean content
            cleaned_content = self._clean_content(raw_content)
            
            # First pass: Extract information in readable format
            logger.info("Starting first pass: Extracting pricing information...")
            extracted_info = self.extract_pricing_info(cleaned_content)
            
            # Save intermediate output
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_path = Path(f"debug_output/extracted_info_mistralv3_{timestamp}.txt")
            debug_path.parent.mkdir(exist_ok=True)
            with open(debug_path, 'w', encoding='utf-8') as f:
                f.write(extracted_info)
            logger.info(f"Extracted info saved to: {debug_path}")
            
            # Second pass: Structure the information
            logger.info("Starting second pass: Structuring data...")
            structured_data = self.structure_pricing_data(extracted_info)
            
            # Add metadata
            structured_data = self._add_metadata(structured_data, url)
            
            total_time = time.time() - self.start_time
            logger.info(f"✅ Successfully parsed content in {total_time:.2f}s")
            return structured_data
            
        except Exception as e:
            logger.error(f"❌ Process failed: {str(e)}", exc_info=True)
            raise

    def _get_schema_template(self) -> Dict:
        """Get the JSON schema for pricing data"""
        schema_path = Path(__file__).parent / "schema.json"
        with open(schema_path, "r") as f:
            return json.load(f)

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

    def _extract_json(self, text: str) -> Dict:
        """Extract and parse JSON from model response"""
        try:
            logger.info("Starting JSON extraction...")
            
            # First try to parse as a list of function calls
            try:
                # Find the first [ and matching ]
                start_idx = text.find('[')
                if start_idx != -1:
                    # Find matching closing bracket
                    bracket_count = 0
                    for i in range(start_idx, len(text)):
                        if text[i] == '[':
                            bracket_count += 1
                        elif text[i] == ']':
                            bracket_count -= 1
                            if bracket_count == 0:
                                end_idx = i
                                break
                    
                    if bracket_count == 0:
                        json_str = text[start_idx:end_idx + 1].strip()
                        # Parse the function call array
                        function_calls = json.loads(json_str)
                        if isinstance(function_calls, list) and len(function_calls) > 0:
                            first_call = function_calls[-1]  # Take the last function call
                            # Check for arguments in function call
                            if "arguments" in first_call:
                                logger.info("Successfully extracted function call arguments")
                                return first_call["arguments"]
                        
                logger.info("No valid function call found, trying direct JSON extraction...")
            except json.JSONDecodeError as e:
                logger.info(f"Function call parsing failed: {str(e)}, trying direct JSON extraction...")
            
            # Find the last complete JSON object in the text
            json_objects = []
            start = 0
            while True:
                try:
                    start_idx = text.find('[{', start)  # Look for array of objects
                    if start_idx == -1:
                        break
                    
                    # Parse incrementally to find valid JSON
                    bracket_count = 0
                    for i in range(start_idx, len(text)):
                        if text[i] == '[':
                            bracket_count += 1
                        elif text[i] == ']':
                            bracket_count -= 1
                            if bracket_count == 0:
                                try:
                                    json_str = text[start_idx:i + 1]
                                    parsed = json.loads(json_str)
                                    if isinstance(parsed, list) and len(parsed) > 0:
                                        if "arguments" in parsed[-1]:
                                            json_objects.append((json_str, parsed[-1]["arguments"]))
                                except:
                                    pass
                    start = start_idx + 1
                    
                except Exception as e:
                    logger.debug(f"Error during JSON search: {str(e)}")
                    break
                
            if not json_objects:
                raise ValueError("No valid JSON found in response")
            
            # Take the last valid JSON object found
            json_str, parsed_data = json_objects[-1]
            logger.info("Successfully extracted JSON object")
            logger.info("Parsed data preview:")
            logger.info(json.dumps(parsed_data, indent=2)[:500] + "...")
            
            return parsed_data
            
        except Exception as e:
            logger.error(f"Failed to extract JSON: {str(e)}")
            logger.error("Full response:")
            logger.error(text)
            raise

    def _add_metadata(self, parsed_data: Dict, url: str) -> Dict:
        """Add metadata to the parsed data"""
        execution_time = time.time() - self.start_time
        
        parsed_data["agent_metadata"] = {
            "agent_name": "mistral-7b-instruct-v0.3",
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
    logger.info("🚀 Starting MistralV3PricingAgent example")
    
    try:
        agent = MistralV3PricingAgent()
        
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
            
            # Generate timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"parsed_content_storage/parsed_mistralv3_airtable_{timestamp}.json")
            output_path.parent.mkdir(exist_ok=True)
            
            logger.info(f"Saving results to: {output_path}")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(parsed_data, f, indent=2)
                
            logger.info("✅ Process completed successfully")
            
    except Exception as e:
        logger.error(f"❌ Process failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 