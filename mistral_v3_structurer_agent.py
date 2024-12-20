#mistral_v3_structurer_agent.py
import json
import time
from datetime import datetime
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import sys
import json
from typing import Dict


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mistral_v3_structurer.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MistralV3StructurerAgent:
    """Second-pass agent that converts extracted info into JSON schema"""
    
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.3"):
        """Initialize the model and tokenizer"""
        try:
            logger.info(f"🚀 Initializing MistralV3StructurerAgent with model: {model_name}")
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
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}", exc_info=True)
            raise

    def _get_schema_template(self) -> Dict:
        """Get the JSON schema for pricing data"""
        schema_path = Path(__file__).parent / "schema.json"
        with open(schema_path, "r") as f:
            return json.load(f)

    def structure_data(self, extracted_info: str) -> Dict:
        """Convert extracted info into structured JSON"""
        schema = self._get_schema_template()
        print(schema)
        
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

Return only valid JSON matching the schema exactly. Ensure:
1. All numbers are parsed as numbers, not strings
2. All booleans are true/false, not strings
3. Arrays are properly formatted
4. Missing values are null
5. Dates are in ISO8601 format

EXTRACTED INFORMATION:
{extracted_info}

"""
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
            
            # Save raw response for debugging
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_path = Path(f"debug_output/json_response_mistralv3_{timestamp}.txt")
            debug_path.parent.mkdir(exist_ok=True)
            with open(debug_path, 'w', encoding='utf-8') as f:
                f.write(json_response)
            logger.info(f"Raw response saved to: {debug_path}")
            
            # Parse and validate JSON
            try:
                data = json.loads(json_response)
                return data
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON response")
                raise
            
        except Exception as e:
            logger.error(f"Structuring failed: {str(e)}")
            raise

    def _add_metadata(self, structured_data: Dict, url: str) -> Dict:
        """Add metadata to the structured data"""
        structured_data["agent_metadata"] = {
            "agent_name": "mistral-7b-instruct-v0.3",
            "agent_version": "1.0",
            "execution_time_seconds": None,
            "fallback_used": False,
            "comments": ""
        }
        
        if "service" not in structured_data:
            structured_data["service"] = {}
        structured_data["service"]["url"] = url
        
        return structured_data

    def process_extracted_info(self, extracted_info_path: Path, url: str) -> Dict:
        """Main processing function"""
        try:
            # Read extracted info
            logger.info(f"Reading extracted info from: {extracted_info_path}")
            with open(extracted_info_path, 'r', encoding='utf-8') as f:
                extracted_info = f.read()
            
            # Structure the data
            logger.info("Starting structuring...")
            structured_data = self.structure_data(extracted_info)
            
            # Add metadata
            structured_data = self._add_metadata(structured_data, url)
            
            # Save output
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"parsed_content_storage/parsed_mistralv3_{timestamp}.json")
            output_path.parent.mkdir(exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(structured_data, f, indent=2)
            
            logger.info(f"✅ Structuring completed. Results saved to: {output_path}")
            return structured_data
            
        except Exception as e:
            logger.error(f"❌ Process failed: {str(e)}", exc_info=True)
            raise

def main():
    """Example usage"""
    logger.info("🚀 Starting MistralV3StructurerAgent")
    
    try:
        agent = MistralV3StructurerAgent()
        
        # Use the latest extracted info file
        extracted_info_dir = Path("extracted_info")
        if not extracted_info_dir.exists():
            raise FileNotFoundError("No extracted info directory found")
            
        latest_file = max(extracted_info_dir.glob("extracted_info_mistralv3_*.txt"))
        
        structured_data = agent.process_extracted_info(
            latest_file,
            url="https://airtable.com/pricing"
        )
        logger.info("✅ Process completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Process failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()