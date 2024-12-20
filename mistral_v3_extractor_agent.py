#mistral_v3_extractor_agent.py
import json
import time
from datetime import datetime
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
from bs4 import BeautifulSoup
import logging
import sys
from extraction_format import EXTRACTION_FORMAT
from example_data import EXAMPLE_INPUT, EXAMPLE_OUTPUT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mistral_v3_extractor.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MistralV3ExtractorAgent:
    """Agent that extracts structured pricing information using Mistral-7B-Instruct-v0.3"""
    
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.3"):
        """Initialize the model and tokenizer"""
        try:
            logger.info(f"🚀 Initializing MistralV3ExtractorAgent with model: {model_name}")
            
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            logger.info("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,  # Using bfloat16 as per docs
                device_map="auto"
            )
            
            device_info = next(self.model.parameters()).device
            logger.info(f"✅ Model loaded successfully on device: {device_info}")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}", exc_info=True)
            raise

    def extract_pricing_info(self, content: str) -> str:
        """Extract pricing information using Mistral's chat format"""
        try:
            # Clean content first
            cleaned_content = self._clean_content(content)
            
            # Construct messages following Mistral's chat format
            messages = [
                {
                    "role": "system",
                    "content": """You are a specialized pricing information extractor that analyzes SaaS products.
                    You must follow the format exactly and extract precise information about pricing, features, and use cases.
                    Your output must start with 'SERVICE INFORMATION' and contain ONLY the extracted information.
                    Do not include the EXAMPLE INPUT or EXAMPLE OUTPUT in your response. Do not include any other text or instructions in your response.
                    Do not include the raw content in your response.
                    """
                },
                {
                    "role": "user",
                    "content": f"Here's an example of input content and how it should be analyzed:\n\nEXAMPLE INPUT:\n{EXAMPLE_INPUT}\n\nEXAMPLE OUTPUT:\n{EXAMPLE_OUTPUT}"
                },
                {
                    "role": "assistant",
                    "content": "I understand. I will analyze the content and provide only the extracted information, starting with SERVICE INFORMATION."
                },
                {
                    "role": "user",
                    "content": f"Analyze this content following the format shown in the example:\n\n{cleaned_content}"
                }
            ]

            # Apply Mistral's chat template
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )

            # Move inputs to correct device
            inputs = inputs.to(self.model.device)

            # Generate with specific parameters for structured output
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=4096,
                temperature=0.0,  # Use 0 temperature for consistent output
                do_sample=False,  # Disable sampling for deterministic output
                top_p=1.0,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1  # Slight penalty to avoid repetition
            )

            # Decode and clean the response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the structured output part
            if "SERVICE INFORMATION" in response:
                response = response[response.index("SERVICE INFORMATION"):]
            
            return response

        except Exception as e:
            logger.error(f"Extraction failed: {str(e)}")
            raise

    def _clean_content(self, content: str) -> str:
        """Clean and prepare content for parsing"""
        try:
            # Remove HTML if present
            soup = BeautifulSoup(content, 'html.parser')
            for element in soup(['script', 'style', 'meta', 'link']):
                element.decompose()
            
            # Get text and normalize spacing
            text = soup.get_text(separator=' ', strip=True)
            text = ' '.join(text.split())
            
            # Remove any markdown-style formatting
            text = text.replace('```', '').replace('`', '')
            
            return text
            
        except Exception as e:
            logger.warning(f"Error cleaning content: {str(e)}")
            return content

    def process_content(self, raw_content: str) -> str:
        """Main processing function with error handling and logging"""
        try:
            logger.info("Starting content processing...")
            
            # Extract information
            extracted_info = self.extract_pricing_info(raw_content)
            
            # Save output with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"extracted_info/extracted_info_mistralv3_{timestamp}.txt")
            output_path.parent.mkdir(exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(extracted_info)
            
            logger.info(f"✅ Extraction completed. Results saved to: {output_path}")
            return extracted_info
            
        except Exception as e:
            logger.error(f"❌ Process failed: {str(e)}", exc_info=True)
            raise

def main():
    """Example usage"""
    logger.info("🚀 Starting MistralV3ExtractorAgent")
    
    try:
        agent = MistralV3ExtractorAgent()
        
        # Read content from file
        content_path = Path("raw_content_storage/raw_content_airtable_demo.txt")
        if content_path.exists():
            logger.info(f"Reading content from: {content_path}")
            with open(content_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            extracted_info = agent.process_content(content)
            logger.info("✅ Process completed successfully")
            
            # Save debug output
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_path = Path(f"debug_output/extracted_info_mistralv3_{timestamp}.txt")
            debug_path.parent.mkdir(exist_ok=True)
            with open(debug_path, 'w', encoding='utf-8') as f:
                f.write(extracted_info)
            logger.info(f"Debug output saved to: {debug_path}")
            
    except Exception as e:
        logger.error(f"❌ Process failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
