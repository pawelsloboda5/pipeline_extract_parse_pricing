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
from example_data import EXAMPLE_OUTPUT
import re

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

# At the top of the file, after other imports
DEBUG_MODERNBERT = True  # Set to False to reduce logging

class LlamaExtractorAgent:
    """Agent that extracts structured pricing information using Llama-3.2-1B-Instruct with 128k context"""
    
    def __init__(self):
        try:
            # Initialize Llama model with extended context
            logger.info("🚀 Initializing Llama-3.2-1B-Instruct model with 128k context")
            self.llama_tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.2-1B-Instruct",
                model_max_length=128000  # Set maximum context length to 128k
            )
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-3.2-1B-Instruct",
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_position_embeddings=128000  # Enable 128k context
            )
            
            # Initialize ModernBERT for embeddings
            logger.info("Loading ModernBERT for embeddings...")
            self.modernbert_tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
            self.modernbert_model = AutoModelForMaskedLM.from_pretrained(
                "answerdotai/ModernBERT-large",
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            
            device_info = next(self.llama_model.parameters()).device
            logger.info(f"✅ Models loaded successfully on device: {device_info}")
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {str(e)}", exc_info=True)
            raise

    def extract_pricing_info(self, content: str, content_data: dict) -> str:
        """Extract pricing information using Llama with long context support"""
        try:
            # Create a clearer, more focused prompt
            prompt = f"""You are a pricing information extractor for {content_data['title']}.

            TASK: Extract structured pricing information from the provided content below.
            URL: {content_data['url']}

            FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
            SERVICE INFORMATION
            - Name: [product name]
            - URL: [pricing page URL]
            - Logo URL: [if available]
            - Description: [brief description]
            - Summary: [longer summary]
            - Category: [main category]
            - Tags: [relevant tags]
            - Business Model: [pricing model type]

            PRICING METADATA
            - Currency: [main currency]
            - Available Regions: [regions or "Not specified"]
            - Billing Cycles: [list cycles]
            - Default Billing: [default cycle]
            - Custom Pricing: ["Available" or "Not Available"]
            - Free Tier: ["Available" or "Not Available"]

            PLANS
            For each plan:
            - Plan Name: [exact name]
              - Slug: [lowercase name]
              - Description: [plan description]
              - Popular: [Yes/No]
              - Base Price: [price + period]
              - Per User: [Yes/No]
              - Key Features: [bullet list]
              - Limits:
                - [limit type]: [number]

            ANALYZE THIS CONTENT:
            ================================================
            {content}
            ================================================

            REQUIREMENTS:
            1. Start with SERVICE INFORMATION section
            2. Include all pricing plans found
            3. Convert numbers to plain digits (e.g., "1000" not "1,000")
            4. Use "Not specified" for missing information
            5. Keep exact feature names and descriptions
            6. Maintain consistent indentation
            """

            # Generate and clean response
            outputs = self._generate_response(prompt)
            cleaned_response = self._clean_extraction(outputs)
            
            # Validate and fix formatting
            if not self._validate_structure(cleaned_response):
                cleaned_response = self._fix_missing_sections(cleaned_response)
            
            return cleaned_response

        except Exception as e:
            logger.error(f"Extraction failed: {str(e)}")
            raise

    def _generate_response(self, prompt: str) -> str:
        """Generate response with proper handling"""
        try:
            inputs = self.llama_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=128000,
                padding=False
            ).to(self.llama_model.device)

            logger.info("Starting generation...")
            outputs = self.llama_model.generate(
                **inputs,
                max_new_tokens=4096,
                temperature=0.1,
                do_sample=False,  # Use greedy decoding for more consistent output
                repetition_penalty=1.1,
                pad_token_id=self.llama_tokenizer.eos_token_id
            )

            response = self.llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the relevant part
            if "SERVICE INFORMATION" in response:
                response = response[response.index("SERVICE INFORMATION"):]
            
            return response

        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise

    def _validate_structure(self, text: str) -> bool:
        """Validate the structure of the extracted information"""
        required_sections = [
            "SERVICE INFORMATION",
            "PRICING METADATA",
            "PLANS",
            "ENTERPRISE OFFERING",
            "DISCOUNTS",
            "USE CASES"
        ]
        
        # Check for required sections
        missing_sections = []
        for section in required_sections:
            if section not in text:
                missing_sections.append(section)
                logger.warning(f"Missing required section: {section}")
        
        if missing_sections:
            logger.warning(f"Missing {len(missing_sections)} required sections: {', '.join(missing_sections)}")
            return False
                
        return True

    def process_content(self, raw_content: str, content_data: dict) -> str:
        """Main processing function"""
        try:
            logger.info("Starting content processing...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create output directories
            output_dir = Path("extracted_info")
            debug_dir = Path("debug_output")
            output_dir.mkdir(exist_ok=True)
            debug_dir.mkdir(exist_ok=True)
            
            # Extract information
            extracted_info = self.extract_pricing_info(raw_content, content_data)
            
            # Save main output
            output_path = output_dir / f"llama_extraction_{timestamp}.txt"
            logger.info(f"💾 Saving main extraction to: {output_path}")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(extracted_info)
            
            # Save debug output with raw content
            debug_path = debug_dir / f"llama_debug_{timestamp}.json"
            debug_data = {
                "timestamp": timestamp,
                "raw_content": raw_content,
                "extracted_info": extracted_info,
                "token_count": len(self.llama_tokenizer.encode(raw_content))
            }
            
            logger.info(f"💾 Saving debug info to: {debug_path}")
            with open(debug_path, 'w', encoding='utf-8') as f:
                json.dump(debug_data, f, indent=2)
            
            logger.info(f"""
            ✅ Extraction completed and saved:
            - Main output: {output_path}
            - Debug output: {debug_path}
            """)
            
            return extracted_info
            
        except Exception as e:
            logger.error(f"❌ Process failed: {str(e)}", exc_info=True)
            raise

    def _get_embeddings(self, text: str) -> torch.Tensor:
        """Get embeddings from ModernBERT"""
        try:
            inputs = self.modernbert_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=8192,
                padding=True,
                return_attention_mask=True
            ).to(self.modernbert_model.device)

            with torch.no_grad():
                # Get model outputs
                outputs = self.modernbert_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=True  # Request hidden states
                )
                
                # Get embeddings from the last hidden state
                # Use mean pooling over sequence length
                last_hidden_state = outputs.hidden_states[-1]
                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                token_embeddings = last_hidden_state * attention_mask
                sentence_embedding = token_embeddings.sum(1) / attention_mask.sum(1)
                
                logger.info(f"Generated embeddings with shape: {sentence_embedding.shape}")
                return sentence_embedding
            
        except Exception as e:
            logger.warning(f"Failed to get embeddings: {str(e)}")
            logger.info("Continuing without embeddings...")
            return None

    def _clean_extraction(self, text: str) -> str:
        """Clean up the extracted text"""
        try:
            # Remove everything before SERVICE INFORMATION
            if "SERVICE INFORMATION" in text:
                text = text[text.index("SERVICE INFORMATION"):]
            
            # Remove any content after the last valid section
            valid_sections = ["SERVICE INFORMATION", "PRICING METADATA", "PLANS"]
            last_index = -1
            for section in valid_sections:
                if section in text:
                    pos = text.rindex(section)
                    if pos > last_index:
                        last_index = pos
            
            if last_index != -1:
                next_section = text.find("\n\n", last_index)
                if next_section != -1:
                    text = text[:next_section]
            
            # Clean up formatting
            lines = []
            current_indent = 0
            for line in text.split('\n'):
                line = line.strip()
                if line:
                    if any(line.startswith(section) for section in valid_sections):
                        current_indent = 0
                        lines.extend(['', line, ''])
                    elif line.startswith('- Plan Name:'):
                        current_indent = 2
                        lines.extend(['', line])
                    elif line.startswith('-'):
                        if current_indent > 0:
                            lines.append('  ' * current_indent + line)
                        else:
                            lines.append(line)
                    else:
                        lines.append('  ' * current_indent + line)
            
            return '\n'.join(lines).strip()
            
        except Exception as e:
            logger.warning(f"Cleaning failed: {str(e)}")
            return text

    def _fix_missing_sections(self, text: str) -> str:
        """Add missing sections with 'Not specified' content"""
        required_sections = {
            "SERVICE INFORMATION": "\n- Name: Not specified\n- URL: Not specified\n- Description: Not specified",
            "PRICING METADATA": "\n- Currency: Not specified\n- Billing Cycles: Not specified",
            "PLANS": "\n- Plan Name: Not specified\n  - Description: Not specified",
            "ENTERPRISE OFFERING": "\n- Available: Not specified\n- Contact Sales Required: Not specified",
            "DISCOUNTS": "\n- Type: Not specified\n- Amount: Not specified",
            "USE CASES": "\n- Use Case: Not specified\n  - Target User: Not specified"
        }
        
        result = text
        for section, default_content in required_sections.items():
            if section not in result:
                result += f"\n\n{section}{default_content}"
                
        return result

    def _format_json_content(self, json_data: dict) -> str:
        """Format JSON content for extraction"""
        try:
            # Extract metadata
            metadata = {
                "title": json_data["data"]["title"],
                "description": json_data["data"]["description"],
                "url": json_data["data"]["url"],
                "tokens": json_data["data"]["usage"]["tokens"]
            }
            
            # Format content with metadata
            formatted_content = f"""
            Title: {metadata['title']}
            Description: {metadata['description']}
            URL: {metadata['url']}
            Token Count: {metadata['tokens']}
            
            Content:
            {json_data['data']['content']}
            """
            
            logger.info(f"Formatted JSON content with {metadata['tokens']} tokens")
            return formatted_content
            
        except KeyError as e:
            logger.error(f"Missing required field in JSON: {e}")
            raise
        except Exception as e:
            logger.error(f"Error formatting JSON content: {e}")
            raise

    def _validate_format(self, text: str) -> str:
        """Validate and fix formatting of the extracted text"""
        try:
            lines = text.split('\n')
            formatted_lines = []
            current_indent = 0
            
            for line in lines:
                line = line.strip()
                
                # Handle section headers
                if line in ["SERVICE INFORMATION", "PRICING METADATA", "PLANS", 
                           "ENTERPRISE OFFERING", "DISCOUNTS", "USE CASES"]:
                    formatted_lines.extend(['', line, ''])
                    current_indent = 0
                    continue
                
                # Handle plan entries
                if line.startswith('- Plan Name:'):
                    formatted_lines.extend(['', line])
                    current_indent = 2
                    continue
                
                # Handle indentation
                if line.startswith('-'):
                    if current_indent > 0:
                        formatted_lines.append('  ' * current_indent + line)
                    else:
                        formatted_lines.append(line)
                else:
                    formatted_lines.append(line)
            
            return '\n'.join(line for line in formatted_lines if line)
            
        except Exception as e:
            logger.warning(f"Format validation failed: {str(e)}")
            return text

def main():
    """Example usage"""
    logger.info("🚀 Starting LlamaExtractorAgent")
    
    try:
        agent = LlamaExtractorAgent()
        
        # Read from JSON file instead of raw text
        json_path = Path("rawContentExampleSchema.json")
        if json_path.exists():
            logger.info(f"Reading content from: {json_path}")
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Extract relevant fields from JSON
            content_data = {
                "title": json_data["data"]["title"],
                "description": json_data["data"]["description"],
                "url": json_data["data"]["url"],
                "content": json_data["data"]["content"]
            }
            
            # Format content for processing
            formatted_content = f"""
            Title: {content_data['title']}
            Description: {content_data['description']}
            URL: {content_data['url']}
            
            Content:
            {content_data['content']}
            """
            
            logger.info(f"Content tokens from JSON: {json_data['data']['usage']['tokens']}")
            
            # Pass both formatted_content and content_data
            extracted_info = agent.process_content(formatted_content, content_data)
            
            logger.info("🎉 Process completed successfully")
            logger.info("Exiting program...")
            sys.exit(0)
            
        else:
            logger.error(f"❌ JSON file not found: {json_path}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Process failed: {str(e)}", exc_info=True)
        sys.exit(1)

def _clean_content(self, content: str) -> str:
    """Clean and prepare content from JSON"""
    try:
        logger.info("Starting content cleaning...")
        
        # Basic HTML and markdown cleaning
        soup = BeautifulSoup(content, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        
        # Remove markdown links
        text = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'\1', text)
        
        # Remove multiple spaces and newlines
        text = ' '.join(text.split())
        
        # Remove any remaining markdown symbols
        text = text.replace('*', '').replace('`', '').replace('#', '')
        
        logger.info(f"Cleaned text length: {len(text)} characters")
        logger.info(f"Preview of cleaned text: {text[:200]}...")
        
        return text
        
    except Exception as e:
        logger.warning(f"❌ Cleaning failed: {str(e)}")
        return content

if __name__ == "__main__":
    main()

