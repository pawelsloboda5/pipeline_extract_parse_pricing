import json
import logging
import sys
from pathlib import Path
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from bs4 import BeautifulSoup
import time
import re
import os

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llama_extractor_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

from extraction_format import EXTRACTION_FORMAT
from example_data import EXAMPLE_OUTPUT

# Add at top of file
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

# Tokenizer config
TOKENIZER_CONFIG = {
    "max_length": 4096,
    "padding_side": "left",
    "truncation_side": "left",
    "use_fast": True
}

# Generation config (more thorough, less speed):
MODEL_CONFIG = {
    "temperature": 0.1,          # Slightly higher for a bit more "creative" extraction
    "repetition_penalty": 1.1,
    "num_return_sequences": 1,
    "do_sample": True,           # Enable sampling
    "num_beams": 2,              # Small beam search
    "length_penalty": 1.0
}

# Memory config for RTX 3060
MEMORY_CONFIG = {
    "device_map": "auto",
    "low_cpu_mem_usage": True,
    "max_memory": {0: "10GiB"},
    "offload_folder": "offload",
    "torch_dtype": torch.float16
}

# Directory for output
OUTPUT_DIR = Path("extracted_info")
OUTPUT_DIR.mkdir(exist_ok=True)

# Format templates
basic_format = """SERVICE INFORMATION
- Name: {title}
- URL: {url}
- Logo URL: [if available]
- Description: [brief description]
- Summary: [comprehensive service summary]
- Category:
  - Primary: [main category]
  - Secondary: [sub-category]
  - Tags: [comma-separated list of all relevant tags]
- Business Model:
  - Type: [list all that apply: subscription, usage-based, flat-rate, etc.]
  - Description: [explain the pricing model]

PRICING METADATA
- Currency: [primary currency code]
- Regions: [list all available regions]
- Billing Cycles:
  - Available: [list all cycles]
  - Default: [primary billing cycle]
- Custom Pricing Available: [Yes/No]
- Free Tier Available: [Yes/No]"""

plans_format = """PLANS
For each plan found, provide:
- Name: [exact name]
- Slug: [kebab-case-name]
- Description: [full description]
- Is Popular: [Yes/No]
- Highlight Features: [3-5 key features or distinctive bullet points if available]
- Pricing:
  - Base:
    - Amount: [numeric value, if available]
    - Period: [monthly/yearly/other, if available]
    - Currency: [currency code]
    - Is Per User: [Yes/No]
  - Usage Based Components:
    - Name: [component name]
    - Type: [usage type]
    - Unit: [billing unit]
    - Tiers:
      - Range: [min-max]
      - Unit Price: [price per unit]
      - Flat Fee: [if applicable]
- Limits:
  - Users:
    - Minimum: [number]
    - Maximum: [number]
  - Storage:
    - Amount: [number]
    - Unit: [GB/TB/etc]
  - API:
    - Requests:
      - Rate: [number]
      - Period: [time period]
      - Quota: [total limit if specified]
  - Compute:
    - VCPU: [number]
    - Memory: [amount]
    - Unit: [GB/TB/etc]
  - Other Limits:
    - Name: [limit name]
    - Value: [limit value]
    - Description: [explanation]
- Features (Group by category if possible):
  - Category Name: [e.g., Security, Integrations, Support, etc.]
    - Feature Name: [specific feature]
    - Description: [any detail]
    - Included: [Yes/No]
    - Limit: [any mention of usage restrictions]"""

remaining_format = """ADD-ONS (if any)
For each add-on product/service:
- Name: [exact add-on name]
- Price: [ALL pricing details]
- Description: [complete description]
- Features: [ALL included features or bullet points]
- Availability: [which plans can use this add-on]
- Requirements: [any prerequisites]
- Limitations: [any usage or functional restrictions]

ENTERPRISE
- Available: [Yes/No]
- Contact Sales Required: [Yes/No]
- Minimum Seats: [number if specified, otherwise "Not specified"]
- Custom Features: [list all enterprise-specific features or capabilities]

DISCOUNTS
For each discount, if mentioned:
- Type: [discount type, e.g. volume discount, coupon, etc.]
- Amount: [value or percentage]
- Description: [full details]
- Conditions: [requirements or constraints]
- Valid Until: [expiration date or "Not specified"]"""

def log_debug_info(stage: str, **kwargs):
    """Enhanced debug logging"""
    logger.debug(f"=== {stage} ===")
    for key, value in kwargs.items():
        if isinstance(value, str) and len(value) > 500:
            logger.debug(f"{key}: {value[:500]}... (truncated)")
        else:
            logger.debug(f"{key}: {value}")

def estimate_tokens(text: str, tokenizer) -> int:
    """Estimate tokens using the model tokenizer"""
    tokens = tokenizer.encode(text)
    logger.debug(f"Token count: {len(tokens)}")
    return len(tokens)

def summarize_content(full_content: str, max_tokens: int = 2000) -> str:
    """
    If content is too large, truncate or summarize. 
    This version just truncates for simplicity.
    """
    tokens = full_content.split()
    if len(tokens) > max_tokens:
        truncated = " ".join(tokens[:max_tokens])
        truncated += "\n\n(Content truncated for brevity)"
        return truncated
    return full_content

def create_prompt(data: dict) -> str:
    """Create a prompt for Llama 3.2's instruction following"""
    title = data.get("title", "Not specified")
    url = data.get("url", "Not specified")
    full_content = data.get("content", "")

    # If content is huge, consider truncating or summarizing here:
    # full_content = summarize_content(full_content, max_tokens=2000)

    prompt = f"""Task: Extract detailed pricing information from the provided content using the structure below. 
Replace any placeholders with real data, or 'Not specified' if missing. 
Try to be as thorough and precise as possible.

SERVICE INFORMATION
- Name: {title}
- URL: {url}
- Logo URL: [if available]
- Description: [brief description]
- Summary: [comprehensive service summary]
- Category:
  - Primary: [main category]
  - Secondary: [sub-category]
  - Tags: [comma-separated list of all relevant tags]
- Business Model:
  - Type: [list all that apply: subscription, usage-based, flat-rate, etc.]
  - Description: [explain the pricing model]

PRICING METADATA
- Currency: [primary currency code]
- Regions: [list all available regions]
- Billing Cycles:
  - Available: [list all cycles]
  - Default: [primary billing cycle]
- Custom Pricing Available: [Yes/No]
- Free Tier Available: [Yes/No]

PLANS
For each plan found:
- Name: ...
- Slug: ...
- Description: ...
- Is Popular: ...
- Highlight Features: ...
- Pricing:
  - Base:
    - Amount:
    - Period:
    - Currency:
    - Is Per User:
  - Usage Based Components:
    - ...
- Limits:
  - ...
- Features:
  - ...

ADD-ONS
- Name:
- Price:
- ...
- Requirements:
- ...

ENTERPRISE
- Available:
- Contact Sales Required:
- ...

DISCOUNTS
- Type:
- Amount:
- Description:
- ...

CONTENT TO ANALYZE:
{full_content}"""

    return prompt.strip()

def run_inference(prompt: str, tokenizer, model) -> str:
    """Run inference with the updated, more thorough config."""
    try:
        token_count = estimate_tokens(prompt, tokenizer)
        log_debug_info("Inference Start",
            model_id=MODEL_ID,
            token_count=token_count,
            config=MODEL_CONFIG
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Adjusted pipeline config to allow slower but deeper generation
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=2048,              # More generation room
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            clean_up_tokenization_spaces=True,
            min_length=50,
            no_repeat_ngram_size=3,
            early_stopping=False,            # Let it explore more
            do_sample=MODEL_CONFIG["do_sample"],
            top_p=0.9,                       # Restrict to top 90% tokens
            temperature=MODEL_CONFIG["temperature"],
            repetition_penalty=MODEL_CONFIG["repetition_penalty"],
            num_beams=MODEL_CONFIG["num_beams"],
            length_penalty=MODEL_CONFIG["length_penalty"]
        )

        start_time = time.time()
        outputs = pipe(prompt)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        result = outputs[0]['generated_text']

        # Optional post-processing, e.g. remove everything after "Content to analyze:"
        # But only if the model repeats itself
        if "CONTENT TO ANALYZE:" in result.upper():
            # Some models repeat the instructions; remove after 'CONTENT TO ANALYZE:'
            pattern = re.compile(r"(CONTENT TO ANALYZE:)", re.IGNORECASE)
            match = pattern.search(result)
            if match:
                result = result[:match.start()].strip()
        
        # Format + validate
        result = format_output(result)
        result = validate_output(result)
        return result.strip()
        
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise

def load_model():
    """Load model with the memory settings & quantization config."""
    try:
        logger.info(f"Loading Llama model: {MODEL_ID}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            **TOKENIZER_CONFIG
        )
        
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            max_memory={0: "10GiB"},
            offload_folder="offload"
        )
        
        logger.info("Model loaded successfully")
        return tokenizer, model
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

def load_json_data(input_path: str) -> dict:
    """Load and validate JSON data."""
    logger.info(f"Loading data from {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    return json_data.get('data', {})

def save_outputs(extracted_info: str, data: dict, prompt: str):
    """Save extraction outputs and debug info."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Main output
    output_file = OUTPUT_DIR / f"llama_extraction_{timestamp}.txt"
    print("Saved in output file")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(extracted_info)
    
    # Debug info
    debug_file = OUTPUT_DIR / f"llama_debug_{timestamp}.json"
    debug_data = {
        "input": {
            "title": data.get("title"),
            "url": data.get("url"),
            "tokens": data.get("usage", {}).get("tokens")
        },
        "prompt": prompt,
        "output": extracted_info
    }
    with open(debug_file, 'w', encoding='utf-8') as f:
        json.dump(debug_data, f, indent=2)

def validate_output(result: str) -> str:
    """Ensure minimal sections exist. Fill missing with 'Not specified'."""
    required_sections = {
        'SERVICE INFORMATION': ['Name', 'URL', 'Logo URL', 'Description'],
        'PRICING METADATA': ['Currency', 'Regions', 'Billing Cycles'],
        'PLANS': ['Name', 'Slug', 'Description', 'Pricing'],
        'ADD-ONS': ['Name', 'Price', 'Description'],
        'ENTERPRISE': ['Available', 'Contact Sales Required'],
        'DISCOUNTS': ['Type', 'Amount', 'Description']
    }
    
    for section, fields in required_sections.items():
        if section not in result:
            result += f"\n\n{section}"
            for field in fields:
                result += f"\n- {field}: Not specified"
                
    return result

def format_output(text: str) -> str:
    """Format for consistent indentation & section grouping."""
    sections = {}
    current_section = None
    current_content = []
    
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # If line is uppercase and doesn't contain a colon, treat as new section
        if line.upper() == line and ':' not in line:
            if current_section and current_content:
                sections[current_section] = current_content
            current_section = line
            current_content = []
        else:
            # Fix spacing for dashes and colons
            line = re.sub(r'\s*-\s*', '- ', line)
            line = re.sub(r'\s*:\s*', ': ', line)
            line = re.sub(r'&amp;', '&', line)
            
            # Indent if starts with dash
            if line.startswith('- '):
                current_content.append(line)
            else:
                current_content.append('- ' + line)
    
    # Add the last section
    if current_section and current_content:
        sections[current_section] = current_content
    
    # Re-order sections
    ordered_sections = [
        'SERVICE INFORMATION',
        'PRICING METADATA',
        'PLANS',
        'ADD-ONS',
        'ENTERPRISE',
        'DISCOUNTS'
    ]
    
    formatted = []
    for sec in ordered_sections:
        if sec in sections:
            formatted.append(sec)
            formatted.extend(['  ' + ln for ln in sections[sec]])
            formatted.append('')  # blank line
        else:
            formatted.append(sec)
            formatted.append('  - Not specified\n')
    
    return '\n'.join(formatted).strip()

def extract_in_steps(prompt: str, tokenizer, model, title: str, url: str) -> str:
    """
    Example multi-step approach.
    You can reduce the instructions in each step so they're not repeated verbatim.
    """
    try:
        # Step 1
        basic_prompt = f"""[INST]
Extract ONLY the SERVICE INFORMATION and PRICING METADATA below. Use the structure here:
{basic_format}

CONTENT:
{prompt}
[/INST]"""

        # Step 2
        plans_prompt = f"""[INST]
Extract ONLY the PLANS below. Use the structure here:
{plans_format}

CONTENT:
{prompt}
[/INST]"""

        # Step 3
        remaining_prompt = f"""[INST]
Extract ADD-ONS, ENTERPRISE, and DISCOUNTS sections below. Use the structure here:
{remaining_format}

CONTENT:
{prompt}
[/INST]"""

        logger.info("Step 1: Extracting basic information...")
        basic_info = run_inference(basic_prompt, tokenizer, model)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Step 2: Extracting plans information...")
        plans_info = run_inference(plans_prompt, tokenizer, model)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Step 3: Extracting additional sections...")
        remaining_info = run_inference(remaining_prompt, tokenizer, model)

        # Combine final output
        combined = f"{basic_info}\n\n{plans_info}\n\n{remaining_info}"
        return format_output(combined)
        
    except Exception as e:
        logger.error(f"Step-wise extraction failed: {str(e)}")
        raise

def main():
    try:
        data = load_json_data("rawContentExampleSchema.json")
        title = data.get("title", "Not specified")
        url = data.get("url", "Not specified")
        
        # Single-step or multi-step is up to you:
        # 1) Single-step:
        # prompt = create_prompt(data)
        # extracted_info = run_inference(prompt, tokenizer, model)

        # 2) Multi-step:
        full_prompt = create_prompt(data)
        tokenizer, model = load_model()
        
        logger.info("Starting extraction...")
        start_time = time.time()
        
        extracted_info = extract_in_steps(full_prompt, tokenizer, model, title, url)
        
        process_time = time.time() - start_time
        logger.info(f"Extraction completed in {process_time:.2f} seconds")
        
        # Save results
        save_outputs(extracted_info, data, full_prompt)
        
    except Exception as e:
        logger.error(f"Process failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
