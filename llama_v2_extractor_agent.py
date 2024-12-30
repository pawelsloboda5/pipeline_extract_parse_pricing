import json
import logging
import sys
from pathlib import Path
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import BitsAndBytesConfig
from time import perf_counter
import re
import os
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import time
from functools import wraps

# ------------------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------------------
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tulu_extractor_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Ensure environment variables are set before torch init
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'

# ------------------------------------------------------------------------
# Model & Tokenizer Config
# ------------------------------------------------------------------------
MODEL_ID = "allenai/Llama-3.1-Tulu-3-8B-SFT"  # Tulu 3 8B
TOKENIZER_CONFIG = {
    "max_length": 4096,
    "padding_side": "left",
    "truncation_side": "left",
    "use_fast": True
}

# Generation config: Enough variety for extraction
GENERATION_CONFIG = {
    "max_new_tokens": 4096,
    "min_length": 200,
    "temperature": 0.3,
    "top_p": 0.95,
    "repetition_penalty": 1.2,
    "num_beams": 2,
    "do_sample": True,
    "early_stopping": False
}

# 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Memory/offload
MEMORY_CONFIG = {
    "device_map": "auto",
    "torch_dtype": torch.float16,
    "low_cpu_mem_usage": True,
    "max_memory": {0: "10GiB"},
    "offload_folder": "offload"
}

# Directory for storing output
OUTPUT_DIR = Path("extracted_info")
OUTPUT_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------
def load_text_data(input_path: str) -> dict:
    """Load text data from a file and return a dict with content."""
    logger.info(f"Loading data from {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # For text files, we'll create a simple dict with the content
    # and derive title from filename
    filename = Path(input_path).stem
    return {
        "title": filename,
        "url": "Not specified",  # Since we're reading from local file
        "content": content
    }

def create_prompt(data: dict) -> str:
    """
    Minimal prompt:
      1) Show the content (HTML/text).
      2) Provide EXACT format instructions.
      3) Instruct to fill ONLY the structure with real data.
    """
    title = "Zapier"
    url = "https://zapier.com"
    content = data.get("content", "")

    prompt = f"""
CONTENT:
{content}

You are an AI assistant that extracts pricing information from any SaaS pricing page. 
Please output ONLY the extracted data in this exact structure. 
If data is missing, use "Not specified". 
DO NOT add extra commentary.

FORMAT:

SERVICE INFORMATION
- Name: {title}
- URL: {url}
- Logo URL: [if available]
- Description: [short description]
- Summary: [long summary or overview]
- Category:
  - Primary: [main category or domain]
  - Secondary: [secondary category if any]
  - Tags: [comma-separated relevant tags]
- Business Model:
  - Type: [subscription, usage-based, flat-rate, etc.]
  - Description: [details if any]

PRICING METADATA
- Currency: 
- Regions: 
- Billing Cycles:
  - Available: 
  - Default: 
- Custom Pricing Available: [Yes/No]
- Free Tier Available: [Yes/No]

PLANS
For each plan found, list:
- Name: 
- Slug: 
- Description: 
- Is Popular: [Yes/No or Not specified]
- Highlight Features: [list of bullet points if any]
- Pricing:
  - Base:
    - Amount:
    - Period:
    - Currency:
    - Is Per User:
  - Usage Based Components:
    - Name:
    - Type:
    - Unit:
    - Tiers:
      - Range:
      - Unit Price:
      - Flat Fee:
- Limits:
  - Users (min/max):
  - Storage (amount/unit):
  - API (requests quota):
  - Compute (vcpu, memory, etc):
  - Other Limits:
- Features: [list or categories of features]

ADD-ONS
- Name:
- Price:
- Description:
- Features:
- Availability:
- Requirements:
- Limitations:

ENTERPRISE
- Available: [Yes/No]
- Contact Sales Required: [Yes/No]
- Minimum Seats: 
- Custom Features: 

DISCOUNTS
- Type:
- Amount:
- Description:
- Conditions:
- Valid Until:
""".strip()

    return prompt

def validate_extraction(extracted_text: str) -> tuple[bool, Optional[Dict]]:
    """
    Validate that the extraction contains required fields and structure.
    Returns (is_valid, parsed_data).
    """
    required_sections = [
        "SERVICE INFORMATION",
        "PRICING METADATA",
        "PLANS"
    ]
    
    try:
        # Basic structure validation
        for section in required_sections:
            if section not in extracted_text:
                logger.error(f"Missing required section: {section}")
                return False, None
        
        # TODO: Add more detailed validation if needed
        # For now, we just ensure the basic structure is present
        return True, None
        
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        return False, None

def load_model():
    """Load Tulu 3 8B in 4-bit quant mode and create pipeline."""
    try:
        logger.info(f"Loading model: {MODEL_ID} with 4-bit quantization...")
        start_time = perf_counter()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **TOKENIZER_CONFIG)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            max_memory={0: "10GiB"},
            offload_folder="offload"
        )

        # Create pipeline once during initialization
        pipe = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **GENERATION_CONFIG
        )

        load_time = perf_counter() - start_time
        logger.info(f"Model and pipeline loaded successfully in {load_time:.2f}s")
        return tokenizer, model, pipe
    
    except Exception as e:
        logger.error(f"Failed to load Tulu model: {str(e)}")
        raise

def run_inference(prompt: str, pipe, validate_full_structure: bool = False) -> str:
    """Single-step inference using pre-initialized pipeline."""
    logger.info("Starting inference...")
    start_time = perf_counter()

    try:
        outputs = pipe(prompt)
        result = outputs[0]["generated_text"].strip()
        
        # Only validate full structure when explicitly requested
        if validate_full_structure:
            is_valid, _ = validate_extraction(result)
            if not is_valid:
                logger.warning("Full structure validation failed - output may be incomplete")
        
        inference_time = perf_counter() - start_time
        logger.info(f"Inference complete in {inference_time:.2f}s")
        return result

    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise

def save_outputs(extracted_info: str, data: dict, prompt: str):
    """Save extraction output & debug info."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save final extraction
    output_file = OUTPUT_DIR / f"tulu_extraction_{timestamp}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(extracted_info)
    print(f"Extraction saved to: {output_file}")

    # Save debug JSON
    debug_file = OUTPUT_DIR / f"tulu_debug_{timestamp}.json"
    debug_data = {
        "input": {
            "title": data.get("title"),
            "url": data.get("url"),
            "content_length": len(data.get("content", "")),
        },
        "prompt": prompt,
        "output": extracted_info
    }
    with open(debug_file, 'w', encoding='utf-8') as f:
        json.dump(debug_data, f, indent=2)
    print(f"Debug info saved to: {debug_file}")

@dataclass
class SectionData:
    """Data structure for storing identified sections"""
    raw_text: str
    start_idx: int
    end_idx: int

# Add debugging utilities
console = Console()

def debug_output(func):
    """Decorator to log function inputs/outputs when DEBUG mode is on"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if os.getenv('DEBUG', '0') == '1':
            # Create a panel for function entry
            console.print(Panel(
                f"[bold cyan]Entering {func.__name__}[/bold cyan]",
                title="Debug Info",
                border_style="blue"
            ))
            
            start = time.perf_counter()
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start
            
            # Create debug table
            table = Table(title=f"{func.__name__} Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("Duration", f"{duration:.2f}s")
            table.add_row("Output Type", str(type(result)))
            table.add_row("Output Size", str(len(str(result))))
            
            # Wrap table in a panel
            console.print(Panel(
                table,
                title="Function Results",
                border_style="green"
            ))
            
            return result
        return func(*args, **kwargs)
    return wrapper

class PipelineVisualizer:
    """Visualize pipeline progress and results"""
    def __init__(self):
        self.console = Console()
        self.steps = []
    
    def add_step(self, step_name: str, status: str, duration: float, details: str = ""):
        self.steps.append({
            "name": step_name,
            "status": status,
            "duration": duration,
            "details": details
        })
    
    def show(self):
        table = Table(title="Pipeline Execution Summary")
        table.add_column("Step", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Duration", style="yellow")
        table.add_column("Details", style="white")
        
        for step in self.steps:
            table.add_row(
                step["name"],
                step["status"],
                f"{step['duration']:.2f}s",
                step["details"]
            )
        
        self.console.print(table)

class SectionIdentifierAgent:
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".StructureIdentifier")
        
    def identify_sections(self, content: str) -> Dict[str, SectionData]:
        """Identify sections using adaptive patterns and hierarchical analysis"""
        self.logger.info("Starting section identification")
        
        # Hierarchical patterns for better section detection
        section_patterns = {
            "PLANS": [
                # Primary patterns
                r"(?i)(?:pricing|plans?|tiers?|packages?)\s*(?:table|options?|details?).*?(?=\n\n|\Z)",
                # Plan-based patterns
                r"(?i)(?:Free|Basic|Starter|Pro|Business|Enterprise).*?(?:\$|\d+/\w+).*?(?=\n\n|\Z)",
                # Price-based patterns
                r"(?i)(?:\$\d+|\d+\s*USD).*?(?:month|year|annual).*?(?=\n\n|\Z)",
                # Feature list patterns
                r"(?i)(?:✓|\*|•|-)\s*(?:unlimited|users?|storage).*?(?=\n\n|\Z)"
            ],
            "FEATURES": [
                # Primary patterns
                r"(?i)(?:features?|capabilities|what.*?included).*?(?=\n\n|\Z)",
                # List-based patterns
                r"(?i)(?:✓|\*|•|-)\s*(?:features?|includes?).*?(?=\n\n|\Z)",
                # Category-based patterns
                r"(?i)(?:core|essential|advanced|premium)\s*features?.*?(?=\n\n|\Z)"
            ]
        }
        
        sections = {}
        content_blocks = self._split_content_blocks(content)
        
        for block in content_blocks:
            section_type = self._identify_block_type(block, section_patterns)
            if section_type:
                if section_type not in sections:
                    sections[section_type] = []
                sections[section_type].append(block)
        
        # Merge related blocks and create final sections
        return self._merge_section_blocks(sections, content)
    
    def _split_content_blocks(self, content: str) -> List[str]:
        """Split content into logical blocks based on spacing and structure"""
        # Split on double newlines first
        blocks = [b.strip() for b in content.split('\n\n') if b.strip()]
        
        # Further split large blocks if they contain distinct sections
        refined_blocks = []
        for block in blocks:
            if len(block.split('\n')) > 10:  # Large block
                sub_blocks = self._split_large_block(block)
                refined_blocks.extend(sub_blocks)
            else:
                refined_blocks.append(block)
        
        return refined_blocks
    
    def _identify_block_type(self, block: str, patterns: Dict[str, List[str]]) -> Optional[str]:
        """Identify the type of a content block using patterns"""
        for section_type, section_patterns in patterns.items():
            for pattern in section_patterns:
                if re.search(pattern, block, re.IGNORECASE | re.MULTILINE):
                    return section_type
        return None
    
    def _merge_section_blocks(self, section_blocks: Dict[str, List[str]], original_content: str) -> Dict[str, SectionData]:
        """Merge related blocks into cohesive sections"""
        sections = {}
        
        for section_type, blocks in section_blocks.items():
            if not blocks:
                continue
                
            # Find the start and end positions in original content
            first_block = blocks[0]
            last_block = blocks[-1]
            start_idx = original_content.find(first_block)
            end_idx = original_content.find(last_block) + len(last_block)
            
            # Get the full text including content between blocks
            section_text = original_content[start_idx:end_idx]
            
            sections[section_type] = SectionData(
                raw_text=section_text,
                start_idx=start_idx,
                end_idx=end_idx
            )
            
            self.logger.info(f"Created {section_type} section with {len(section_text)} chars")
            self.logger.debug(f"Section preview: {section_text[:100]}...")
        
        return sections
    
    def _split_large_block(self, block: str) -> List[str]:
        """Split large content blocks into smaller logical sections"""
        # Split on common section indicators
        section_markers = [
            r'\n#{2,}',           # Markdown headers
            r'\n\*{2,}',          # Asterisk dividers
            r'\n-{2,}',           # Dash dividers
            r'\n\s*\d+\.',        # Numbered lists
            r'\n\s*[A-Z][^.!?]+:' # Capitalized labels with colons
        ]
        
        blocks = [block]
        for marker in section_markers:
            new_blocks = []
            for b in blocks:
                splits = re.split(marker, b)
                splits = [s.strip() for s in splits if s.strip()]
                new_blocks.extend(splits)
            blocks = new_blocks
        
        # Further split very large blocks by double newlines
        if any(len(b) > 1000 for b in blocks):
            new_blocks = []
            for b in blocks:
                if len(b) > 1000:
                    sub_splits = [s.strip() for s in b.split('\n\n') if s.strip()]
                    new_blocks.extend(sub_splits)
                else:
                    new_blocks.append(b)
            blocks = new_blocks
        
        return blocks

class PlanExtractorAgent:
    """Agent 2: Extracts specific plan details"""
    
    def __init__(self, model, pipe):
        self.model = model
        self.pipe = pipe
        self.logger = logging.getLogger(__name__ + ".PlanExtractor")
    
    def extract_plans(self, plans_section: SectionData) -> Dict[str, Any]:
        """Extract plans using Chain-of-Thought reasoning"""
        if not plans_section.raw_text.strip():
            self.logger.warning("No plan section content provided")
            return {}
            
        # Step 1: Identify plan names
        plan_names = self._extract_plan_names(plans_section.raw_text)
        if not plan_names:
            self.logger.error("No plan names found")
            return {}
            
        # Step 2: Extract details for each plan
        plans_data = {}
        for plan_name in plan_names:
            plan_data = self._extract_plan_details(plan_name, plans_section.raw_text)
            if plan_data:
                plans_data[plan_name] = plan_data
        
        return plans_data
    
    def _extract_plan_names(self, content: str) -> List[str]:
        """Extract plan names using focused prompt"""
        prompt = f"""Analyze this pricing page content and list ONLY the plan names.
Common plan names include: Free, Basic, Starter, Pro, Business, Enterprise.

Content to analyze:
{content}

Return ONLY a JSON array of plan names, like this:
["Free", "Pro", "Enterprise"]"""

        # Skip full structure validation for this focused extraction
        result = run_inference(prompt, self.pipe, validate_full_structure=False)
        try:
            json_match = re.search(r'\[.*\]', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            self.logger.error("Failed to extract plan names")
        return []
    
    def _extract_plan_details(self, plan_name: str, content: str) -> Optional[Dict]:
        """Extract details for a specific plan"""
        prompt = f"""Focus ONLY on the "{plan_name}" plan and extract its details.

Guidelines:
1. Find the exact price and billing period
2. Look for features specifically mentioned for this plan
3. Note any limits or restrictions
4. Identify if this is marked as popular/recommended

Content to analyze:
{content}

Return ONLY a JSON object with these fields:
{{
    "price": <number>,
    "period": "monthly/annual",
    "description": "exact text",
    "is_popular": true/false,
    "features": ["feature 1", "feature 2"],
    "limits": {{
        "users": "exact limit",
        "storage": "exact limit",
        "api": "exact limit"
    }}
}}"""

        # Skip full structure validation for plan details
        result = run_inference(prompt, self.pipe, validate_full_structure=False)
        try:
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            self.logger.error(f"Failed to extract details for {plan_name}")
        return None
    
    def _extract_price(self, price_value: Any) -> float:
        """Extract numeric price from various formats"""
        if isinstance(price_value, (int, float)):
            return float(price_value)
        if isinstance(price_value, str):
            match = re.search(r'(\d+(?:\.\d{2})?)', price_value)
            if match:
                return float(match.group(1))
        return 0.0

class FeatureExtractorAgent:
    """Agent for extracting and categorizing features"""
    
    def __init__(self, model, pipe):
        self.model = model
        self.pipe = pipe
        self.logger = logging.getLogger(__name__ + ".FeatureExtractor")
    
    def extract_features(self, features_section: SectionData) -> Dict[str, Any]:
        """Extract features using multi-step approach"""
        self.logger.info("Starting feature extraction")
        
        if not features_section.raw_text.strip():
            self.logger.warning("No features section content provided")
            return {}
            
        # Step 1: Extract core product description
        description = self._extract_description(features_section.raw_text)
        
        # Step 2: Identify feature categories
        categories = self._identify_categories(features_section.raw_text)
        
        # Step 3: Extract features for each category
        feature_categories = {}
        for category in categories:
            features = self._extract_category_features(category, features_section.raw_text)
            if features:
                feature_categories[category] = features
        
        # Step 4: Extract limitations
        limitations = self._extract_limitations(features_section.raw_text)
        
        # Step 5: Extract add-ons
        addons = self._extract_addons(features_section.raw_text)
        
        return {
            "description": description,
            "feature_categories": feature_categories,
            "limitations": limitations,
            "addons": addons
        }
    
    def _extract_description(self, content: str) -> str:
        """Extract product description"""
        prompt = """Extract the main product description from this text.
Focus on the core purpose and value proposition.
Return ONLY the exact description text found, no other text."""
        
        # Skip full structure validation for description extraction
        result = run_inference(prompt + f"\n\nContent:\n{content}", 
                             self.pipe, validate_full_structure=False)
        return result.strip() or "Not specified"
    
    def _identify_categories(self, content: str) -> List[str]:
        """Identify feature categories"""
        prompt = """Identify all feature categories mentioned in this text.
Common categories include:
- Core Features
- Integration Capabilities
- Security & Compliance
- Collaboration Features
- Analytics & Reporting
- Support & Training

Return ONLY a JSON array of category names found in the text."""
        
        result = run_inference(prompt + f"\n\nContent:\n{content}", self.pipe)
        try:
            json_match = re.search(r'\[.*\]', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            self.logger.error("Failed to parse categories")
        return ["Core Features"]  # Fallback
    
    def _extract_category_features(self, category: str, content: str) -> List[str]:
        """Extract features for a specific category"""
        prompt = f"""Focus ONLY on {category} features in this text.
Extract each feature exactly as written.
Include ONLY features that clearly belong to this category.
Return a JSON array of feature texts."""
        
        result = run_inference(prompt + f"\n\nContent:\n{content}", self.pipe)
        try:
            json_match = re.search(r'\[.*\]', result, re.DOTALL)
            if json_match:
                features = json.loads(json_match.group())
                return [f.strip() for f in features if f.strip()]
        except:
            self.logger.error(f"Failed to parse features for {category}")
        return []
    
    def _extract_limitations(self, content: str) -> Dict[str, Any]:
        """Extract all limitations and restrictions"""
        prompt = """Extract ALL limitations and restrictions from this text.
Focus on:
1. API limits
2. Storage limits
3. User limits
4. Other restrictions

Return as JSON with these exact fields:
{
    "api_limits": "exact text",
    "storage_limits": "exact text",
    "user_limits": "exact text",
    "other_limits": ["limitation 1", "limitation 2"]
}"""
        
        result = run_inference(prompt + f"\n\nContent:\n{content}", self.pipe)
        try:
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            self.logger.error("Failed to parse limitations")
        return {
            "api_limits": "Not specified",
            "storage_limits": "Not specified",
            "user_limits": "Not specified",
            "other_limits": []
        }
    
    def _extract_addons(self, content: str) -> List[Dict[str, Any]]:
        """Extract all add-ons and their details"""
        prompt = """Extract ALL add-ons or extra features that cost additional money.
For each add-on, find:
1. Exact name
2. Exact price
3. Description
4. Features included

Return as JSON array of add-ons with these fields:
[{
    "name": "exact name",
    "price": "exact price text",
    "description": "exact description",
    "features": ["feature 1", "feature 2"]
}]"""
        
        result = run_inference(prompt + f"\n\nContent:\n{content}", self.pipe)
        try:
            json_match = re.search(r'\[.*\]', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            self.logger.error("Failed to parse add-ons")
        return []

class FinalFormatterAgent:
    """Agent 4: Formats and validates the final output"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".FinalFormatter")
    
    def format_output(self, plans_data: Dict, features_data: Dict) -> Dict[str, Any]:
        """Merge and format all extracted data into the required structure"""
        self.logger.info("Formatting final output")
        
        output = {
            "SERVICE_INFORMATION": {
                "Name": "Zapier",
                "URL": "https://zapier.com",
                "Logo_URL": "Not specified",
                "Description": self._extract_description(features_data),
                "Summary": "Not specified",
                "Category": {
                    "Primary": "Automation",
                    "Secondary": "Integration Platform",
                    "Tags": ["automation", "workflow", "integration", "saas"]
                },
                "Business_Model": {
                    "Type": "subscription",
                    "Description": "Per-seat subscription with usage-based tiers"
                }
            },
            "PRICING_METADATA": {
                "Currency": self._extract_currency(plans_data),
                "Regions": "Global",
                "Billing_Cycles": {
                    "Available": ["monthly", "annual"],
                    "Default": "annual"
                },
                "Custom_Pricing_Available": "Yes" if self._has_enterprise_plan(plans_data) else "No",
                "Free_Tier_Available": "Yes" if self._has_free_plan(plans_data) else "No"
            },
            "PLANS": self._format_plans(plans_data),
            "ADD_ONS": self._format_addons(features_data),
            "ENTERPRISE": self._format_enterprise(plans_data),
            "DISCOUNTS": self._format_discounts(plans_data)
        }
        
        return output
    
    def _format_plans(self, plans_data: Dict) -> Dict[str, Any]:
        """Format plans according to the required structure"""
        formatted_plans = {}
        
        for plan_name, plan_data in plans_data.items():
            formatted_plans[plan_name] = {
                "Name": plan_name,
                "Slug": plan_name.lower().replace(" ", "-"),
                "Description": plan_data.get("description", "Not specified"),
                "Is_Popular": plan_data.get("is_popular", "Not specified"),
                "Highlight_Features": plan_data.get("features", []),
                "Pricing": {
                    "Base": {
                        "Amount": plan_data.get("price", 0),
                        "Period": plan_data.get("period", "monthly"),
                        "Currency": plan_data.get("currency", "USD"),
                        "Is_Per_User": plan_data.get("is_per_user", True)
                    },
                    "Usage_Based_Components": []  # Add if present in data
                },
                "Limits": {
                    "Users": self._extract_user_limits(plan_data),
                    "Storage": self._extract_storage_limits(plan_data),
                    "API": self._extract_api_limits(plan_data),
                    "Compute": "Not specified",
                    "Other_Limits": []
                },
                "Features": plan_data.get("features", [])
            }
        
        return formatted_plans
    
    def _format_addons(self, features_data: Dict) -> List[Dict[str, Any]]:
        """Format add-ons section"""
        addons = []
        if "addons" in features_data:
            for addon in features_data["addons"]:
                addons.append({
                    "Name": addon.get("name", "Not specified"),
                    "Price": addon.get("price", "Not specified"),
                    "Description": addon.get("description", "Not specified"),
                    "Features": addon.get("features", []),
                    "Availability": addon.get("availability", "Not specified"),
                    "Requirements": addon.get("requirements", "Not specified"),
                    "Limitations": addon.get("limitations", "Not specified")
                })
        return addons
    
    def _format_enterprise(self, plans_data: Dict) -> Dict[str, Any]:
        """Format enterprise section"""
        return {
            "Available": "Yes" if self._has_enterprise_plan(plans_data) else "No",
            "Contact_Sales_Required": "Yes",
            "Minimum_Seats": "Not specified",
            "Custom_Features": []
        }
    
    def _format_discounts(self, plans_data: Dict) -> List[Dict[str, Any]]:
        """Format discounts section"""
        discounts = []
        if "discounts" in plans_data:
            for discount in plans_data["discounts"]:
                discounts.append({
                    "Type": discount.get("type", "Not specified"),
                    "Amount": discount.get("amount", "Not specified"),
                    "Description": discount.get("description", "Not specified"),
                    "Conditions": discount.get("conditions", "Not specified"),
                    "Valid_Until": discount.get("valid_until", "Not specified")
                })
        return discounts
    
    def _extract_user_limits(self, plan_data: Dict) -> str:
        """Extract user limits from plan data"""
        if "limits" in plan_data and "users" in plan_data["limits"]:
            return plan_data["limits"]["users"]
        return "Not specified"
    
    def _extract_storage_limits(self, plan_data: Dict) -> str:
        """Extract storage limits from plan data"""
        if "limits" in plan_data and "storage" in plan_data["limits"]:
            return plan_data["limits"]["storage"]
        return "Not specified"
    
    def _extract_api_limits(self, plan_data: Dict) -> str:
        """Extract API limits from plan data"""
        if "limits" in plan_data and "api" in plan_data["limits"]:
            return plan_data["limits"]["api"]
        return "Not specified"
    
    def _has_enterprise_plan(self, plans_data: Dict) -> bool:
        """Check if enterprise plan exists"""
        return any(p.lower() in ["enterprise", "business", "custom"] 
                  for p in plans_data.keys())
    
    def _has_free_plan(self, plans_data: Dict) -> bool:
        """Check if free plan exists"""
        return any(p.lower() == "free" for p in plans_data.keys())
    
    def _extract_currency(self, plans_data: Dict) -> str:
        """Extract currency from plans data"""
        for plan in plans_data.values():
            if "currency" in plan:
                return plan["currency"]
        return "USD"
    
    def _extract_description(self, features_data: Dict) -> str:
        """Extract service description from features data"""
        if "description" in features_data:
            return features_data["description"]
        return "Not specified"

def load_json_data(input_path: str) -> dict:
    """Load and validate JSON data from SaaS pricing pages"""
    logger.info(f"Loading JSON data from {input_path}")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_json = json.load(f)
            
        data = raw_json.get('data', {})
        if not data or not data.get('content'):
            raise ValueError("Invalid JSON structure: missing data or content")
            
        return {
            "title": data.get("title", "Not specified"),
            "url": data.get("url", "Not specified"),
            "description": data.get("description", ""),
            "content": data.get("content", ""),
            "usage": data.get("usage", {})  # Track token usage if needed
        }
    except Exception as e:
        logger.error(f"Failed to load JSON: {str(e)}")
        raise

class ValidationAgent:
    """Agent for validating extraction results with context tracking"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".Validator")
        self.context = {}  # Track context between validations
    
    def validate_sections(self, sections: Dict[str, SectionData]) -> bool:
        """Validate sections with content analysis"""
        required_sections = ["PLANS", "FEATURES"]
        found_sections = sections.keys()
        
        # Track section sizes for context
        self.context["section_sizes"] = {
            name: len(section.raw_text) 
            for name, section in sections.items()
        }
        
        # Check for minimum content size
        min_content_size = 100  # Characters
        for section in required_sections:
            if section not in found_sections:
                self.logger.error(f"Missing required section: {section}")
                return False
            
            content = sections[section].raw_text.strip()
            if not content:
                self.logger.error(f"Empty content in section: {section}")
                return False
                
            if len(content) < min_content_size:
                self.logger.warning(f"Section {section} may be too small: {len(content)} chars")
                
        # Look for pricing indicators in PLANS section
        plans_content = sections.get("PLANS", SectionData("", 0, 0)).raw_text
        if not any(indicator in plans_content.lower() for indicator in ['$', 'usd', 'free', 'price']):
            self.logger.warning("No pricing indicators found in PLANS section")
            
        return True
    
    def validate_plan_data(self, plans_data: Dict) -> bool:
        """Validate plan data with detailed checks"""
        if not plans_data:
            self.logger.error("No plans extracted")
            return False
            
        required_fields = {
            "price": (lambda x: isinstance(x, (int, float, str))),
            "period": (lambda x: isinstance(x, str) and x in ["monthly", "annual", "yearly"]),
            "features": (lambda x: isinstance(x, list) and len(x) > 0),
            "description": (lambda x: isinstance(x, str) and len(x) > 10)
        }
        
        valid_plans = []
        for plan_name, plan in plans_data.items():
            plan_issues = []
            
            # Check required fields
            for field, validator in required_fields.items():
                value = plan.get(field)
                if not value:
                    plan_issues.append(f"Missing {field}")
                elif not validator(value):
                    plan_issues.append(f"Invalid {field} format")
            
            # Validate price format and value
            price = plan.get("price")
            if isinstance(price, str):
                try:
                    # Extract numeric value from price string
                    price_match = re.search(r'\d+(?:\.\d{2})?', price)
                    if price_match:
                        price = float(price_match.group())
                    else:
                        plan_issues.append("Cannot parse price value")
                except ValueError:
                    plan_issues.append("Invalid price format")
            
            # Track validation results
            if not plan_issues:
                valid_plans.append(plan_name)
            else:
                self.logger.warning(f"Plan '{plan_name}' issues: {', '.join(plan_issues)}")
        
        # Update context
        self.context["valid_plans"] = valid_plans
        self.context["total_plans"] = len(plans_data)
        
        return len(valid_plans) > 0
    
    def validate_features_data(self, features_data: Dict) -> bool:
        """Validate features with category analysis"""
        if not features_data:
            self.logger.error("No features extracted")
            return False
            
        if "feature_categories" not in features_data:
            self.logger.error("Missing feature categories")
            return False
            
        categories = features_data.get("feature_categories", {})
        valid_categories = []
        total_features = 0
        
        for category, features in categories.items():
            if not isinstance(features, list):
                self.logger.error(f"Invalid features format in category: {category}")
                continue
                
            # Clean and validate features
            valid_features = [
                f.strip() for f in features 
                if isinstance(f, str) and len(f.strip()) > 3
            ]
            
            if valid_features:
                valid_categories.append(category)
                total_features += len(valid_features)
                
        # Update context
        self.context["valid_categories"] = valid_categories
        self.context["total_features"] = total_features
        
        # Check for reasonable feature count
        min_features = 3  # Minimum features per category
        if total_features < min_features * len(valid_categories):
            self.logger.warning("Low feature count detected")
            
        return len(valid_categories) > 0
    
    def validate_final_output(self, output: Dict) -> bool:
        """Validate final output with context awareness"""
        required_sections = [
            "SERVICE_INFORMATION",
            "PRICING_METADATA",
            "PLANS"
        ]
        
        # Check required sections
        missing_sections = [
            section for section in required_sections 
            if section not in output
        ]
        if missing_sections:
            self.logger.error(f"Missing sections in final output: {missing_sections}")
            return False
        
        # Validate against context
        plans_count = len(output.get("PLANS", {}))
        if plans_count != self.context.get("total_plans", 0):
            self.logger.warning(f"Plan count mismatch: {plans_count} vs {self.context.get('total_plans')}")
        
        # Check for data consistency
        if "PLANS" in output and output["PLANS"]:
            for plan_name, plan_data in output["PLANS"].items():
                if not all(k in plan_data for k in ["Pricing", "Features", "Limits"]):
                    self.logger.error(f"Incomplete plan data for {plan_name}")
                    return False
        else:
            self.logger.error("No plans in final output")
            return False
            
        return True
    
    def get_validation_summary(self) -> Dict:
        """Return summary of validation results"""
        return {
            "sections_found": list(self.context.get("section_sizes", {}).keys()),
            "valid_plans": self.context.get("valid_plans", []),
            "valid_categories": self.context.get("valid_categories", []),
            "total_features": self.context.get("total_features", 0)
        }

def run_pipeline(content: str, model, pipe) -> Dict[str, Any]:
    """Run extraction pipeline with validation"""
    logger.info("Starting extraction pipeline")
    start_time = perf_counter()
    visualizer = PipelineVisualizer()
    validator = ValidationAgent()
    
    try:
        # Initialize agents
        agents_start = perf_counter()
        structure_agent = SectionIdentifierAgent()
        plan_agent = PlanExtractorAgent(model, pipe)
        feature_agent = FeatureExtractorAgent(model, pipe)
        formatter_agent = FinalFormatterAgent()
        visualizer.add_step("Init", "Success", perf_counter() - agents_start)
        
        # Step 1: Clean and prepare content
        clean_start = perf_counter()
        cleaned_content = content.replace('\u2028', '\n').replace('\u2029', '\n')
        visualizer.add_step("Content Preparation", "Success", perf_counter() - clean_start)
        
        # Step 2: Identify and validate sections
        sections = structure_agent.identify_sections(cleaned_content)
        sections_valid = validator.validate_sections(sections)
        visualizer.add_step(
            "Section Identification",
            "Success" if sections_valid else "Partial",
            perf_counter() - clean_start,
            f"Found {len(sections)} sections"
        )
        
        # Step 3: Extract and validate plans
        plans_start = perf_counter()
        plans_data = {}
        plans_valid = False
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                plans_data = plan_agent.extract_plans(sections.get("PLANS", SectionData("", 0, 0)))
                plans_valid = validator.validate_plan_data(plans_data)
                if plans_valid:
                    break
                logger.warning(f"Plan validation failed on attempt {attempt + 1}")
            except Exception as e:
                logger.error(f"Plan extraction failed on attempt {attempt + 1}: {str(e)}")
        
        visualizer.add_step(
            "Plan Extraction",
            "Success" if plans_valid else "Failed",
            perf_counter() - plans_start,
            f"Found {len(plans_data)} valid plans"
        )
        
        # Step 4: Extract and validate features
        features_start = perf_counter()
        features_data = {}
        features_valid = False
        
        for attempt in range(max_retries):
            try:
                features_data = feature_agent.extract_features(sections.get("FEATURES", SectionData("", 0, 0)))
                features_valid = validator.validate_features_data(features_data)
                if features_valid:
                    break
                logger.warning(f"Feature validation failed on attempt {attempt + 1}")
            except Exception as e:
                logger.error(f"Feature extraction failed on attempt {attempt + 1}: {str(e)}")
        
        visualizer.add_step(
            "Feature Extraction",
            "Success" if features_valid else "Failed",
            perf_counter() - features_start,
            f"Features valid: {features_valid}"
        )
        
        # Final formatting with full structure validation
        format_start = perf_counter()
        formatted_prompt = f"""Based on the extracted information, format the complete output.
Use this exact structure and include all sections.
If information is missing, use "Not specified".

{create_prompt({"content": ""})}  # Use the base prompt structure

Extracted information to format:
Plans: {json.dumps(plans_data, indent=2)}
Features: {json.dumps(features_data, indent=2)}
"""
        
        final_output = run_inference(formatted_prompt, pipe, validate_full_structure=True)
        output_valid = validator.validate_final_output(final_output)
        
        visualizer.add_step(
            "Final Formatting",
            "Success" if output_valid else "Partial",
            perf_counter() - format_start,
            f"Output valid: {output_valid}"
        )
        
        # Show pipeline summary
        pipeline_time = perf_counter() - start_time
        logger.info(f"Pipeline completed in {pipeline_time:.2f}s")
        visualizer.show()
        
        return final_output
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        visualizer.show()
        raise

def main():
    total_start = perf_counter()
    try:
        # Load JSON data
        data_start = perf_counter()
        data = load_json_data("rawContentExampleSchema.json")
        logger.info(f"Data loading took {perf_counter() - data_start:.2f}s")
        
        # Initialize model and pipeline
        model_start = perf_counter()
        tokenizer, model, pipe = load_model()
        logger.info(f"Model initialization took {perf_counter() - model_start:.2f}s")
        
        # Run extraction pipeline
        extracted_info = run_pipeline(data["content"], model, pipe)
        
        # Save outputs
        save_start = perf_counter()
        save_outputs(json.dumps(extracted_info, indent=2), data, "")
        logger.info(f"Saving outputs took {perf_counter() - save_start:.2f}s")
        
        total_time = perf_counter() - total_start
        logger.info(f"Total process took {total_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Process failed: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
