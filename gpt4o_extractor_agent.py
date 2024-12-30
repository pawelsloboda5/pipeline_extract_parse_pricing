import json
import os
import sys
from pathlib import Path
import openai
import time
from datetime import datetime
import logging
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler
from rich.progress import track
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union, Any

# Set up rich console
console = Console()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        RichHandler(rich_tracebacks=True),
        logging.FileHandler(f"extracted_info/debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger("gpt4o_extractor")

# ------------------------------------------
# Configure your OpenAI credentials
# ------------------------------------------
# Make sure you have set OPENAI_API_KEY in your env
openai.api_key = os.getenv("OPENAI_API_KEY")

# For usage tracking
MAX_TOKENS = 6000  # or whatever fits your $10 budget

# Define all schema classes using Pydantic to match schema.json exactly
class Category(BaseModel):
    primary: str
    secondary: str
    tags: List[str]

class BusinessModel(BaseModel):
    type: List[str]
    description: str

class Service(BaseModel):
    name: str
    url: str
    logo_url: Optional[str]
    description: Optional[str]
    category: Optional[Category]
    business_model: Optional[BusinessModel]

class BillingCycles(BaseModel):
    available: List[str]
    default: str

class Versioning(BaseModel):
    current: str
    history: List[str]

class PricingMetadata(BaseModel):
    last_updated: str  # ISO 8601 datetime string
    currency: str
    regions: List[str]
    billing_cycles: BillingCycles
    custom_pricing_available: bool
    free_tier_available: bool
    versioning: Versioning

class AmountRange(BaseModel):
    min: float
    max: float

class BasePricing(BaseModel):
    amount: Optional[float]
    amount_range: Optional[AmountRange]
    period: str
    currency: str
    is_per_user: bool

class TierRange(BaseModel):
    min: float
    max: float

class PricingTier(BaseModel):
    range: TierRange
    unit_price: float
    flat_fee: Optional[float]

class UsageBasedPricing(BaseModel):
    name: str
    type: str
    unit: str
    tiers: List[PricingTier]

class Pricing(BaseModel):
    base: BasePricing
    usage_based: List[UsageBasedPricing]

class UserLimits(BaseModel):
    min: float
    max: float

class StorageLimits(BaseModel):
    amount: float
    unit: str

class ApiRequests(BaseModel):
    rate: float
    period: str
    quota: Optional[float]

class ApiLimits(BaseModel):
    requests: ApiRequests

class ComputeLimits(BaseModel):
    vcpu: float
    memory: float
    unit: str

class OtherLimit(BaseModel):
    name: str
    value: str
    description: str

class Limits(BaseModel):
    users: Optional[UserLimits]
    storage: Optional[StorageLimits]
    api: Optional[ApiLimits]
    compute: Optional[ComputeLimits]
    other_limits: Optional[List[OtherLimit]]

class Feature(BaseModel):
    name: str
    description: str
    included: bool
    limit: str

class FeatureCategory(BaseModel):
    name: str
    features: List[Feature]

class Features(BaseModel):
    categories: List[FeatureCategory]

class Plan(BaseModel):
    name: str
    slug: str
    description: Optional[str]
    highlight_features: Optional[List[str]]
    is_popular: Optional[bool]
    pricing: Optional[Pricing]
    limits: Optional[Limits]
    features: Optional[Features]

class Discount(BaseModel):
    type: str
    amount: str
    description: str
    conditions: str
    valid_until: str  # ISO 8601 datetime string

class Enterprise(BaseModel):
    available: bool
    contact_sales: bool
    minimum_seats: Optional[float]
    custom_features: List[str]

class Embeddings(BaseModel):
    model: str
    version: str
    vectors: List[float]

class ConfidenceScores(BaseModel):
    pricing_accuracy: float
    feature_accuracy: float

class MLMetadata(BaseModel):
    embeddings: Embeddings
    confidence_scores: ConfidenceScores
    last_validated: str  # ISO 8601 datetime string

class AgentMetadata(BaseModel):
    agent_name: str
    agent_version: str
    execution_time_seconds: float
    fallback_used: bool
    comments: str

class AddOn(BaseModel):
    name: str
    description: str
    pricing: Optional[Dict[str, Union[float, str]]]

class UseCase(BaseModel):
    use_case: str
    target_user: str
    pain_points_solved: List[str]
    key_benefits: List[str]
    recommended_plan: str
    roi_potential: str

class PricingSchema(BaseModel):
    service: Service
    pricing_metadata: Optional[PricingMetadata]
    plans: List[Plan]
    add_ons: Optional[List[AddOn]]
    use_cases: Optional[List[UseCase]]
    discounts: Optional[List[Discount]]
    enterprise: Optional[Enterprise]
    ml_metadata: Optional[MLMetadata]
    agent_metadata: Optional[AgentMetadata]

def call_gpt4_for_extraction(content: str, page_title: str, page_url: str) -> tuple[PricingSchema, dict]:
    """Calls GPT-4 with structured output enforcement using Pydantic models"""
    client = openai.OpenAI()
    start_time = time.time()
    logger.info(f"Starting GPT-4 extraction for: {page_title} ({page_url})")
    
    system_prompt = """You are a pricing data extraction specialist. Extract detailed pricing information from SaaS websites into a structured format following the exact schema provided.

Key extraction rules:

1. Core Components:
   - Service details must be complete (name, description, categorization)
   - ALL pricing plans with EXACT pricing details (including ranges and conditions)
   - Version information should be specific (e.g., "v2024.1")
   - Add-ons must include pricing when available
   - ALL numerical limits must be captured
   - Enterprise information must ALWAYS be populated if available

2. Pricing & Limits:
   - Base price must include exact amount and billing period
   - Usage-based components must specify unit price and tiers
   - User limits (min/max) for EACH plan
   - Calendar/resource limits per plan
   - API rate limits if available
   - Storage quotas if applicable

3. Feature Extraction:
   - EVERY feature mentioned must be captured
   - Group features into logical categories
   - Mark features as included/not included
   - Include any usage limits or restrictions
   - Note if features require add-ons
   - Capture security and compliance features

4. Enterprise Details:
   - Always populate enterprise section if enterprise offering exists
   - Include all enterprise-specific features
   - Note security and compliance capabilities
   - Document support and SLA details
   - Capture custom/advanced features

5. Use Cases & Target Users:
   - Identify distinct use cases from plan descriptions and features
   - Map features to specific user types (e.g., individuals, teams, enterprises)
   - Extract pain points solved by each plan
   - Document key benefits for different user segments
   - Note ROI potential for each use case
   - Include industry-specific use cases if mentioned
   - Map recommended plans to specific user needs

6. Metadata & Documentation:
   - Exact version numbers
   - Geographic availability
   - Currency options
   - Implementation requirements
   - Integration limitations

7. Trial Information:
   - Document trial availability and duration
   - Note credit card requirements
   - List any trial restrictions
   - Include auto-renewal information
   - Capture trial-specific limitations

8. Enterprise Details:
   - Document minimum seats and contract length
   - List all security features separately
   - Capture support level details (type, response time, availability)
   - Note dedicated support availability
   - Include all custom enterprise features

9. Feature Details:
   - Include ALL feature limitations and restrictions
   - Document specific limits for each feature
   - Note feature availability across plans
   - Capture integration limitations
   - Include any feature-specific conditions

10. Add-on Details:
    - Extract EXACT pricing for all add-ons
    - Include per-user vs flat-rate pricing
    - Document pricing conditions and restrictions
    - Note plan availability requirements
    - Capture pricing ranges if variable
    - Include billing period information
    - Document any usage-based components
    - Note integration requirements

Be thorough and consistent. NEVER omit information that was previously captured. Ensure ALL numerical values and limits are captured exactly as shown in the pricing page."""

    try:
        logger.info("Making API call to GPT-4...")
        response = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract pricing data from this page:\nTitle: {page_title}\nURL: {page_url}\n\nContent:\n{content}"}
            ],
            response_format=PricingSchema
        )
        
        # Log token usage and extraction quality metrics
        usage = response.usage
        logger.info(f"Token usage - Prompt: {usage.prompt_tokens}, Completion: {usage.completion_tokens}, Total: {usage.total_tokens}")
        
        # Calculate and log cost
        cost = (usage.prompt_tokens * 0.01 + usage.completion_tokens * 0.03) / 1000
        logger.info(f"Estimated cost: ${cost:.4f}")
        
        # Validate and log plan details
        extracted = response.choices[0].message.parsed
        logger.info(f"Successfully extracted {len(extracted.plans)} pricing plans")
        
        # Validation logging
        if extracted.pricing_metadata:
            logger.info("Validating metadata formatting...")
            meta = extracted.pricing_metadata
            if meta.last_updated == "/" or not meta.last_updated:
                logger.warning("Invalid last_updated format in metadata")
            if meta.billing_cycles and not meta.billing_cycles.default:
                logger.warning("Empty billing_cycles.default in metadata")
        
        execution_time = time.time() - start_time
        logger.info(f"Extraction completed in {execution_time:.2f} seconds")
        
        return extracted, {
            "token_usage": usage.dict(),
            "execution_time": execution_time,
            "cost": cost,
            "quality_metrics": {
                "plans_with_limits": sum(1 for p in extracted.plans if p.limits is not None),
                "plans_with_features": sum(1 for p in extracted.plans if p.features is not None),
                "plans_with_popularity": sum(1 for p in extracted.plans if p.is_popular is not None),
                "total_plans": len(extracted.plans)
            }
        }

    except Exception as e:
        logger.error(f"GPT-4 extraction failed: {str(e)}", exc_info=True)
        return PricingSchema(service=Service(name="", url=""), plans=[]), {}

def create_extraction_summary(extracted_data: PricingSchema, metrics: dict) -> Table:
    """Create a rich table summarizing the extraction results"""
    table = Table(title="Extraction Summary")
    
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    # Add basic metrics
    table.add_row("Service Name", extracted_data.service.name)
    table.add_row("Number of Plans", str(len(extracted_data.plans)))
    table.add_row("Has Enterprise", str(extracted_data.enterprise is not None))
    table.add_row("Number of Discounts", str(len(extracted_data.discounts or [])))
    
    # Add quality metrics
    quality = metrics.get("quality_metrics", {})
    if quality:
        table.add_row("Plans with Limits", f"{quality['plans_with_limits']}/{quality['total_plans']}")
        table.add_row("Plans with Features", f"{quality['plans_with_features']}/{quality['total_plans']}")
        table.add_row("Plans with Popularity", f"{quality['plans_with_popularity']}/{quality['total_plans']}")
    
    # Add performance metrics
    table.add_row("Execution Time", f"{metrics['execution_time']:.2f}s")
    table.add_row("Total Tokens", str(metrics['token_usage']['total_tokens']))
    table.add_row("Estimated Cost", f"${metrics['cost']:.4f}")
    
    return table

def process_pricing_json(file_path: str) -> str:
    """Process pricing page content with schema validation"""
    logger.info(f"Processing file: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    content = data["data"].get("content", "")
    title = data["data"].get("title", "") 
    url = data["data"].get("url", "")

    # Extract with schema enforcement
    extracted, metrics = call_gpt4_for_extraction(content, title, url)

    # Generate timestamp filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    service_name = url.split('/')[2].split('.')[0]  # Extract service name from URL
    out_file = Path("extracted_info") / f"parsed_{service_name}_{timestamp}.json"
    debug_file = Path("extracted_info") / f"debug_metrics_{timestamp}.json"
    
    # Ensure extracted_info directory exists
    Path("extracted_info").mkdir(exist_ok=True)
    
    # Save extracted data
    with open(out_file, "w", encoding="utf-8") as out:
        json.dump(extracted.model_dump(), out, indent=2, default=str)

    # Save debug metrics
    debug_data = {
        "input_file": file_path,
        "timestamp": timestamp,
        "metrics": metrics,
        "extraction_summary": {
            "num_plans": len(extracted.plans),
            "has_enterprise": extracted.enterprise is not None,
            "num_discounts": len(extracted.discounts or []),
        }
    }
    
    with open(debug_file, "w", encoding="utf-8") as debug:
        json.dump(debug_data, debug, indent=2)

    # Display summary table
    console.print(create_extraction_summary(extracted, metrics))
    logger.info(f"Saved parsed data to: {out_file}")
    logger.info(f"Saved debug metrics to: {debug_file}")
    
    return str(out_file)

def main():
    console.print("[bold blue]Starting Pricing Data Extraction[/bold blue]")
    
    input_files = [
        "rawContentExampleSchema.json"
    ]

    for f in track(input_files, description="Processing files"):
        process_pricing_json(f)

    console.print("[bold green]Extraction Complete![/bold green]")

if __name__ == "__main__":
    main()
