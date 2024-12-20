"""
Contains the format specification for pricing information extraction
"""

EXTRACTION_FORMAT = """STEP 1: Identify SERVICE INFORMATION
- Name: The exact name of the service/product.
- URL: If a pricing page URL is known or can be inferred from content, provide it.
- Logo URL: If available.
- Description: A brief summary of the service.
- Summary: A longer summary of the service.
- Category: [primary/secondary if applicable]
- Tags: Relevant keywords as comma-separated values.
- Business Model: One of [fixed, usage_based, tiered, hybrid] + brief description.

STEP 2: PRICING METADATA
- Currency: e.g., USD
- Available Regions: If specified, else 'Not specified'
- Billing Cycles: List all (e.g., monthly, annually)
- Default Billing: Which cycle is default?
- Custom Pricing: "Available" or "Not Available"
- Free Tier: "Available" or "Not Available"

STEP 3: PLANS
For each plan:
- Plan Name: exact name
- Slug: lowercase, no spaces
- Description: short summary
- Popular: Yes/No
- Base Price: numeric amount + currency + period (e.g., 20 USD per month)
- Per User: Yes/No
- Key Features: bullet list (qualitative features only)
- Limits: Only numeric or capacity-related constraints (e.g., users, storage, records) as numeric values + unit.

If something isn't mentioned, write 'Not specified'.

STEP 4: ENTERPRISE OFFERING
- Available: Yes/No
- Contact Sales Required: Yes/No
- Minimum Seats: numeric or 'Not specified'
- Custom Features: bullet list or 'Not specified'

STEP 5: DISCOUNTS (if any)
- Type
- Amount
- Description
- Conditions
- Valid Until
If none, write all as 'Not specified'.

STEP 6: USE CASES
For each identified use case:
- Use Case: Clear name of the use case scenario
  - Target User: Who would benefit most from this use case
  - Pain Points Solved: List specific problems this solves
  - Key Benefits: How the service addresses these problems
  - Recommended Plan: Which pricing tier best fits this use case
  - ROI Potential: Expected return on investment (High/Medium/Low) with justification

Consider these perspectives when analyzing use cases:
- Small business owners with limited technical knowledge
- Automation specialists looking to integrate multiple systems
- Department managers needing workflow automation
- Citizen developers building internal tools
- Data analysts requiring integration capabilities
- Operations teams managing complex workflows
- IT teams looking for governance and security
- Marketing teams needing campaign automation
- Sales teams requiring CRM integration
- HR teams managing employee processes

Focus on:
- How the service connects to other tools and APIs
- Automation capabilities and their business impact
- Data processing and transformation features
- Integration with popular business tools
- Security and compliance features for enterprise
- Scalability aspects for growing organizations
- Cost-effectiveness compared to manual processes
- Learning curve and technical requirements
- Support and training resources available
- Time-to-value for different use cases

ADDITIONAL REQUIREMENTS:
- Ensure product name is the core product name, not a solution name
- Include both brief Description and longer Summary fields
- Use the actual pricing page URL when available
- Be precise with pricing details including billing period information
- Maintain consistent formatting for features and limits
- Include all relevant add-ons and extensions
- Ensure proper categorization based on primary product function
- No commentary outside the required sections.
- Convert all numeric fields to numbers without extra units in the number itself. For example, "1000 records" → "Records: 1000".
- For booleans use Yes/No.
- If a field is not applicable or unknown, write 'Not specified'.
- End by double-checking that you followed all instructions.
Format limits consistently as:
  - Limits:
    - Records per base: [number]
    - Storage: [number] GB
    - Editors: [number]
    - Automation runs: [number]
    - [Other limit name]: [number] [unit]

Double-check:
1. All required sections and fields are present.
2. No extra commentary.
3. ALL numeric limits are captured under each plan's Limits section.
4. Numeric values are properly formatted without units in the number.

Your response must start with 'SERVICE INFORMATION' and contain only the extracted information."""