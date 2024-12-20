"""
Contains example input and output for the model
"""

EXAMPLE_INPUT = """
Acme Workflow Platform - The Ultimate Business Process Automation Solution

Plans & Pricing:
1. Free Starter
- $0/month forever
- Up to 5 team members
- 5GB storage
- 100 automation runs/month
- Basic email support
- Standard integrations (Gmail, Slack)

2. Business Pro
- $29/user/month (billed annually) or $35/user/month (billed monthly)
- Unlimited team members
- 100GB storage per workspace
- 10,000 automation runs/month
- Priority support via chat/email
- Advanced integrations (Salesforce, SAP)
- Custom workflows
- API access
- SAML SSO

3. Enterprise
- Custom pricing (contact sales)
- Unlimited storage
- Unlimited automation runs
- Dedicated support team
- Custom integrations
- Advanced security features
- Audit logs
- Custom contract required
- Minimum 50 seats

Special Offers:
- Nonprofit organizations: 50% off any plan
- Educational institutions: Free Business Pro for up to 50 users
- Startup Program: 90% off for first year (must be VC funded)

Available worldwide with data centers in US, EU, and APAC regions.
"""

EXAMPLE_OUTPUT = """SERVICE INFORMATION
- Name: Acme Workflow Platform
- URL: Not specified
- Logo URL: Not specified
- Description: A business process automation platform with comprehensive workflow capabilities.
- Summary: Enterprise-grade workflow automation platform that enables organizations to automate business processes, integrate with popular services, and scale operations efficiently. Features include custom workflows, extensive integrations, and robust security controls.
- Category: Business Process Automation
- Tags: workflow automation, business process, integrations, enterprise software, automation platform
- Business Model: hybrid (combines free tier with user-based pricing and enterprise custom pricing)

PRICING METADATA
- Currency: USD
- Available Regions: US, EU, APAC
- Billing Cycles: monthly, annually
- Default Billing: annually
- Custom Pricing: Available
- Free Tier: Available

PLANS
- Plan Name: Free Starter
  - Slug: free-starter
  - Description: Basic plan for small teams to start with automation
  - Popular: No
  - Base Price: 0 USD per month
  - Per User: No
  - Key Features:
    - Basic email support
    - Standard integrations
    - Gmail integration
    - Slack integration
  - Limits:
    - Team members: 5
    - Storage: 5 GB
    - Automation runs: 100 per month

- Plan Name: Business Pro
  - Slug: business-pro
  - Description: Comprehensive solution for growing businesses
  - Popular: Yes
  - Base Price: 29 USD per user per month
  - Per User: Yes
  - Key Features:
    - Priority support
    - Advanced integrations
    - Salesforce integration
    - SAP integration
    - Custom workflows
    - API access
    - SAML SSO
  - Limits:
    - Team members: Unlimited
    - Storage: 100 GB
    - Automation runs: 10,000 per month

- Plan Name: Enterprise
  - Slug: enterprise
  - Description: Customizable solution for large organizations
  - Popular: No
  - Base Price: Contact Sales
  - Per User: Yes
  - Key Features:
    - Dedicated support team
    - Custom integrations
    - Advanced security features
    - Audit logs
    - Custom contract options
  - Limits:
    - Team members: Unlimited
    - Storage: Unlimited
    - Automation runs: Unlimited

ENTERPRISE OFFERING
- Available: Yes
- Contact Sales Required: Yes
- Minimum Seats: 50
- Custom Features:
  - Custom integrations
  - Advanced security features
  - Audit logs
  - Custom contract terms
  - Dedicated support team

DISCOUNTS
- Type: Multiple programs
- Amount: Various (50-90% off)
- Description: Special pricing for qualified organizations
- Conditions:
  - Nonprofit: 50% off any plan
  - Education: Free Business Pro (up to 50 users)
  - Startups: 90% off first year (VC funding required)
- Valid Until: Not specified

USE CASES
- Use Case: Small Business Process Automation
  - Target User: Small business owners and teams
  - Pain Points Solved:
    - Manual repetitive tasks consuming valuable time
    - Limited budget for automation tools
    - Need for basic integrations with common tools
  - Key Benefits:
    - Free tier to start automation journey
    - Easy integration with common business tools
    - No technical expertise required
  - Recommended Plan: Free Starter
  - ROI Potential: Medium-High for teams spending 5+ hours/week on manual tasks

- Use Case: Mid-Market Business Operations
  - Target User: Operations managers and department heads
  - Pain Points Solved:
    - Complex workflows across departments
    - Need for advanced integrations
    - Security and compliance requirements
  - Key Benefits:
    - Custom workflow creation
    - Advanced integration capabilities
    - SAML SSO for security
  - Recommended Plan: Business Pro
  - ROI Potential: High for organizations with 10+ team members and multiple workflows

- Use Case: Enterprise Digital Transformation
  - Target User: Enterprise IT and Operations leaders
  - Pain Points Solved:
    - Large-scale automation needs
    - Complex security requirements
    - Custom integration requirements
    - Audit and compliance needs
  - Key Benefits:
    - Unlimited scalability
    - Custom security controls
    - Dedicated support
    - Audit capabilities
  - Recommended Plan: Enterprise
  - ROI Potential: Very High for organizations with 50+ users and complex workflows""" 