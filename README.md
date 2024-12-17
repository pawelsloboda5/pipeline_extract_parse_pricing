# **Pricing Pipeline Extraction, Parsing, and Saving to MongoDB**

## **Overview**
This document outlines the architecture and implementation of a simplified pipeline for extracting, parsing, validating, and saving pricing page URLs to MongoDB. The system is designed with modular components to enhance scalability, reliability, and ease of maintenance.

## **Architecture**
The pipeline consists of the following stages:

1. **Extraction**
   - **Primary Method**: Use Jina.ai to extract pricing page links efficiently.
   - **Fallback Method**: Selenium-based web scraping for reliability when Jina.ai encounters failures.

2. **Parsing**
   - **GPT Agents**: Use GPT agents to parse the extracted pricing pages.
     - Incorporate open-source models (e.g., HuggingFace models) whenever possible to reduce dependency on paid APIs.
   - Parsed data will feed into two primary pathways:
     1. **Frontend-Ready Collections**: Processed data prepared for frontend display.
     2. **ML-Ready Collections**: Data prepared for machine learning tasks such as vectorization and augmentation.

3. **Validation**
   - Validate and clean parsed data before saving it to MongoDB.
   - Ensure data integrity and consistency across different collections.

4. **Data Preparation and Saving**
   - Store validated data in MongoDB with **separate collections** based on the intended use:
     - **Frontend Collection**: For serving UI components.
     - **Machine Learning Collection**: Vectorized, split, and augmented data.
     - **Agent-Processed Collection**: Intermediate storage for additional agent-based transformations.

---

## **Pipeline Workflow**
### **1. Extraction**
#### **Primary: Jina.ai**
- Use Jina.ai's web crawling capabilities to extract URLs from pricing pages.
- Benefits:
  - Fast and scalable.
  - API-driven, reducing manual intervention.
- Integration: Implement API calls with Jina.ai for input websites and output clean links.

#### **Fallback: Selenium**
- Use Selenium for web crawling when Jina.ai fails.
- Automate browser navigation to locate and extract pricing page URLs.
- Error Handling:
  - Detect Jina.ai failures and fallback to Selenium scraping automatically.

---

### **2. Parsing**
#### **GPT Agents**
- Use GPT-based parsing agents to analyze and extract structured data from the raw HTML content of pricing pages.
- Preferred Models:
  - Open-source GPT models from **HuggingFace** or **other LLMs** to reduce cost.
- Features:
  - Data extraction (pricing tables, feature comparisons, plans, etc.).
  - Robust parsing with fallback to simple regex for critical elements.

#### **Data Flow**
Parsed data will be split into two pathways:
1. **Frontend-Ready Path**: Prepared for user-facing display with clean formatting.
2. **ML-Ready Path**: 
   - Data vectorization (e.g., embeddings via SentenceTransformers or OpenAI API).
   - Splitting and augmentation for downstream machine learning tasks.

---

### **3. Validation**
- **File Validation**: Check if extracted data meets formatting and content requirements.
- **Integrity Checks**:
  - Remove duplicates.
  - Validate pricing plans and key data fields.
- **Agent-Based Transformation**: If data fails validation, route it back to an agent for re-parsing or augmentation.

---

### **4. Saving to MongoDB**
- **Collections**:
   1. **Frontend-Ready Collection**:
      - Clean and formatted data optimized for UI components.
      - Example Fields: `plan_name`, `price`, `features`, `currency`, `updated_at`.
   2. **ML-Ready Collection**:
      - Vectorized and augmented data for machine learning models.
      - Example Fields: `embeddings`, `parsed_content`, `split_versions`, `augmentations`.
   3. **Agent-Processed Collection**:
      - Intermediate storage for data after agent-based transformations.
      - Example Fields: `raw_data`, `agent_version`, `error_logs`, `validated`.

- **Storage**:
  - Ensure efficient indexing for fast retrieval.
  - Use versioning to track updates to collections.

---

## **Future Improvements**
- Introduce a task scheduler (e.g., Celery, APScheduler) for automating pipeline runs.
- Add monitoring and logging for extraction success rates and agent failures.
- Expand support for multiple extraction tools to improve reliability.
- Integrate a feedback loop to re-train parsing models based on validation results.

---

## **Next Steps**
1. Implement the Jina.ai extraction module with a fallback to Selenium.
2. Build the parsing pipeline using GPT agents and open-source models.
3. Set up MongoDB collections for frontend and machine learning data.
4. Write unit tests for validation and storage components.
5. Document the initial implementation in the README.
