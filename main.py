import os
import requests
from typing import Dict, List, Union
from datetime import datetime
from urllib.parse import urlparse
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import logging
from dotenv import load_dotenv
import json
from pathlib import Path
from content_processor import ContentProcessor
import time
import re
from pricing_agent import PricingAgent
from check_json import PricingSchemaValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

logging.getLogger('gpt4o_extractor').setLevel(logging.INFO)

class WebExtractor:
    """Base class for web extraction"""
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        return urlparse(url).netloc

    def is_valid_url(self, url: str) -> bool:
        """Basic URL validation"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False

class JinaExtractorContent(WebExtractor):
    """Jina.ai based content extraction"""
    def __init__(self):
        super().__init__()
        self.base_url = 'https://r.jina.ai'
        
    def extract_content(self, url: str) -> Dict[str, Union[str, str]]:
        """Extract content using Jina.ai"""
        try:
            logger.info(f"Making Jina.ai request for URL: {url}")
            response = requests.get(
                f'{self.base_url}/{url}',
                headers={
                    **self.headers,
                    'X-Return-Format': 'markdown',
                    'X-Retain-Images': 'none',
                    'X-With-Iframe': 'true',
                    'X-With-Shadow-Dom': 'true',
                    'Accept': 'application/json'
                }
            )
            response.raise_for_status()
            
            # Parse JSON response
            content_data = response.json()
            logger.info(f"Successfully extracted content from Jina.ai for {url}")
            logger.debug(f"Response size: {len(str(content_data))} characters")
            
            # Extract the actual content from the nested structure
            return {
                'content': content_data['data']['content'],
                'title': content_data['data']['title'],
                'description': content_data['data']['description'],
                'url': url
            }
            
        except requests.RequestException as e:
            logger.error(f"Jina content extraction failed for {url}: {str(e)}")
            return None

class JinaExtractorLinks(WebExtractor):
    """Jina.ai based links extraction"""
    def __init__(self):
        super().__init__()
        self.base_url = 'https://r.jina.ai'
        
    def extract_links(self, url: str) -> List[Dict[str, str]]:
        """Extract links using Jina.ai"""
        try:
            response = requests.get(
                f'{self.base_url}/{url}',
                headers={
                    **self.headers,
                    'X-With-Links-Summary': 'true'
                }
            )
            response.raise_for_status()
            
            # Parse the links from the response
            return self._parse_links(response.text)
            
        except requests.RequestException as e:
            logger.error(f"Jina links extraction failed: {str(e)}")
            return None

    def _parse_links(self, content: str) -> List[Dict[str, str]]:
        """Parse links from content"""
        links = []
        lines = content.split('\n')
        
        for line in lines:
            # Look for markdown-style links [text](url)
            matches = re.findall(r'\[(.*?)\]\((.*?)\)', line)
            for text, url in matches:
                links.append({
                    'text': text.strip(),
                    'url': url.strip()
                })
                
        return links

class SeleniumExtractor:
    """Selenium based extraction fallback"""
    def __init__(self):
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        self.driver = webdriver.Chrome(options=chrome_options)
        
    def extract_content(self, url: str) -> Dict[str, Union[str, List[str]]]:
        """Extract content using Selenium"""
        try:
            self.driver.get(url)
            time.sleep(5)  # Wait for dynamic content to load
            
            # Get the page title and description
            title = self.driver.title
            description = self._get_meta_description()
            
            # Extract main content text (avoiding scripts, styles, etc.)
            content = self._extract_clean_content()
            
            # Extract links
            links = self._extract_links()
            
            # Format the response to match Jina.ai structure
            return {
                'content': {
                    'content': content,
                    'title': title,
                    'description': description,
                    'url': url
                },
                'links': links
            }
            
        except Exception as e:
            logger.error(f"Selenium extraction failed: {str(e)}")
            return None
            
    def _get_meta_description(self) -> str:
        """Extract meta description"""
        try:
            meta = self.driver.find_element('xpath', "//meta[@name='description']")
            return meta.get_attribute('content')
        except:
            return ""
            
    def _extract_clean_content(self) -> str:
        """Extract and clean content text"""
        try:
            # Remove unwanted elements
            for element in self.driver.find_elements('xpath', 
                "//script | //style | //noscript | //iframe | //svg"):
                self.driver.execute_script(
                    "arguments[0].parentNode.removeChild(arguments[0])", element)
            
            # Try to find pricing specific content first
            pricing_content = None
            for selector in [
                "//div[contains(@class, 'pricing')]",
                "//section[contains(@class, 'pricing')]",
                "//div[contains(@id, 'pricing')]",
                "//main"
            ]:
                elements = self.driver.find_elements('xpath', selector)
                if elements:
                    pricing_content = elements[0]
                    break
            
            if pricing_content:
                content = pricing_content.text
            else:
                content = self.driver.find_element('xpath', "//body").text
            
            # Clean up the content
            content = self._clean_text(content)
            
            return content
            
        except Exception as e:
            logger.error(f"Content extraction failed: {str(e)}")
            return ""
            
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove non-printable characters
        text = ''.join(char for char in text if char.isprintable())
        # Remove any remaining HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        return text.strip()
            
    def _extract_links(self) -> List[Dict[str, str]]:
        """Extract relevant links"""
        links = []
        elements = self.driver.find_elements('xpath', "//a[@href]")
        for element in elements:
            try:
                href = element.get_attribute('href')
                text = element.text.strip()
                if href and text and not href.startswith('javascript:'):
                    links.append({
                        'text': text,
                        'url': href
                    })
            except:
                continue
        return links
        
    def __del__(self):
        """Clean up Selenium driver"""
        try:
            self.driver.quit()
        except:
            pass

class PricingExtractor:
    """Main class for extracting pricing information"""
    def __init__(self):
        self.content_extractor = JinaExtractorContent()
        self.links_extractor = JinaExtractorLinks()
        self.selenium_extractor = SeleniumExtractor()
        self.pricing_agent = PricingAgent()
        self.schema_validator = PricingSchemaValidator()
        
        # Create all necessary directories
        self.directories = {
            'raw_content': Path("raw_content"),
            'raw_links': Path("raw_links"),
            'parsed_content': Path("parsed_content"),
            'validated_content': Path("validated_content"),
            'extracted_info': Path("extracted_info")  # Keep for compatibility
        }
        
        for directory in self.directories.values():
            directory.mkdir(exist_ok=True)

    def _get_service_name(self, url: str) -> str:
        """Extract service name from URL"""
        domain = urlparse(url).netloc
        return domain.split('.')[0] if domain.startswith('www.') else domain.split('.')[0]

    def extract_and_parse(self, url: str) -> tuple:
        """Extract content, parse it, and validate the results"""
        if not self.content_extractor.is_valid_url(url):
            raise ValueError(f"Invalid URL provided: {url}")

        try:
            # Extract raw content
            logger.info(f"Starting extraction process for {url}")
            extracted_data = self.extract(url)
            
            if not extracted_data:
                logger.error(f"No data extracted from {url}")
                return None, None, None, None
            
            # Save raw content
            service_name = self._get_service_name(url)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. Save raw content
            raw_content_path = self.save_raw_content(extracted_data, service_name, timestamp)
            logger.info(f"Raw content saved to: {raw_content_path}")
            
            # 2. Process with GPT-4 extractor
            logger.info("Starting GPT-4 extraction process...")
            parsed_path = self.process_with_gpt4(extracted_data, service_name, timestamp)
            logger.info(f"Parsed content saved to: {parsed_path}")
            
            # 3. Validate and transform
            logger.info("Validating and transforming parsed data...")
            validated_path = self.validate_and_transform(parsed_path, service_name, timestamp)
            logger.info(f"Validated content saved to: {validated_path}")
            
            return raw_content_path, parsed_path, validated_path
            
        except Exception as e:
            logger.error(f"Pipeline failed for {url}: {str(e)}", exc_info=True)
            return None, None, None

    def save_raw_content(self, extracted_data: dict, service_name: str, timestamp: str) -> Path:
        """Save raw extracted content"""
        raw_content_path = self.directories['raw_content'] / f"raw_{service_name}_{timestamp}.json"
        content_data = {
            "service_name": service_name,
            "extraction_date": datetime.now().isoformat(),
            "content": extracted_data['content'],
            "links": extracted_data.get('links', [])
        }
        
        with open(raw_content_path, 'w', encoding='utf-8') as f:
            json.dump(content_data, f, indent=2)
            
        return raw_content_path

    def process_with_gpt4(self, extracted_data: dict, service_name: str, timestamp: str) -> Path:
        """Process content with GPT-4 extractor"""
        from gpt4o_extractor_agent import process_pricing_json
        
        # Prepare data for GPT-4
        extraction_data = {
            "data": {
                "content": extracted_data['content'],
                "title": extracted_data.get('title', ''),
                "description": extracted_data.get('description', ''),
                "url": extracted_data.get('url', '')
            }
        }
        
        # Save to parsed content directory
        parsed_path = self.directories['parsed_content'] / f"parsed_{service_name}_{timestamp}.json"
        
        # Process with GPT-4
        result = process_pricing_json(extraction_data)
        
        with open(parsed_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
            
        return parsed_path

    def validate_and_transform(self, parsed_path: Path, service_name: str, timestamp: str) -> Path:
        """Validate and transform parsed data"""
        validated_path = self.directories['validated_content'] / f"validated_{service_name}_{timestamp}.json"
        
        # Validate and transform
        is_valid, transformed_data, errors = self.schema_validator.validate_and_transform(str(parsed_path))
        
        if errors:
            logger.warning("Validation produced warnings:")
            for error in errors:
                logger.warning(f"• {error}")
        
        # Save transformed data
        self.schema_validator.save_transformed(transformed_data, str(validated_path))
        
        return validated_path

    def extract(self, url: str) -> Dict[str, Union[str, List[str]]]:
        """Extract pricing information with fallback mechanism"""
        # Try Jina.ai content extraction
        content = self.content_extractor.extract_content(url)
        links = self.links_extractor.extract_links(url)
        
        # Fallback to Selenium if either fails
        if content is None or links is None:
            logger.info("Falling back to Selenium extraction")
            selenium_result = self.selenium_extractor.extract_content(url)
            if selenium_result:
                content = selenium_result['content']
                links = [{'text': 'Link', 'url': url} for url in selenium_result['links']]

        if content is None and links is None:
            raise Exception("Both extraction methods failed")

        return {
            'content': content,
            'links': links
        }

def main():
    try:
        extractor = PricingExtractor()
        urls = [
            "https://apify.com/pricing"
            # Add more URLs as needed
        ]
        
        for url in urls:
            logger.info(f"\nProcessing {url}")
            try:
                raw_path, parsed_path, validated_path = extractor.extract_and_parse(url)
                
                if all(path is None for path in [raw_path, parsed_path, validated_path]):
                    logger.warning(f"Skipped processing for {url} due to extraction failure")
                    continue
                    
                logger.info(f"Successfully processed {url}")
                logger.info(f"Raw content: {raw_path}")
                logger.info(f"Parsed content: {parsed_path}")
                logger.info(f"Validated content: {validated_path}")
                    
            except Exception as e:
                logger.error(f"Failed to process {url}: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")

if __name__ == "__main__":
    main()
