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
        
    def extract_content(self, url: str) -> str:
        """Extract content using Jina.ai"""
        try:
            response = requests.get(
                f'{self.base_url}/{url}',
                headers={
                    **self.headers,
                    'X-Return-Format': 'text'
                }
            )
            response.raise_for_status()
            return response.text
            
        except requests.RequestException as e:
            logger.error(f"Jina content extraction failed: {str(e)}")
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

class SeleniumExtractor(WebExtractor):
    """Selenium-based fallback extraction"""
    def __init__(self):
        super().__init__()
        self.options = Options()
        self.options.add_argument('--headless')
        self.options.add_argument('--no-sandbox')
        self.options.add_argument('--disable-dev-shm-usage')

    def extract_content(self, url: str) -> Dict[str, Union[str, List[str]]]:
        """Extract content using Selenium"""
        try:
            driver = webdriver.Chrome(options=self.options)
            driver.get(url)
            content = driver.page_source
            domain = self.extract_domain(url)
            
            # Extract links using updated Selenium syntax
            links = []
            from selenium.webdriver.common.by import By
            elements = driver.find_elements(By.TAG_NAME, 'a')
            for element in elements:
                href = element.get_attribute('href')
                if href and domain in href:
                    links.append(href)

            return {
                'content': content,
                'links': list(set(links))
            }
        except Exception as e:
            logger.error(f"Selenium extraction failed: {str(e)}")
            return None
        finally:
            if 'driver' in locals():
                driver.quit()

class PricingExtractor:
    """Main class for extracting pricing information"""
    def __init__(self):
        self.content_extractor = JinaExtractorContent()
        self.links_extractor = JinaExtractorLinks()
        self.selenium_extractor = SeleniumExtractor()
        self.pricing_agent = PricingAgent()
        
        # Create storage directories
        self.raw_content_dir = Path("raw_content_storage")
        self.raw_links_dir = Path("raw_links_storage")
        self.parsed_content_dir = Path("parsed_content_storage")
        
        # Create all necessary directories
        for directory in [self.raw_content_dir, self.raw_links_dir, self.parsed_content_dir]:
            directory.mkdir(exist_ok=True)

    def _get_service_name(self, url: str) -> str:
        """Extract service name from URL"""
        domain = urlparse(url).netloc
        return domain.split('.')[0] if domain.startswith('www.') else domain.split('.')[0]

    def _generate_filename(self, service_name: str, file_type: str) -> str:
        """Generate filename with service name and current date"""
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{file_type}_{service_name}_{current_date}"

    def extract_and_parse(self, url: str) -> tuple:
        """Extract content and parse it using the pricing agent"""
        if not self.content_extractor.is_valid_url(url):
            raise ValueError("Invalid URL provided")

        try:
            # Extract raw content
            logger.info(f"Extracting content from {url}")
            extracted_data = self.extract(url)
            
            # Save raw content and links
            content_path, links_path = self.save_content(extracted_data, url)
            logger.info(f"Raw content saved to {content_path}")
            logger.info(f"Links saved to {links_path}")
            
            # Parse content using pricing agent with the newly saved content
            logger.info("Parsing content with pricing agent")
            with open(content_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()
                
            parsed_data = self.pricing_agent.parse_content(
                raw_content,
                url=url
            )
            
            # Save parsed data
            service_name = self._get_service_name(url)
            parsed_filename = self._generate_filename(service_name, "parsed")
            parsed_path = self.parsed_content_dir / f"{parsed_filename}.json"
            
            with open(parsed_path, 'w', encoding='utf-8') as f:
                json.dump(parsed_data, f, indent=2)
            
            logger.info(f"Parsed content saved to {parsed_path}")
            
            # Process content for ML if needed
            processed_path = self.process_content(extracted_data, url)
            if processed_path:
                logger.info(f"Processed content saved to {processed_path}")
            
            return content_path, links_path, parsed_path, processed_path
            
        except Exception as e:
            logger.error(f"Extraction and parsing failed: {str(e)}")
            raise

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

    def save_content(self, content: Dict[str, Union[str, List]], url: str) -> tuple:
        """Save content and links to respective directories"""
        service_name = self._get_service_name(url)
        
        # Save raw content
        content_filename = self._generate_filename(service_name, "raw_content")
        content_path = self.raw_content_dir / f"{content_filename}.txt"
        with open(content_path, 'w', encoding='utf-8') as f:
            f.write(content['content'])
        
        # Save links
        links_filename = self._generate_filename(service_name, "raw_links")
        links_path = self.raw_links_dir / f"{links_filename}.json"
        links_data = {
            "service_name": service_name,
            "extraction_date": datetime.now().isoformat(),
            "source_url": url,
            "links": content['links']
        }
        with open(links_path, 'w', encoding='utf-8') as f:
            json.dump(links_data, f, indent=2)
            
        return content_path, links_path

    def process_content(self, content: Dict[str, Union[str, List[str]]], url: str) -> tuple:
        """Process the extracted content"""
        processor = ContentProcessor()
        processed_text = processor.extract_text_content(content['content'])
        
        if processed_text:
            # Save processed content
            service_name = self._get_service_name(url)
            filename = self._generate_filename(service_name, "processed")
            processed_path = Path("processed_content_storage") / f"{filename}.txt"
            processed_path.parent.mkdir(exist_ok=True)
            
            with open(processed_path, 'w', encoding='utf-8') as f:
                f.write(processed_text)
            
            return processed_path
        return None

def main():
    try:
        extractor = PricingExtractor()
        urls = [
            "https://airtable.com/pricing",
            # Add more URLs as needed
        ]
        
        for url in urls:
            logger.info(f"\nProcessing {url}")
            try:
                content_path, links_path, parsed_path, processed_path = extractor.extract_and_parse(url)
                
                logger.info(f"Successfully processed {url}")
                logger.info(f"Raw content: {content_path}")
                logger.info(f"Links: {links_path}")
                logger.info(f"Parsed data: {parsed_path}")
                if processed_path:
                    logger.info(f"Processed content: {processed_path}")
                    logger.info(f"Checking JSON: {parsed_path}")
                    validator = PricingSchemaValidator()
                    validator.validate_json(parsed_path)
                    
            except Exception as e:
                logger.error(f"Failed to process {url}: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")

if __name__ == "__main__":
    main()
