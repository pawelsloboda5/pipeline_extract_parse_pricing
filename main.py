import os
import requests
from typing import Dict, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass
from urllib.parse import urlparse
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import logging
from dotenv import load_dotenv
import json
from pathlib import Path
from content_processor import ContentProcessor

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

class JinaExtractor(WebExtractor):
    """Jina.ai based extraction"""
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv('JINA_API_KEY')
        if not self.api_key:
            raise ValueError("JINA_API_KEY not found in environment variables")
        self.base_url = 'https://r.jina.ai'
        self.headers.update({'Authorization': f'Bearer {self.api_key}'})

    def extract_content(self, url: str) -> Dict[str, Union[str, List[str]]]:
        """Extract content and links using Jina.ai"""
        try:
            # Get internal links
            links_response = requests.get(
                f'{self.base_url}/{url}',
                headers={**self.headers, 'X-With-Links-Summary': 'true'}
            )
            links_response.raise_for_status()

            # Get main content
            content_response = requests.get(
                f'{self.base_url}/{url}',
                headers={**self.headers, 'X-Return-Format': 'text'}
            )
            content_response.raise_for_status()

            return {
                'content': content_response.text,
                'links': self._parse_links(links_response.text, self.extract_domain(url))
            }
        except requests.RequestException as e:
            logger.error(f"Jina extraction failed: {str(e)}")
            return None

    def _parse_links(self, content: str, domain: str) -> List[str]:
        """Parse internal links from content"""
        soup = BeautifulSoup(content, 'html.parser')
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            if domain in href:
                links.append(href)
        return list(set(links))

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
        self.jina_extractor = JinaExtractor()
        self.selenium_extractor = SeleniumExtractor()
        
        # Create storage directories if they don't exist
        self.raw_content_dir = Path("raw_content_storage")
        self.raw_links_dir = Path("raw_links_storage")
        self.raw_content_dir.mkdir(exist_ok=True)
        self.raw_links_dir.mkdir(exist_ok=True)

    def _get_service_name(self, url: str) -> str:
        """Extract service name from URL"""
        domain = urlparse(url).netloc
        return domain.split('.')[0] if domain.startswith('www.') else domain.split('.')[0]

    def _generate_filename(self, service_name: str, file_type: str) -> str:
        """Generate filename with service name and current date"""
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"raw_{file_type}_{service_name}_{current_date}"

    def save_content(self, content: Dict[str, Union[str, List[str]]], url: str) -> tuple:
        """Save content and links to respective directories"""
        service_name = self._get_service_name(url)
        
        # Save raw content
        content_filename = self._generate_filename(service_name, "content")
        content_path = self.raw_content_dir / f"{content_filename}.txt"
        with open(content_path, 'w', encoding='utf-8') as f:
            f.write(content['content'])
        
        # Save links
        links_filename = self._generate_filename(service_name, "links")
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
        processed_content = processor.extract_pricing_content(content['content'])
        
        if processed_content:
            # Save processed content
            service_name = self._get_service_name(url)
            filename = self._generate_filename(service_name, "processed")
            processed_path = Path("processed_content_storage") / f"{filename}.json"
            processed_path.parent.mkdir(exist_ok=True)
            
            with open(processed_path, 'w', encoding='utf-8') as f:
                json.dump(processed_content, f, indent=2)
            
            return processed_path
        return None

    def extract(self, url: str) -> Dict[str, Union[str, List[str]]]:
        """Extract pricing information with fallback mechanism"""
        if not self.jina_extractor.is_valid_url(url):
            raise ValueError("Invalid URL provided")

        # Try Jina.ai first
        content = self.jina_extractor.extract_content(url)
        
        # Fallback to Selenium if Jina fails
        if content is None:
            logger.info("Falling back to Selenium extraction")
            content = self.selenium_extractor.extract_content(url)

        if content is None:
            raise Exception("Both extraction methods failed")

        return content

def main():
    # Example usage
    try:
        extractor = PricingExtractor()
        url = "https://airtable.com/pricing"
        
        logger.info(f"Starting extraction for {url}")
        content = extractor.extract(url)
        
        # Save content and links to respective directories
        content_path, links_path = extractor.save_content(content, url)
        
        logger.info(f"Extracted {len(content['links'])} internal links")
        logger.info(f"Content saved to {content_path}")
        logger.info(f"Links saved to {links_path}")
        
        processed_path = extractor.process_content(content, url)
        if processed_path:
            logger.info(f"Processed content saved to {processed_path}")
        
    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}")

if __name__ == "__main__":
    main()
