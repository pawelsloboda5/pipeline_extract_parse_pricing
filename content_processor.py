from bs4 import BeautifulSoup
import re
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class ContentProcessor:
    """Process and clean raw HTML content to extract pricing-related information"""
    
    def __init__(self):
        self.pricing_keywords = [
            'pricing', 'price', 'plan', 'subscription', 'tier', 'free',
            'starter', 'premium', 'enterprise', 'business', 'professional',
            'monthly', 'annually', 'year', 'month', 'cost', 'fee'
        ]

    def extract_pricing_content(self, raw_html: str) -> Dict[str, any]:
        """Extract relevant pricing information from raw HTML"""
        try:
            soup = BeautifulSoup(raw_html, 'html.parser')
            
            # Remove script and style elements
            for element in soup(['script', 'style', 'noscript', 'iframe', 'meta']):
                element.decompose()

            # Get the title
            title = self._extract_title(soup)
            
            # Get main pricing content
            pricing_sections = self._find_pricing_sections(soup)
            
            # Extract pricing tables if they exist
            pricing_tables = self._extract_pricing_tables(soup)
            
            # Extract text content from pricing sections
            pricing_text = self._extract_text_content(pricing_sections)
            
            # Get feature lists
            feature_lists = self._extract_feature_lists(soup)

            return {
                "title": title,
                "pricing_sections": pricing_text,
                "pricing_tables": pricing_tables,
                "feature_lists": feature_lists,
                "raw_text": self._clean_text(soup.get_text(separator=' ', strip=True))
            }
            
        except Exception as e:
            logger.error(f"Error processing content: {str(e)}")
            return None

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract the page title"""
        title = soup.find('title')
        if title:
            return title.text.strip()
        
        # Try h1 if title tag is not found
        h1 = soup.find('h1')
        if h1:
            return h1.text.strip()
        
        return ""

    def _find_pricing_sections(self, soup: BeautifulSoup) -> List[BeautifulSoup]:
        """Find sections likely to contain pricing information"""
        pricing_sections = []
        
        # Look for sections with pricing-related IDs or classes
        pricing_patterns = re.compile(r'|'.join(self.pricing_keywords), re.IGNORECASE)
        
        # Find elements with pricing-related attributes
        for element in soup.find_all(['div', 'section']):
            # Check id and class attributes
            element_id = element.get('id', '')
            element_classes = ' '.join(element.get('class', []))
            
            if (pricing_patterns.search(element_id) or 
                pricing_patterns.search(element_classes)):
                pricing_sections.append(element)
        
        return pricing_sections

    def _extract_pricing_tables(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract pricing tables from the content"""
        tables = []
        
        # Find table elements
        for table in soup.find_all(['table', 'div[role="table"]']):
            table_data = []
            
            # Handle traditional table elements
            if table.name == 'table':
                rows = table.find_all('tr')
                for row in rows:
                    cols = row.find_all(['td', 'th'])
                    table_data.append([self._clean_text(col.text) for col in cols])
            
            # Handle div-based tables
            else:
                rows = table.find_all('div[role="row"]')
                for row in rows:
                    cols = row.find_all('div[role="cell"]')
                    table_data.append([self._clean_text(col.text) for col in cols])
            
            if table_data:
                tables.append(table_data)
        
        return tables

    def _extract_feature_lists(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract feature lists from pricing sections"""
        features = []
        
        # Look for common feature list patterns
        for ul in soup.find_all(['ul', 'ol']):
            list_items = ul.find_all('li')
            if list_items:
                feature_list = []
                for li in list_items:
                    feature_text = self._clean_text(li.text)
                    if feature_text:
                        feature_list.append(feature_text)
                if feature_list:
                    features.append(feature_list)
        
        return features

    def _extract_text_content(self, sections: List[BeautifulSoup]) -> List[str]:
        """Extract and clean text content from pricing sections"""
        content = []
        
        for section in sections:
            # Extract text from paragraphs and headers
            for element in section.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                text = self._clean_text(element.text)
                if text:
                    content.append(text)
        
        return content

    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        if not text:
            return ""
            
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        return text.strip()

def main():
    # Example usage
    try:
        with open('raw_content_storage/raw_content_airtable_20241217_010235.txt', 'r', encoding='utf-8') as f:
            raw_html = f.read()
        
        processor = ContentProcessor()
        processed_content = processor.extract_pricing_content(raw_html)
        
        # Save processed content
        import json
        from pathlib import Path
        
        output_dir = Path('processed_content_storage')
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / 'processed_airtable_20241217_010235.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_content, f, indent=2)
            
        logger.info(f"Processed content saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")

if __name__ == "__main__":
    main() 