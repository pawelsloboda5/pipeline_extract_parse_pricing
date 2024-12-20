from bs4 import BeautifulSoup
import re
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class ContentProcessor:
    """Process and clean raw HTML content to extract plain text"""
    
    def __init__(self):
        # Tags to be completely removed
        self.remove_tags = [
            'script', 'style', 'noscript', 'iframe', 'meta',
            'header', 'footer', 'nav', 'aside', 'svg', 'path',
            'button', 'img', 'defs', 'clipPath'
        ]
        
        # Tags that likely contain pricing content
        self.pricing_tags = [
            'div[class*="pricing"]',
            'div[class*="plan"]',
            'div[class*="tier"]',
            'section[class*="pricing"]',
            'section[id*="pricing"]'
        ]

    def extract_text_content(self, raw_html: str) -> str:
        """Extract clean text content from raw HTML"""
        try:
            soup = BeautifulSoup(raw_html, 'html.parser')
            
            # Remove unwanted tags and their content
            for tag in self.remove_tags:
                for element in soup.find_all(tag):
                    element.decompose()
            
            # Try to find pricing-specific sections first
            pricing_content = []
            for selector in self.pricing_tags:
                sections = soup.select(selector)
                for section in sections:
                    text = self._clean_text(section.get_text(separator=' ', strip=True))
                    if text:
                        pricing_content.append(text)
            
            # If no pricing sections found, get all visible text
            if not pricing_content:
                visible_text = soup.get_text(separator=' ', strip=True)
                pricing_content = [self._clean_text(visible_text)]
            
            # Join all text with newlines between sections
            return '\n\n'.join(filter(None, pricing_content))
            
        except Exception as e:
            logger.error(f"Error processing content: {str(e)}")
            return ""

    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove non-ASCII characters but keep basic punctuation and currency symbols
        text = re.sub(r'[^\x20-\x7E$€£¥]', '', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove empty parentheses and brackets
        text = re.sub(r'\(\s*\)|\[\s*\]|\{\s*\}', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '', text)
        
        # Remove multiple newlines
        text = re.sub(r'\n\s*\n', '\n', text)
        
        return text.strip()

def main():
    # Example usage
    try:
        with open('raw_content_storage/raw_content_airtable_20241217_010235.txt', 'r', encoding='utf-8') as f:
            raw_html = f.read()
        
        processor = ContentProcessor()
        clean_text = processor.extract_text_content(raw_html)
        
        # Save processed content
        from pathlib import Path
        
        output_dir = Path('processed_content_storage')
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / 'processed_text_airtable_20241217_010235.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(clean_text)
            
        logger.info(f"Processed text saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")

if __name__ == "__main__":
    main() 