# news_extraction.py 
import requests
from bs4 import BeautifulSoup
import re
import extraction # Import the new library


def fetch_and_extract_article_data_from_url(url):
    """
    Fetches HTML from URL, scrapes full text using BeautifulSoup,
    and extracts metadata (title, description) using the 'extraction' library.
    Returns: title, text, description, canonical_url
    """
    print(f"  Fetching and extracting data from URL: {url}")
    article_title = "Title Not Found"
    article_text = None
    article_description = None # From 'extraction' library
    canonical_url = url # Default to input URL

    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        html_content = response.text # Use .text for extraction library

        # Use 'extraction' library for metadata
        try:
            extracted_meta = extraction.Extractor().extract(html_content, source_url=url)
            if extracted_meta:
                article_title = extracted_meta.title if extracted_meta.title else article_title
                article_description = extracted_meta.description
                canonical_url = extracted_meta.url if extracted_meta.url else url
                print(f"    Extraction lib: Title='{article_title[:50]}...', Desc='{str(article_description)[:70]}...'")
        except Exception as e_extract:
            print(f"    Warning: 'extraction' library failed or found no metadata: {e_extract}")

        # Use BeautifulSoup for full text
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find a main content area (common class/id names)
        main_content_selectors = ['article', 'main', '.main-content', '#main-content', '.post-content', '#content']
        content_area = None
        for selector in main_content_selectors:
            if selector.startswith('.'): # class
                content_area = soup.find(class_=selector[1:])
            elif selector.startswith('#'): # id
                content_area = soup.find(id=selector[1:])
            else: # tag
                content_area = soup.find(selector)
            if content_area:
                break
        
        target_soup = content_area if content_area else soup # Use content_area if found, else whole soup

        paragraphs = target_soup.find_all('p')
        article_text_parts = [para.get_text(separator=" ", strip=True) for para in paragraphs]
        article_text = "\n\n".join(filter(None, article_text_parts)) # Join non-empty paragraphs

        # If title wasn't found by 'extraction', try with BeautifulSoup from h1
        if article_title == "Title Not Found" or not article_title:
            h1_tag = soup.find('h1')
            if h1_tag:
                article_title = h1_tag.get_text(strip=True)

        if not article_text: # Fallback if no <p> tags yielded substantial text
            print("    No <p> tags found or they were empty. Falling back to broader text extraction.")
            # Remove script, style, nav, header, footer tags before getting all text
            for unwanted_tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                unwanted_tag.decompose()
            all_text_stripped = soup.get_text(separator='\n', strip=True)
            article_text = re.sub(r'\n\s*\n+', '\n\n', all_text_stripped) # Consolidate multiple newlines

        print(f"    BS4 Scraped: Title='{article_title[:50]}...', Text len: {len(article_text if article_text else '')}")
        return article_title, article_text, article_description, canonical_url

    except requests.exceptions.RequestException as e_req:
        print(f"    Error fetching URL {url}: {e_req}")
    except Exception as e_main:
        print(f"    Error processing URL {url}: {e_main}")
    return article_title, None, None, url # Return defaults on error