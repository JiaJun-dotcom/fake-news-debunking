# news_extraction.py
import requests
from bs4 import BeautifulSoup
import re
import extraction 
import time

# --- Selenium ---
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import WebDriverException, TimeoutException as SeleniumTimeoutException


# --- Helper function to fetch HTML using Selenium ---
def _fetch_with_selenium(url: str, timeout_seconds: int = 30) -> str | None:
    """Fetches HTML content using Selenium (headless Chrome)."""
    print(f"    Attempting fetch with Selenium for: {url}")
    html_content = None
    driver = None
    try:
        options = ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox") # Often needed in containerized/Linux environments
        options.add_argument("--disable-dev-shm-usage") # Overcome limited resource problems
        options.add_argument("--window-size=1920,1080") # Can sometimes help with layout-dependent sites
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36")
        options.add_argument("--log-level=3")

        driver_service = ChromeService(ChromeDriverManager().install())
        
        driver = webdriver.Chrome(service=driver_service, options=options)
        driver.set_page_load_timeout(timeout_seconds) 
        
        driver.get(url)

        time.sleep(3) # 3 seconds explicit wait
        
        html_content = driver.page_source
        print(f"    Selenium successfully fetched content (len: {len(html_content)}).")

    except SeleniumTimeoutException:
        print(f"    Selenium Timeout after {timeout_seconds}s for URL: {url}")
    except WebDriverException as e_wd:
        print(f"    Selenium WebDriverException for URL {url}: {e_wd}")
    except Exception as e_sel: 
        print(f"    Unexpected Selenium error for URL {url}: {e_sel}")
    finally:
        if driver:
            try:
                driver.quit() 
            except Exception as e_quit:
                print(f"    Error quitting Selenium driver: {e_quit}")
    return html_content

def fetch_and_extract_article_data_from_url(url, use_selenium_fallback=True):
    """
    Fetches HTML from URL, scrapes full text using BeautifulSoup,
    and extracts metadata (title, description) using the 'extraction' library.
    Tries requests first, then Selenium as fallback if specified and requests fails.
    Returns: title, text, description, canonical_url
    """
    print(f"  Fetching and extracting data from URL: {url}")
    article_title = "Title Not Found"
    article_text = None
    article_description = None
    canonical_url = url
    html_content = None

    try:
        print("    Attempting fetch with 'requests' library...")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)
        html_content = response.text
        print(f"    'requests' successful (status: {response.status_code}, len: {len(html_content)}).")
    except requests.exceptions.HTTPError as e_http:
        print(f"    'requests' HTTPError: {e_http.response.status_code} for URL {url}. Message: {e_http}")
        if use_selenium_fallback and e_http.response.status_code in [400, 401, 403, 404, 406, 429, 503]: # Codes indicating bot protection, not found, or overload
            print("    Falling back to Selenium due to HTTPError.")
            html_content = _fetch_with_selenium(url)
        else:
            print(f"    'requests' HTTPError ({e_http.response.status_code}) not triggering Selenium fallback or fallback disabled.")
    
    except requests.exceptions.RequestException as e_req: # Other issues like DNS failure, connection timeout
        print(f"    'requests' RequestException (e.g., connection error, timeout): {e_req} for URL {url}")
        if use_selenium_fallback:
            print("    Falling back to Selenium due to RequestException.")
            html_content = _fetch_with_selenium(url)
    
    if not html_content and use_selenium_fallback:
        print("    'requests' resulted in no content or an issue not caught for fallback, trying Selenium.")
        html_content = _fetch_with_selenium(url)

    if html_content:
        try:
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
            
            main_content_selectors = ['article', 'main', '.main-content', '#main-content', '.post-content', '#content', '[role="main"]']
            content_area = None
            for selector in main_content_selectors:
                try:
                    content_area = soup.select_one(selector)
                    if content_area:
                        break
                except Exception:
                    continue
            
            target_soup = content_area if content_area else soup

            paragraphs = target_soup.find_all('p')
            article_text_parts = [para.get_text(separator=" ", strip=True) for para in paragraphs]
            
            min_meaningful_paragraph_length = 50
            meaningful_text_parts = [p for p in article_text_parts if len(p) > min_meaningful_paragraph_length]

            if not meaningful_text_parts and article_text_parts:
                article_text = "\n\n".join(filter(None, article_text_parts))
            else:
                article_text = "\n\n".join(filter(None, meaningful_text_parts))

            if (article_title == "Title Not Found" or not article_title.strip()) and soup.find('h1'):
                article_title = soup.find('h1').get_text(strip=True)

            if not article_text or len(article_text) < 100:
                print("    No substantial <p> tags found or they were empty. Falling back to broader text extraction.")
                for unwanted_tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "figure", "figcaption"]):
                    unwanted_tag.decompose()
                all_text_stripped = soup.get_text(separator='\n', strip=True)
                lines = (line.strip() for line in all_text_stripped.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                article_text = '\n\n'.join(chunk for chunk in chunks if chunk)

            print(f"    BS4 Scraped: Title='{article_title[:50]}...', Text len: {len(article_text if article_text else '')}")
            return article_title, article_text, article_description, canonical_url

        except Exception as e_parse:
            print(f"    Error parsing HTML for URL {url}: {e_parse}")
    else:
        print(f"    Failed to fetch HTML content for {url} using all available methods.")

    return article_title, None, None, url # Return defaults on error

if __name__ == "__main__":
    print("\n--- Testing News Extraction from URLs ---")

    test_urls = [
        "https://www.reuters.com/world/europe/russia-pounds-ukraines-kharkiv-says-it-has-seized-two-more-villages-2024-05-14/", 
        "https://www.bbc.com/news/world-us-canada-68993200", 
        "https://techcrunch.com/2024/05/13/google-search-is-getting-an-ai-makeover/", 
        "https://www.theverge.com/2024/5/14/24156238/google-io-2024-ai-android-gemini-search-biggest-announcements", # 
        "https://www.simplypsychology.org/cognitive-dissonance.html", 
        "http://nonexistentfakewebsite12345.com/article", # Example of a bad URL
        "https://example.com" 
    ]

    for url in test_urls:
        title, text, description, canon_url = fetch_and_extract_article_data_from_url(url)

        print("\n--- Results for:", url, "---")
        print(f"  Canonical URL: {canon_url}")
        print(f"  Extracted Title: {title}")
        print(f"  Extracted Description (first 100 chars): {str(description)[:100] if description else 'N/A'}...")
        
        if text:
            print(f"  Extracted Text Length: {len(text)}")
            print(f"  Text Snippet (first 200 chars): {text[:200].strip()}...")
        else:
            print("  Extracted Text: None or Empty")
        print("-" * 50)