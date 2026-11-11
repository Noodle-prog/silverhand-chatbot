#requires-python = ">=3.14"
dependencies = [
     "chromadb>=1.5.9",
     "cloudscraper>=1.2.71",
     "gradio>=6.31.0",
     "httpx>=0.28.1",
     "markdownify>=1.2.3",
     "openai>=3.42.0",
     "requests>=2.34.2",
]

import glob
import os
import time
import random
import requests  # type: ignore
import markdownify  # type: ignore
import chromadb  # type: ignore
import re

# List the exact wiki page titles instead of full URLs
johnny_wiki_pages = [
    "Johnny_Silverhand",
    "Kerry_Eurodyne",
    "Rogue_Amendiares",
    ]

def download_data() -> str:
    directory_path = "./data/"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        
    api_url = "https://cyberpunk.fandom.com/api.php"
    headers = {
        "User-Agent": "JohnnyRAGBot/1.0 (Educational Python Script; contact@example.com)"
    }

    for page_title in johnny_wiki_pages:
        print(f"Fetching API data for {page_title}...")
        params = {
            "action": "parse",
            "page": page_title,
            "format": "json"
        }
        
        try:
            response = requests.get(api_url, params=params, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            # Extract the raw HTML content from the API response
            html_content = data["parse"]["text"]["*"]
        except Exception as e:
            print(f"Failed to fetch {page_title}: {e}")
            time.sleep(5)
            continue

        # Convert HTML to markdown
        markdown_content = markdownify.markdownify(
            html_content, strip=["a", "img"], heading_style="ATX"
        )
        
        # Safely clean up sections if needed
        gallery_idx = markdown_content.find("## Gallery")
        if gallery_idx != -1:
            markdown_content = markdown_content[:gallery_idx]

        filename = directory_path + page_title + ".md"
        print(f"Writing to {filename}...")
        with open(filename, "w", encoding="utf-8") as text_file:
            text_file.write(markdown_content)

        # Polite random delay
        sleep_time = random.uniform(1.5, 3.0)
        time.sleep(sleep_time)

def index_data():
    for filename in glob.glob(os.path.join("./data/", "*.md")):
        print(f"Indexing {filename}...")
        with open(filename, "r", encoding="utf-8") as text_file:
            markdown_content = text_file.read()
            
        paragraphs = split_markdown_by_header(markdown_content)
        if not paragraphs:
            continue
            
        chroma_client = chromadb.PersistentClient(path="./chromadb-data-johnny")
        collection = chroma_client.get_or_create_collection(name="character")
        collection.upsert(
            documents=paragraphs,
            metadatas=[{"source": filename}] * len(paragraphs),
            ids=[f"{filename}-{i}" for i in range(len(paragraphs))],
        )

def split_markdown_by_header(text):
    lines = text.splitlines()
    chunks = []
    current_chunk = ""

    for line in lines:
        if line.startswith("#"):
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = line + "\n"
        else:
            current_chunk += line + "\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

if __name__ == "__main__":
    download_data() 
    index_data()
