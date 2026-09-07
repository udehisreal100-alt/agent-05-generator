import os
import json
import re
import zipfile
import requests
from groq import Groq

# ---------------------------------------------------------------------------
# 1. Environment & API Credentials Configuration
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GUMROAD_ACCESS_TOKEN = os.environ.get("GUMROAD_ACCESS_TOKEN")

if not GROQ_API_KEY:
    print("[!] Warning: GROQ_API_KEY environment variable is missing.")

if not GUMROAD_ACCESS_TOKEN:
    print("[!] Warning: GUMROAD_ACCESS_TOKEN environment variable is missing.")

# Initialize Groq Client
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ---------------------------------------------------------------------------
# 2. Topic Discovery Engine
# ---------------------------------------------------------------------------
def get_target_topic() -> str:
    """Returns target topic for digital asset generation."""
    return "FastAPI middleware for rate-limiting and Supabase authentication logging"

# ---------------------------------------------------------------------------
# 3. Groq Generation Module (qwen/qwen3.8-27b with token limit fixes)
# ---------------------------------------------------------------------------
SYSTEM_INSTRUCTION = """
You are an expert software engineer and digital product creator.
Generate a complete, production-ready Python script and README for the target topic.
Keep the code, docs, and sales copy concise and direct so the entire JSON stays under 900 output tokens.

You MUST return your response as a single, valid JSON object matching this exact schema:
{
  "filename": "middleware.py",
  "code": "FULL_PYTHON_CODE_HERE",
  "readme": "CONCISE_README_MARKDOWN_HERE",
  "product_title": "PUNCHY_GUMROAD_PRODUCT_TITLE",
  "product_description": "SHORT_GUMROAD_SALES_COPY_MARKDOWN",
  "price_usd": 9
}

CRITICAL JSON ESCAPING RULES:
1. Output strictly valid JSON.
2. Escape all internal double quotes inside string values as \\\".
3. Represent line breaks in code and docs with \\n.
"""

def generate_digital_asset_with_groq(topic: str) -> dict:
    """Queries Groq API using qwen/qwen3.8-27b with explicit max_tokens (950) to fit 1k OTPM limit."""
    if not groq_client:
        raise ValueError("Groq client not initialized. Check GROQ_API_KEY.")
        
    print(f"[*] Querying Groq API for topic: '{topic}'...")
    
    response = groq_client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": f"Generate a complete digital asset for topic: {topic}"}
        ],
        temperature=0.1,
        max_tokens=950  # Enforces response size below Groq's 1000 OTPM rate limit cap
    )
    
    raw_content = response.choices[0].message.content.strip()
    
    # Strip markdown code block wrappers if present (e.g. ```json ... ```)
    clean_json = re.sub(r"^```(?:json)?\s*", "", raw_content, flags=re.MULTILINE)
    clean_json = re.sub(r"\s*```$", "", clean_json, flags=re.MULTILINE).strip()
    
    try:
        return json.loads(clean_json)
    except json.JSONDecodeError as e:
        print(f"[!] JSON decoding error: {e}")
        print(f"Raw Output Snippet:\n{raw_content[:300]}...")
        raise e

# ---------------------------------------------------------------------------
# 4. Local ZIP Packaging Module
# ---------------------------------------------------------------------------
def create_asset_zip(filename: str, code: str, readme: str, zip_output_path: str):
    """Packages the code file and README.md into a compressed ZIP archive."""
    with zipfile.ZipFile(zip_output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr(filename, code)
        zipf.writestr("README.md", readme)
    print(f"[✔] Package created successfully: {zip_output_path}")

# ---------------------------------------------------------------------------
# 5. Storefront Publisher Modules
# ---------------------------------------------------------------------------
def publish_to_gumroad(title: str, description: str, price_usd: int, zip_file_path: str):
    """Publishes the generated product and uploads the ZIP payload to Gumroad via REST API."""
    if not GUMROAD_ACCESS_TOKEN:
        print("[✘] Skipping Gumroad publishing: GUMROAD_ACCESS_TOKEN not set.")
        return None

    url = "https://api.gumroad.com/v2/products"
    payload = {
        "access_token": GUMROAD_ACCESS_TOKEN,
        "name": title,
        "price": price_usd * 100,  # Convert USD to cents ($9 = 900)
        "description": description,
        "customizable_price": "false",
    }
    
    print(f"[*] Publishing '{title}' (${price_usd}) to Gumroad...")
    with open(zip_file_path, "rb") as f:
        files = {"file": (os.path.basename(zip_file_path), f, "application/zip")}
        response = requests.post(url, data=payload, files=files)
        
    if response.status_code in (200, 201):
        res_data = response.json()
        product_url = res_data.get("product", {}).get("short_url", "N/A")
        print(f"[✔] Gumroad Listing Success! URL: {product_url}")
        return res_data
    else:
        print(f"[✘] Gumroad Error ({response.status_code}): {response.text}")
        return None

# ---------------------------------------------------------------------------
# 6. Pipeline Orchestrator
# ---------------------------------------------------------------------------
def run_agent_pipeline():
    print("==================================================")
    print("      AGENT #05: AUTOMATED DIGITAL ASSET PIPELINE")
    print("==================================================")
    
    topic = get_target_topic()
    print(f"[*] Selected Target Topic: {topic}")
    
    asset_data = generate_digital_asset_with_groq(topic)
    
    filename = asset_data.get("filename", "middleware.py")
    code = asset_data.get("code", "")
    readme = asset_data.get("readme", "")
    title = asset_data.get("product_title", "Automated Digital Asset")
    description = asset_data.get("product_description", "")
    price = asset_data.get("price_usd", 9)
    
    zip_path = f"payload_{os.urandom(3).hex()}.zip"
    create_asset_zip(filename, code, readme, zip_path)
    
    try:
        publish_to_gumroad(title, description, price, zip_path)
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print("[*] Temporary ZIP file cleaned up.")
            
    print("==================================================")
    print("      PIPELINE EXECUTION COMPLETE")
    print("==================================================")

if __name__ == "__main__":
    run_agent_pipeline()