import os
import json
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
# 3. Groq Generation Module (llama-3.3-70b-versatile)
# ---------------------------------------------------------------------------
SYSTEM_INSTRUCTION = """
You are an expert software engineer and digital product creator.
Generate a complete, fully functional, production-ready Python script for the user topic.
Ensure robust error handling, inline comments, clean structure, and a comprehensive README.md.

You MUST return your response as a valid JSON object matching this exact schema:
{
  "filename": "script_name.py",
  "code": "FULL_PYTHON_CODE_HERE",
  "readme": "FULL_README_MARKDOWN_HERE",
  "product_title": "PUNCHY_GUMROAD_PRODUCT_TITLE",
  "product_description": "ATTRACTIVE_GUMROAD_SALES_COPY_MARKDOWN",
  "price_usd": 9
}
Do NOT wrap the output in markdown code fences like ```json. Return pure JSON.
"""

def generate_digital_asset_with_groq(topic: str) -> dict:
    """Queries Groq API using Llama 3.3 70B to generate code, docs, and sales copy."""
    if not groq_client:
        raise ValueError("Groq client not initialized. Check GROQ_API_KEY.")
        
    print(f"[*] Querying Groq API for topic: '{topic}'...")
    response = groq_client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": f"Generate a complete digital asset for topic: {topic}"}
        ],
        temperature=0.2
    )
    
    raw_content = response.choices[0].message.content
    return json.loads(raw_content)

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

    url = "[https://api.gumroad.com/v2/products](https://api.gumroad.com/v2/products)"
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
    
    filename = asset_data.get("filename", "main.py")
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
