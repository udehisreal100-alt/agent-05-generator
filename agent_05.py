import os
import json
import re
import zipfile
import smtplib
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from groq import Groq
from dotenv import load_dotenv

# Load local .env file if available
load_dotenv()

# ---------------------------------------------------------------------------
# 1. Environment & API Credentials Configuration
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GUMROAD_ACCESS_TOKEN = os.environ.get("GUMROAD_ACCESS_TOKEN")
LEMONSQUEEZY_API_KEY = os.environ.get("LEMONSQUEEZY_API_KEY")
LEMONSQUEEZY_STORE_ID = os.environ.get("LEMONSQUEEZY_STORE_ID")

# SMTP Email Credentials (Step 4)
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = os.environ.get("SMTP_PORT", "587")
SENDER_EMAIL = os.environ.get("SENDER_EMAIL")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD")
RECIPIENT_EMAIL = os.environ.get("RECIPIENT_EMAIL")

if not GROQ_API_KEY:
    print("[!] Warning: GROQ_API_KEY environment variable is missing.")

if not GUMROAD_ACCESS_TOKEN:
    print("[!] Notice: GUMROAD_ACCESS_TOKEN missing. Gumroad publishing will be skipped.")

if not LEMONSQUEEZY_API_KEY or not LEMONSQUEEZY_STORE_ID:
    print("[!] Notice: LEMONSQUEEZY credentials missing. Lemon Squeezy publishing will be skipped.")

if not SENDER_EMAIL or not SENDER_PASSWORD or not RECIPIENT_EMAIL:
    print("[!] Notice: SMTP Email credentials missing. Email summary notifications will be skipped.")

# Initialize Groq Client
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ---------------------------------------------------------------------------
# 2. Dynamic Topic Discovery Engine (Step 1)
# ---------------------------------------------------------------------------
def fetch_google_trends(geo: str = "US", count: int = 5) -> list[str]:
    """Fetches real-time search trends from Google Trends RSS feed."""
    url = f"https://trends.google.com/trending/rss?geo={geo}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    try:
        response = requests.get(url, headers=headers, timeout=8)
        if response.status_code == 200:
            root = ET.fromstring(response.text)
            titles = []
            for item in root.findall(".//item"):
                title_elem = item.find("title")
                if title_elem is not None and title_elem.text:
                    titles.append(title_elem.text.strip())
                if len(titles) >= count:
                    break
            return titles
    except Exception as e:
        print(f"[!] Could not fetch Google Trends RSS: {e}")
        
    return []

def get_target_topic() -> str:
    """
    Dynamically discovers and synthesizes a high-demand digital asset topic.
    Combines live Google Trends with Groq LLM synthesis, falling back gracefully if offline.
    """
    print("[*] Discovering target topic via Google Trends & Groq...")
    raw_trends = fetch_google_trends(geo="US", count=5)
    
    if raw_trends:
        print(f"[✔] Retrieved live search trends: {', '.join(raw_trends)}")
    else:
        print("[!] No live trends fetched. Using default topic domain context.")

    if groq_client:
        prompt = (
            "Select or generate a high-value, practical Python code micro-SaaS or developer tool topic. "
            f"Contextual real-time query trends: {', '.join(raw_trends) if raw_trends else 'FastAPI, Supabase, Rate Limiting, AI Agents'}. "
            "Return ONLY a single concise phrase describing the Python code asset topic (e.g., 'FastAPI middleware for rate-limiting and Supabase authentication logging'). "
            "Do not include quotes, markdown formatting, or introductory commentary."
        )
        try:
            res = groq_client.chat.completions.create(
                model="qwen/qwen3.8-27b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=60
            )
            topic = res.choices[0].message.content.strip().strip('"').strip("'")
            if topic:
                return topic
        except Exception as e:
            print(f"[!] Groq topic selection error: {e}. Falling back to default.")

    return "FastAPI middleware for rate-limiting and Supabase authentication logging"

# ---------------------------------------------------------------------------
# 3. Groq Generation Module (Step 2: Universal Multi-Format Asset Generator)
# ---------------------------------------------------------------------------
SYSTEM_INSTRUCTION = """
You are an expert digital product creator and monetizer.
Your task is to generate a complete, high-value digital product asset based on the given topic.

Select the optimal digital asset format for the topic:
1. 'PROMPT_PACK': Complete AI system prompts or prompt workflows (.json or .md).
2. 'GUIDE_EBOOK': Comprehensive strategic guide, framework, or playbook (.md).
3. 'CHEAT_SHEET': Operational cheat sheet, checklist, or reference card (.md or .csv).
4. 'SOFTWARE_TOOL': Production-ready Python script, API connector, or utility (.py).

You MUST return your response as a single, valid JSON object matching this exact schema:
{
  "asset_type": "PROMPT_PACK" | "GUIDE_EBOOK" | "CHEAT_SHEET" | "SOFTWARE_TOOL",
  "product_title": "PUNCHY_PRODUCT_TITLE",
  "product_description": "SHORT_SALES_COPY_MARKDOWN",
  "price_usd": 9,
  "files": [
    {
      "filename": "primary_asset.ext",
      "content": "FULL_PRODUCT_CONTENT_HERE"
    },
    {
      "filename": "README.md",
      "content": "CONCISE_README_MARKDOWN_HERE"
    }
  ]
}

CRITICAL JSON ESCAPING RULES:
1. Output strictly valid JSON.
2. Escape all internal double quotes inside string values as \\\".
3. Represent line breaks in code and docs with \\n.
"""

def generate_digital_asset_with_groq(topic: str) -> dict:
    """Queries Groq API using qwen/qwen3.8-27b to generate multi-format universal digital assets."""
    if not groq_client:
        raise ValueError("Groq client not initialized. Check GROQ_API_KEY.")
        
    print(f"[*] Querying Groq API for topic: '{topic}'...")
    
    response = groq_client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": f"Generate a complete digital asset for topic: {topic}"}
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
        max_tokens=2500
    )
    
    raw_content = response.choices[0].message.content.strip()
    
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
def create_asset_zip(files: list[dict], zip_output_path: str):
    """Packages all generated product files into a compressed ZIP archive."""
    with zipfile.ZipFile(zip_output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for f in files:
            filename = f.get("filename", "asset.txt")
            content = f.get("content", "")
            zipf.writestr(filename, content)
    print(f"[✔] Package created successfully: {zip_output_path}")

# ---------------------------------------------------------------------------
# 5. Multi-Storefront Publishing Hub (Step 3)
# ---------------------------------------------------------------------------
def publish_to_gumroad(title: str, description: str, price_usd: int, zip_file_path: str) -> dict | None:
    """Publishes the generated product and uploads the ZIP payload to Gumroad via REST API."""
    if not GUMROAD_ACCESS_TOKEN:
        print("[✘] Skipping Gumroad: GUMROAD_ACCESS_TOKEN not set.")
        return None

    url = "https://api.gumroad.com/v2/products"
    payload = {
        "access_token": GUMROAD_ACCESS_TOKEN,
        "name": title,
        "price": price_usd * 100,  # USD to cents ($9 = 900)
        "description": description,
        "customizable_price": "false",
    }
    
    print(f"[*] Publishing '{title}' (${price_usd}) to Gumroad...")
    try:
        with open(zip_file_path, "rb") as f:
            files = {"file": (os.path.basename(zip_file_path), f, "application/zip")}
            response = requests.post(url, data=payload, files=files, timeout=30)
            
        if response.status_code in (200, 201):
            res_data = response.json()
            product = res_data.get("product", {})
            
            product_id = product.get("id")
            product_url = (
                product.get("short_url") 
                or product.get("url") 
                or (f"https://gumroad.com/l/{product_id}" if product_id else "N/A")
            )
            print(f"[✔] [Gumroad] Listing Success! URL: {product_url}")
            return {"platform": "Gumroad", "status": "success", "url": product_url}
        else:
            print(f"[✘] [Gumroad] Error ({response.status_code}): {response.text}")
            return {"platform": "Gumroad", "status": "failed", "error": response.text}
    except Exception as e:
        print(f"[✘] [Gumroad] Exception occurred: {e}")
        return {"platform": "Gumroad", "status": "error", "error": str(e)}

def publish_to_lemonsqueezy(title: str, description: str, price_usd: int) -> dict | None:
    """Publishes the generated product listing to Lemon Squeezy via JSON:API v1."""
    if not LEMONSQUEEZY_API_KEY or not LEMONSQUEEZY_STORE_ID:
        print("[✘] Skipping Lemon Squeezy: Credentials missing.")
        return None

    url = "https://api.lemonsqueezy.com/v1/products"
    headers = {
        "Accept": "application/vnd.api+json",
        "Content-Type": "application/vnd.api+json",
        "Authorization": f"Bearer {LEMONSQUEEZY_API_KEY}"
    }
    
    payload = {
        "data": {
            "type": "products",
            "attributes": {
                "name": title,
                "description": description,
                "price": price_usd * 100,  # USD to cents ($9 = 900)
            },
            "relationships": {
                "store": {
                    "data": {
                        "type": "stores",
                        "id": str(LEMONSQUEEZY_STORE_ID)
                    }
                }
            }
        }
    }

    print(f"[*] Publishing '{title}' (${price_usd}) to Lemon Squeezy...")
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code in (200, 201):
            res_data = response.json()
            product_attrs = res_data.get("data", {}).get("attributes", {})
            product_url = product_attrs.get("buy_now_url") or "https://app.lemonsqueezy.com/products"
            print(f"[✔] [Lemon Squeezy] Listing Success! URL: {product_url}")
            return {"platform": "Lemon Squeezy", "status": "success", "url": product_url}
        else:
            print(f"[✘] [Lemon Squeezy] Error ({response.status_code}): {response.text}")
            return {"platform": "Lemon Squeezy", "status": "failed", "error": response.text}
    except Exception as e:
        print(f"[✘] [Lemon Squeezy] Exception occurred: {e}")
        return {"platform": "Lemon Squeezy", "status": "error", "error": str(e)}

# ---------------------------------------------------------------------------
# 6. SMTP Email Reporter Module (Step 4)
# ---------------------------------------------------------------------------
def send_email_report(entry_data: dict):
    """Sends an automated HTML email report summarizing the agent's execution run."""
    if not SENDER_EMAIL or not SENDER_PASSWORD or not RECIPIENT_EMAIL:
        print("[!] Notice: SMTP credentials incomplete. Skipping email summary dispatch.")
        return

    print(f"[*] Sending execution summary report to {RECIPIENT_EMAIL}...")

    subject = f"🚀 Agent #05 Execution Report: {entry_data.get('product_title')}"

    storefronts_info = ""
    for platform, data in entry_data.get("storefronts", {}).items():
        status = data.get("status")
        url = data.get("url", "N/A")
        if status == "success":
            storefronts_info += f"<li><b>{platform.title()}:</b> <a href='{url}'>{url}</a></li>"
        else:
            storefronts_info += f"<li><b>{platform.title()}:</b> Failed ({data.get('error', 'Unknown error')})</li>"

    if not storefronts_info:
        storefronts_info = "<li>No storefronts were active or configured for this run.</li>"

    html_body = f"""
    <html>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
        <h2 style="color: #2c3e50;">🤖 Agent #05 Pipeline Execution Report</h2>
        <p><b>Timestamp:</b> {entry_data.get('timestamp')}</p>
        <p><b>Target Topic:</b> {entry_data.get('topic')}</p>
        
        <hr style="border: 0; border-top: 1px solid #eee;"/>
        
        <h3>📦 Product Details</h3>
        <ul>
            <li><b>Title:</b> {entry_data.get('product_title')}</li>
            <li><b>Archetype:</b> {entry_data.get('asset_type')}</li>
            <li><b>Listing Price:</b> ${entry_data.get('price_usd')}</li>
        </ul>
        
        <h3>🏪 Storefront Publishing Results</h3>
        <ul>
            {storefronts_info}
        </ul>
        
        <hr style="border: 0; border-top: 1px solid #eee;"/>
        <p style="font-size: 0.8em; color: #777;">Automated report generated by Agent #05 Publishing Engine.</p>
    </body>
    </html>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECIPIENT_EMAIL
    msg.attach(MIMEText(html_body, "html"))

    try:
        port = int(SMTP_PORT)
        with smtplib.SMTP(SMTP_SERVER, port, timeout=15) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, [RECIPIENT_EMAIL], msg.as_string())
        print("[✔] Email summary report sent successfully!")
    except Exception as e:
        print(f"[✘] Failed to send email report: {e}")

# ---------------------------------------------------------------------------
# 7. Persistent Local Catalog Module
# ---------------------------------------------------------------------------
def log_to_catalog(entry_data: dict, catalog_file: str = "digital_asset_catalog.json"):
    """Appends newly generated product details and listing links into a persistent local JSON database."""
    catalog = []
    if os.path.exists(catalog_file):
        try:
            with open(catalog_file, "r", encoding="utf-8") as f:
                catalog = json.load(f)
        except Exception as e:
            print(f"[!] Notice: Could not parse existing catalog, initializing new one ({e}).")
            catalog = []
            
    catalog.append(entry_data)
    
    with open(catalog_file, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)
        
    print(f"[✔] Persistent Asset Catalog updated: {catalog_file}")

# ---------------------------------------------------------------------------
# 8. Pipeline Orchestrator
# ---------------------------------------------------------------------------
def run_agent_pipeline():
    print("==================================================")
    print("      AGENT #05: AUTOMATED DIGITAL ASSET PIPELINE")
    print("==================================================")
    
    # Step 1: Trend Discovery
    topic = get_target_topic()
    print(f"[*] Selected Target Topic: {topic}")
    
    # Step 2: Product Generation
    asset_data = generate_digital_asset_with_groq(topic)
    
    asset_type = asset_data.get("asset_type", "GENERIC")
    title = asset_data.get("product_title", "Automated Digital Asset")
    description = asset_data.get("product_description", "")
    price = asset_data.get("price_usd", 9)
    files = asset_data.get("files", [])
    
    if not files:
        filename = asset_data.get("filename", "middleware.py")
        code = asset_data.get("code", "")
        readme = asset_data.get("readme", "")
        files = [
            {"filename": filename, "content": code},
            {"filename": "README.md", "content": readme}
        ]
        
    print(f"[✔] Asset Archetype: [{asset_type}]")
    print(f"[*] Package File Manifest ({len(files)} files):")
    file_manifest = []
    for f in files:
        fname = f.get('filename', 'unnamed')
        flen = len(f.get('content', ''))
        file_manifest.append({"filename": fname, "length": flen})
        print(f"    - {fname} ({flen} chars)")
    
    zip_path = f"payload_{os.urandom(3).hex()}.zip"
    create_asset_zip(files, zip_path)
    
    # Step 3: Multi-Storefront Publishing
    publish_results = {}
    try:
        gumroad_res = publish_to_gumroad(title, description, price, zip_path)
        if gumroad_res:
            publish_results["gumroad"] = gumroad_res
            
        lemon_res = publish_to_lemonsqueezy(title, description, price)
        if lemon_res:
            publish_results["lemonsqueezy"] = lemon_res
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print("[*] Temporary ZIP file cleaned up.")

    # Record catalog entry
    catalog_entry = {
        "timestamp": datetime.now().isoformat(),
        "topic": topic,
        "product_title": title,
        "asset_type": asset_type,
        "price_usd": price,
        "files_manifest": file_manifest,
        "storefronts": publish_results
    }
    
    log_to_catalog(catalog_entry)
    
    # Step 4: Dispatch Email Summary Notification
    send_email_report(catalog_entry)
            
    print("==================================================")
    print("      PIPELINE EXECUTION COMPLETE")
    print("==================================================")

if __name__ == "__main__":
    run_agent_pipeline()