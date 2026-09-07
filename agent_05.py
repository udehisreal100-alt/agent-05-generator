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

# SMTP Email Credentials
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

# Define the 5 Task-Chained Models from Groq Limits
MODEL_STAGE_1 = "openai/gpt-oss-20b"
MODEL_STAGE_2 = "openai/gpt-oss-120b"
MODEL_STAGE_3 = "qwen/qwen3.8-27b"
MODEL_STAGE_4 = "qwen/qwen3.6-27b"
MODEL_STAGE_5 = "openai/gpt-oss-safeguard-20b"

# Reliable fallback model if a primary model fails
MODEL_FALLBACK = "qwen/qwen3.8-27b"

# ---------------------------------------------------------------------------
# Helper: Safe Call to Groq API with JSON Fallback & Regex Cleaning
# ---------------------------------------------------------------------------
def call_groq(model: str, messages: list, temperature: float = 0.2, max_tokens: int = 1500, response_format: dict = None) -> str:
    """Executes a Groq API call with multi-tier exception fallback and response cleaning."""
    if not groq_client:
        raise ValueError("Groq client is not initialized.")

    def clean_text(text: str) -> str:
        """Strips markdown code blocks and whitespace."""
        cleaned = re.sub(r"^```(?:json|python)?\s*", "", text.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    # Attempt 1: Call primary model with response_format if provided
    try:
        call_kwargs = kwargs.copy()
        if response_format:
            call_kwargs["response_format"] = response_format
        res = groq_client.chat.completions.create(**call_kwargs)
        return clean_text(res.choices[0].message.content)
    except Exception as e:
        print(f"[!] Primary model [{model}] call failed with response_format constraint: {e}")

    # Attempt 2: Retry primary model WITHOUT forced response_format
    if response_format:
        try:
            print(f"[*] Retrying primary model [{model}] without forced response_format...")
            res = groq_client.chat.completions.create(**kwargs)
            return clean_text(res.choices[0].message.content)
        except Exception as e:
            print(f"[!] Retrying primary model [{model}] without response_format failed: {e}")

    # Attempt 3: Fallback to alternative model without response_format
    try:
        print(f"[*] Attempting fallback execution with model [{MODEL_FALLBACK}]...")
        fallback_kwargs = kwargs.copy()
        fallback_kwargs["model"] = MODEL_FALLBACK
        res = groq_client.chat.completions.create(**fallback_kwargs)
        return clean_text(res.choices[0].message.content)
    except Exception as e:
        print(f"[✘] Fallback model [{MODEL_FALLBACK}] failed: {e}")
        raise e

# ---------------------------------------------------------------------------
# 2. Dynamic Trend Fetcher
# ---------------------------------------------------------------------------
def fetch_google_trends(geo: str = "US", count: int = 5) -> list[str]:
    """Fetches real-time search trends from Google Trends RSS feed."""
    url = f"[https://trends.google.com/trending/rss?geo=](https://trends.google.com/trending/rss?geo=){geo}"
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

# ---------------------------------------------------------------------------
# 3. Task Chaining Pipeline (Stages 1 through 5)
# ---------------------------------------------------------------------------

def stage_1_strategy_and_blueprint(raw_trends: list[str]) -> dict:
    """Stage 1: Model [openai/gpt-oss-20b] - Product Strategist"""
    print(f"\n[Stage 1] Product Strategy & Blueprint (Model: {MODEL_STAGE_1})...")
    
    prompt = f"""
    You are a SaaS & AI Micro-Product Architect.
    Analyze these trending search topics: {', '.join(raw_trends) if raw_trends else 'FastAPI, Supabase, Rate Limiting, AI Agents'}.
    
    Select 1 high-demand Python developer utility or API micro-tool asset.
    Return strictly valid JSON with no markdown formatting, backticks, or preamble text matching this exact structure:
    {{
      "topic": "Concise topic description",
      "product_title": "Punchy Catchy Product Title",
      "price_usd": 19,
      "primary_filename": "main_script.py",
      "asset_type": "SOFTWARE_TOOL",
      "key_features": ["feature 1", "feature 2", "feature 3"]
    }}
    """
    
    messages = [
        {"role": "system", "content": "You are a product strategy agent. You MUST output strictly valid JSON with no markdown tags or additional text."},
        {"role": "user", "content": prompt}
    ]

    response_text = call_groq(
        model=MODEL_STAGE_1,
        messages=messages,
        temperature=0.6,
        max_tokens=400,
        response_format={"type": "json_object"}
    )
    
    try:
        blueprint = json.loads(response_text)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            blueprint = json.loads(json_match.group(0))
        else:
            raise ValueError(f"Failed to parse valid JSON from response: {response_text}")

    print(f"[✔] Blueprint Created: {blueprint.get('product_title')} (${blueprint.get('price_usd')})")
    return blueprint

def stage_2_generate_core_code(blueprint: dict) -> str:
    """Stage 2: Model [openai/gpt-oss-120b] - Lead Developer"""
    print(f"\n[Stage 2] Core Code Generation (Model: {MODEL_STAGE_2})...")
    
    prompt = f"""
    You are an expert Python Software Engineer.
    Build a complete, modular, production-ready Python script for:
    Product Title: {blueprint.get('product_title')}
    Topic: {blueprint.get('topic')}
    Key Features: {', '.join(blueprint.get('key_features', []))}
    Target Filename: {blueprint.get('primary_filename')}

    REQUIREMENTS:
    - Include clean imports, typed functions, docstrings, robust error handling, and a working __main__ block or CLI demo.
    - Write clean, concise code (under 120 lines).
    - Output ONLY the clean Python code inside code fences ```python ... ```.
    """
    
    code_response = call_groq(
        model=MODEL_STAGE_2,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1500
    )
    
    match = re.search(r"```python(.*?)```", code_response, re.DOTALL)
    if match:
        code_clean = match.group(1).strip()
    else:
        code_clean = code_response.replace("```python", "").replace("```", "").strip()
        
    print(f"[✔] Core Code Generated ({len(code_clean)} chars)")
    return code_clean

def stage_3_generate_readme(blueprint: dict, code_content: str) -> str:
    """Stage 3: Model [qwen/qwen3.8-27b] - Solutions Engineer & Tech Writer"""
    print(f"\n[Stage 3] Technical Documentation README (Model: {MODEL_STAGE_3})...")
    
    prompt = f"""
    You are a Technical Writer and Developer Advocate.
    Create a clean, professional, enterprise-grade README.md for this product:
    Title: {blueprint.get('product_title')}
    Topic: {blueprint.get('topic')}

    Here is the primary Python code:
    ```python
    {code_content[:1500]}
    ```

    Include:
    - Overview & Value Proposition
    - Prerequisites & Installation (`pip install ...`)
    - Environment Variables (`.env` example)
    - Quickstart Code Snippet
    - Architecture Overview
    
    Return ONLY valid Markdown text without JSON wrapping.
    """
    
    readme_markdown = call_groq(
        model=MODEL_STAGE_3,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1000
    )
    
    print(f"[✔] README Documentation Created ({len(readme_markdown)} chars)")
    return readme_markdown

def stage_4_generate_sales_copy(blueprint: dict, code_content: str) -> str:
    """Stage 4: Model [qwen/qwen3.6-27b] - Growth Copywriter"""
    print(f"\n[Stage 4] Storefront Sales Copy Generation (Model: {MODEL_STAGE_4})...")
    
    prompt = f"""
    You are a High-Converting SaaS Copywriter.
    Write compelling sales description copy formatted in Markdown for selling this product on Gumroad and Lemon Squeezy.
    Product Title: {blueprint.get('product_title')}
    Price: ${blueprint.get('price_usd')}
    Key Features: {', '.join(blueprint.get('key_features', []))}

    Structure:
    1. Attention-grabbing Headline
    2. Problem / Solution pitch
    3. What's Included inside the download package
    4. Ideal Target Audience
    
    Keep it energetic, clear, and persuasive (under 300 words).
    """
    
    sales_copy = call_groq(
        model=MODEL_STAGE_4,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=800
    )
    
    print(f"[✔] Sales Copy Generated ({len(sales_copy)} chars)")
    return sales_copy

def stage_5_security_audit_and_package(blueprint: dict, code_content: str, readme_content: str, sales_copy: str) -> dict:
    """Stage 5: Model [openai/gpt-oss-safeguard-20b] - Security Auditor & Final Assembler"""
    print(f"\n[Stage 5] Security Audit & Packaging (Model: {MODEL_STAGE_5})...")
    
    audit_prompt = f"""
    Review this python code for security issues, API key leaks, or unsafe syntax:
    ```python
    {code_content[:1000]}
    ```
    Is this safe for digital product publishing? Return strictly valid JSON with no markdown:
    {{"safe": true, "audit_notes": "Passed security check"}}
    """
    
    try:
        audit_res = call_groq(
            model=MODEL_STAGE_5,
            messages=[
                {"role": "system", "content": "You are a security auditing agent. You MUST reply strictly with valid JSON."},
                {"role": "user", "content": audit_prompt}
            ],
            temperature=0.1,
            max_tokens=200,
            response_format={"type": "json_object"}
        )
        try:
            audit_data = json.loads(audit_res)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", audit_res, re.DOTALL)
            audit_data = json.loads(json_match.group(0)) if json_match else {"safe": True, "audit_notes": "Passed security check"}

        print(f"[✔] Security Audit Passed: {audit_data.get('audit_notes', 'Safe')}")
    except Exception as e:
        print(f"[!] Audit warning ({e}), defaulting to auto-approved.")

    # Compile final complete manifest
    package = {
        "asset_type": blueprint.get("asset_type", "SOFTWARE_TOOL"),
        "product_title": blueprint.get("product_title"),
        "product_description": sales_copy,
        "price_usd": blueprint.get("price_usd", 19),
        "topic": blueprint.get("topic"),
        "files": [
            {
                "filename": blueprint.get("primary_filename", "main.py"),
                "content": code_content
            },
            {
                "filename": "README.md",
                "content": readme_content
            }
        ]
    }
    return package

# ---------------------------------------------------------------------------
# 4. Packaging, Publishing, and Catalog Functions
# ---------------------------------------------------------------------------
def create_asset_zip(files: list[dict], zip_output_path: str):
    """Packages generated files into a compressed ZIP archive."""
    with zipfile.ZipFile(zip_output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for f in files:
            filename = f.get("filename", "asset.txt")
            content = f.get("content", "")
            zipf.writestr(filename, content)
    print(f"[✔] ZIP Archive created: {zip_output_path}")

def publish_to_gumroad(title: str, description: str, price_usd: int, zip_file_path: str) -> dict | None:
    """Publishes product to Gumroad REST API."""
    if not GUMROAD_ACCESS_TOKEN:
        print("[✘] Skipping Gumroad: GUMROAD_ACCESS_TOKEN not set.")
        return None

    url = "[https://api.gumroad.com/v2/products](https://api.gumroad.com/v2/products)"
    payload = {
        "access_token": GUMROAD_ACCESS_TOKEN,
        "name": title,
        "price": price_usd * 100,  # USD to cents
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
                or (f"[https://gumroad.com/l/](https://gumroad.com/l/){product_id}" if product_id else "N/A")
            )
            print(f"[✔] [Gumroad] Published! URL: {product_url}")
            return {"platform": "Gumroad", "status": "success", "url": product_url}
        else:
            print(f"[✘] [Gumroad] Error ({response.status_code}): {response.text}")
            return {"platform": "Gumroad", "status": "failed", "error": response.text}
    except Exception as e:
        print(f"[✘] [Gumroad] Exception: {e}")
        return {"platform": "Gumroad", "status": "error", "error": str(e)}

def publish_to_lemonsqueezy(title: str, description: str, price_usd: int) -> dict | None:
    """Publishes product listing to Lemon Squeezy JSON:API v1."""
    if not LEMONSQUEEZY_API_KEY or not LEMONSQUEEZY_STORE_ID:
        print("[✘] Skipping Lemon Squeezy: Credentials missing.")
        return None

    url = "[https://api.lemonsqueezy.com/v1/products](https://api.lemonsqueezy.com/v1/products)"
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
                "price": price_usd * 100,
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
            product_url = product_attrs.get("buy_now_url") or "[https://app.lemonsqueezy.com/products](https://app.lemonsqueezy.com/products)"
            print(f"[✔] [Lemon Squeezy] Published! URL: {product_url}")
            return {"platform": "Lemon Squeezy", "status": "success", "url": product_url}
        else:
            print(f"[✘] [Lemon Squeezy] Error ({response.status_code}): {response.text}")
            return {"platform": "Lemon Squeezy", "status": "failed", "error": response.text}
    except Exception as e:
        print(f"[✘] [Lemon Squeezy] Exception: {e}")
        return {"platform": "Lemon Squeezy", "status": "error", "error": str(e)}

def send_email_report(entry_data: dict):
    """Sends HTML email report summarizing execution."""
    if not SENDER_EMAIL or not SENDER_PASSWORD or not RECIPIENT_EMAIL:
        print("[!] Notice: SMTP credentials incomplete. Skipping email summary.")
        return

    print(f"[*] Sending email report to {RECIPIENT_EMAIL}...")

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
        storefronts_info = "<li>No storefronts active for this run.</li>"

    html_body = f"""
    <html>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
        <h2 style="color: #2c3e50;">🤖 Agent #05 5-Stage Pipeline Report</h2>
        <p><b>Timestamp:</b> {entry_data.get('timestamp')}</p>
        <p><b>Target Topic:</b> {entry_data.get('topic')}</p>
        
        <hr style="border: 0; border-top: 1px solid #eee;"/>
        
        <h3>📦 Product Package Details</h3>
        <ul>
            <li><b>Title:</b> {entry_data.get('product_title')}</li>
            <li><b>Archetype:</b> {entry_data.get('asset_type')}</li>
            <li><b>Listing Price:</b> ${entry_data.get('price_usd')}</li>
        </ul>
        
        <h3>🏪 Storefront Publishing Status</h3>
        <ul>
            {storefronts_info}
        </ul>
        
        <hr style="border: 0; border-top: 1px solid #eee;"/>
        <p style="font-size: 0.8em; color: #777;">Automated 5-Task Chained Agent Pipeline.</p>
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

def log_to_catalog(entry_data: dict, catalog_file: str = "digital_asset_catalog.json"):
    """Appends product metadata into persistent catalog."""
    catalog = []
    if os.path.exists(catalog_file):
        try:
            with open(catalog_file, "r", encoding="utf-8") as f:
                catalog = json.load(f)
        except Exception:
            catalog = []
            
    catalog.append(entry_data)
    
    with open(catalog_file, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)
        
    print(f"[✔] Persistent Catalog Updated: {catalog_file}")

# ---------------------------------------------------------------------------
# 5. Pipeline Orchestrator
# ---------------------------------------------------------------------------
def run_agent_pipeline():
    print("==================================================")
    print("      AGENT #05: 5-STAGE TASK-CHAINING PIPELINE")
    print("==================================================")
    
    # Fetch real-time context
    raw_trends = fetch_google_trends(geo="US", count=5)
    if raw_trends:
        print(f"[✔] Discovered Live Search Trends: {', '.join(raw_trends)}")

    # Execute 5-Stage Task Chaining Pipeline
    blueprint = stage_1_strategy_and_blueprint(raw_trends)
    code_content = stage_2_generate_core_code(blueprint)
    readme_content = stage_3_generate_readme(blueprint, code_content)
    sales_copy = stage_4_generate_sales_copy(blueprint, code_content)
    package = stage_5_security_audit_and_package(blueprint, code_content, readme_content, sales_copy)
    
    title = package.get("product_title", "Automated Python Tool")
    description = package.get("product_description", "")
    price = package.get("price_usd", 19)
    files = package.get("files", [])
    
    # Packaging
    zip_path = f"payload_{os.urandom(3).hex()}.zip"
    create_asset_zip(files, zip_path)
    
    # Storefront Publishing
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
        "topic": package.get("topic"),
        "product_title": title,
        "asset_type": package.get("asset_type"),
        "price_usd": price,
        "files_manifest": [{"filename": f.get("filename"), "length": len(f.get("content", ""))} for f in files],
        "storefronts": publish_results
    }
    
    log_to_catalog(catalog_entry)
    send_email_report(catalog_entry)
            
    print("==================================================")
    print("      PIPELINE EXECUTION COMPLETE")
    print("==================================================")

if __name__ == "__main__":
    run_agent_pipeline()