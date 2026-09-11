import os
from datetime import datetime
import requests
from dotenv import load_dotenv
from config import CONFIG

load_dotenv()

# Set OpenRouter API Key from environment or config
API_KEY = os.getenv("OPENROUTER_API_KEY") or CONFIG.openrouter_api_key
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# 1. Fetch live pricing model list safely
models_data = {}
try:
    models_res = requests.get("https://openrouter.ai/api/v1/models", timeout=10)
    if models_res.status_code == 200:
        models_data = {
            m["id"]: m.get("pricing", {}) 
            for m in models_res.json().get("data", [])
        }
except Exception as e:
    print(f"Warning: Could not fetch pricing database ({e}). Cost estimation may default to 0.")

# 2. Execute completion request
selected_model = "openai/gpt-5.6-luna"

# Record request start time
execution_time = datetime.now()

response = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers=HEADERS,
    json={
        "model": selected_model,
        "messages": [{"role": "user", "content": "Hello world"}]
    },
    timeout=30
)

# Check if response is successful before parsing JSON
if response.status_code != 200:
    print(f"API Error: HTTP {response.status_code}")
    print(f"Response: {response.text}")
    exit(1)

try:
    data = response.json()
except Exception as e:
    print(f"Error parsing JSON response: {e}")
    print(f"Response text: {response.text}")
    exit(1)

if "usage" in data:
    model_used = data.get("model", selected_model)
    prompt_tokens = data["usage"].get("prompt_tokens", 0)
    completion_tokens = data["usage"].get("completion_tokens", 0)
    total_tokens = data["usage"].get("total_tokens", 0)

    # Retrieve pricing rates per token
    pricing = models_data.get(model_used, {})
    prompt_rate = float(pricing.get("prompt", 0) or 0)
    completion_rate = float(pricing.get("completion", 0) or 0)

    # Calculate costs locally
    prompt_cost = prompt_tokens * prompt_rate
    completion_cost = completion_tokens * completion_rate
    total_cost_usd = prompt_cost + completion_cost

    print("--- Execution Summary ---")
    print(f"Date:              {execution_time.strftime('%Y-%m-%d')}")
    print(f"Time:              {execution_time.strftime('%H:%M:%S')}")
    print(f"Model Used:        {model_used}")
    print(f"Prompt Tokens:     {prompt_tokens}")
    print(f"Completion Tokens: {completion_tokens}")
    print(f"Total Tokens:      {total_tokens}")
    print(f"Estimated Cost:    ${total_cost_usd:.8f} USD")

    # 3. Fetch cumulative key stats
    key_res = requests.get("https://openrouter.ai/api/v1/key", headers=HEADERS, timeout=10)
    if key_res.status_code == 200:
        key_info = key_res.json().get("data", {})
        print("\n--- Key Balance Info ---")
        print(f"Key Label:         {key_info.get('label', 'N/A')}")
        print(f"Total Key Spend:   ${key_info.get('usage', 0):.4f} USD")
        
        limit_remaining = key_info.get("limit_remaining")
        if limit_remaining is not None:
            print(f"Limit Remaining:   ${float(limit_remaining):.4f} USD")
        else:
            print("Limit Remaining:   Unlimited")
else:
    print("API Error: No usage data in response")
    print(f"Response data: {data}")