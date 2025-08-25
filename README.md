# Privacy Agent LLM
This is a Python agent that searches the web for public mentions of a target email address, analyzes the context with OpenAI, and generates a Markdown and PDF report about what information might be exposed. It also detects common email obfuscations like `name [at] example [dot] com`.

Script outputs include:
1. <out>/discoveries.csv (verified pages)
2. <out>/report.md (human-readable LLM responses per page)
3. <out>/report.pdf (PDF version of the report)
4. <out>/domains.json (grouping + contacts)

---
## ✨ Features
- Searches the web (via DuckDuckGo / ddgs library).
- Detects direct and obfuscated email addresses.
- Uses OpenAI LLMs (e.g., `gpt-4o-mini`) to infer:
  - What platform/service the page belongs to.
  - What public data may be tied to the email.
  - What API actions/features could expose this data.
  - Recommended next steps.
- Generates both report.md and report.pdf for readability.

---
## 🛠 Installation
1. **Clone this repo**
   ```bash
   git clone https://github.com/YOUR_USERNAME/privacy-agent-llm.git
   cd privacy-agent-llm
   ```
2. Create a virtual environment (recommended)
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Mac/Linux
   # OR
   .venv\Scripts\Activate      # Windows (PowerShell)
   ```
3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

---
## 🔑 OpenAI API Key

You’ll need your own [OpenAI API key](https://platform.openai.com/).

### Option 1: Add to `.env`  
Create a `.env` file in the project root:
```bash 
OPENAI_API_KEY=sk-your_api_key_here
```

### Option 2: Export in your terminal  
- **Mac/Linux**:
  ```bash
  export OPENAI_API_KEY="sk-your_api_key_here"
  ```
- **Windows (PowerShell)**:
  ```bash
  $env:OPENAI_API_KEY="sk-your_api_key_here"
  ```

---
## 🚀 Usage

Run the script with your target email:
```bash
python privacy_agent_llm.py --email you@example.com
```

### Arguments
- `--email` (required): The email address to search for  
- `--out`: Output directory (default: `outputs`)  
- `--max`: Max search results to fetch (default: 80)  
- `--model`: OpenAI model (default: `gpt-4o-mini`)  
- `--max-tokens`: Max tokens per LLM call (default: 800)  

---
## 📂 Outputs

After running, you’ll find:
- `outputs/discoveries.csv` → Pages where your email was found  
- `outputs/domains.json` → Grouped domains with their pages  
- `outputs/report.md` → Human-readable analysis  
- `outputs/report.pdf` → Nicely formatted PDF report  

If no results are found, the report will simply state that no public exposures were detected.

---
## ⚠️ Disclaimer

This project is for **educational and personal research purposes only**.  
It does **not** guarantee complete coverage of all public data sources, and results depend on the search engine index. Use responsibly.


