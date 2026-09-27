# Autonomous ATS & Career Tech Pipeline

A stateful, multi-agent AI pipeline built to autonomously analyze and optimize resumes against specific job descriptions. 

## Architecture
This project leverages **LangGraph** to orchestrate specialized AI agents powered by the **Google Gemini API**. 
* **Scraper Agent:** Extracts and processes target job description requirements.
* **Analyzer Agent:** Performs a gap analysis between the candidate's resume and job requirements.
* **Writer Agent:** Autonomously rewrites and outputs ATS-optimized, production-ready HTML.

## Tech Stack
* **Framework:** LangGraph, Python
* **LLM:** Google Gemini 
* **Environment:** python-dotenv

## Installation & Setup
1. Clone the repository:
   `git clone https://github.com/yourusername/ats-agent-pipeline.git`
2. Create and activate a virtual environment:
   `python -m venv venv`
3. Install dependencies:
   `pip install -r requirements.txt`
4. Create a `.env` file and add your Gemini API key:
   `GOOGLE_API_KEY=your_api_key_here`
5. Execute the multi-agent workflow:
   `python main.py`