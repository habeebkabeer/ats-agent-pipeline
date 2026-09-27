from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

class AgentState(TypedDict):
    job_description_url: str
    scraped_job_text: str
    original_resume: str
    analysis_feedback: str
    optimized_html: str

print("State dictionary successfully defined!")

# Load your secure API key from the .env file
load_dotenv()

# Initialize the Gemini LLM 
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

# --- NODE 1: The Scraper Agent ---
def scrape_job_description(state: AgentState):
    print("-> Agent 1: Scraping target job description...")
    
    # We are simulating the scraped web text for this initial structural build. 
    mock_job_text = "Requirements: Python, LangGraph, LLM Orchestration, React.js. We are looking for an AI Engineer to build autonomous agents."
    
    # Return a dictionary containing only the exact state key we want to update
    return {"scraped_job_text": mock_job_text}

# --- NODE 2: The Analyzer Agent ---
def analyze_resume_gap(state: AgentState):
    print("-> Agent 2: Analyzing resume against job requirements...")
    resume = state.get("original_resume", "")
    job_desc = state.get("scraped_job_text", "")
    
    prompt = f"""
    You are an expert technical recruiter and ATS specialist. 
    Analyze this candidate's resume against the target job description.
    Identify exactly which required skills or keywords are missing from the resume.
    
    Job Description: {job_desc}
    
    Resume: {resume}
    
    Return a concise, bulleted list of the missing keywords and skills.
    """
    
    # Trigger the Gemini API
    response = llm.invoke(prompt)
    
    return {"analysis_feedback": response.content}

# --- NODE 3: The Writer Agent ---
def write_optimized_resume(state: AgentState):
    print("-> Agent 3: Writing ATS-optimized resume HTML...")
    resume = state.get("original_resume", "")
    feedback = state.get("analysis_feedback", "")
    
    prompt = f"""
    You are an expert resume writer. 
    Update the following resume by seamlessly incorporating the missing skills identified in the feedback.
    Format the final output as clean, production-ready HTML.
    
    Original Resume: {resume}
    Missing Skills to Add: {feedback}
    
    Return ONLY the raw HTML code.
    """
    
    response = llm.invoke(prompt)
    return {"optimized_html": response.content}

# --- WIRING THE GRAPH ---
print("-> Compiling the LangGraph pipeline...")

# 1. Initialize the graph with our State dictionary
workflow = StateGraph(AgentState)

# 2. Add all our agent nodes
workflow.add_node("scraper", scrape_job_description)
workflow.add_node("analyzer", analyze_resume_gap)
workflow.add_node("writer", write_optimized_resume)

# 3. Define the flow (Edges)
workflow.add_edge(START, "scraper")
workflow.add_edge("scraper", "analyzer")
workflow.add_edge("analyzer", "writer")
workflow.add_edge("writer", END)

# 4. Compile the final application
ats_pipeline = workflow.compile()
print("-> Pipeline completely wired and ready!")

if __name__ == "__main__":
    print("\n--- Starting ATS Pipeline Test ---")
    
    # Initialize the workflow with all required state keys
    initial_state: AgentState = {
        "job_description_url": "https://example-job-board.com/ai-engineer",
        "original_resume": "Full Stack Developer transitioning into AI Engineering. 700+ days training in Python and ML workflows. Experience building AI-powered ATS resume optimization engines and dynamic web applications. Skills: Python, Django, HTML5, CSS3, JavaScript.",
        "scraped_job_text": "",
        "analysis_feedback": "",
        "optimized_html": ""
    }
    
    # Execute the LangGraph pipeline
    result = ats_pipeline.invoke(initial_state)
    
    # Display the results from our agents
    print("\n--- AGENT 2: GAP ANALYSIS ---")
    print(result.get("analysis_feedback"))
    
    print("\n--- AGENT 3: FINAL HTML OUTPUT ---")
    print(result.get("optimized_html"))