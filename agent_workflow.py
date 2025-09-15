import os
import json
import re
from typing import List, Dict, TypedDict, Annotated
from langchain.agents import AgentType, initialize_agent, Tool
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import chemrxiv
from pymupdf4llm import to_markdown

# --------------------------------------------------------------------------
# --- CONFIGURATION ---
# Set this flag to True to use Azure AI Foundry, or False to use the local model.
# If USE_AZURE is True, make sure you have a .env file with your Azure credentials.
USE_AZURE = False

# --- CUSTOMIZABLE PARAMETERS ---
# Number of most important topics to extract from the initial paper
NUM_TOPICS_TO_EXTRACT = 2
# Number of relevant papers to download and summarize (max)
NUM_PAPERS_TO_DOWNLOAD = 3
# Initial pdf file with complete path
PDF_PAPER_PATH = ''    
# --------------------------------------------------------------------------


# Initialize the LLM based on the switch
if USE_AZURE:
    print("--- Using Azure OpenAI Configuration ---")
    # Load environment variables from .env file for Azure credentials
    load_dotenv()
    
    # Check if essential environment variables are set
    if not all([os.getenv("AZURE_OPENAI_ENDPOINT"), os.getenv("AZURE_OPENAI_API_KEY"), os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")]):
        raise ValueError("Azure credentials not found in environment variables. Please create a .env file.")

    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "yyy-mm-dd"), # Default API version
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0,
        max_tokens=4024,
    )
else:
    print("--- Using Local Model Configuration ---")
    # Original local model initialization
    llm = ChatOpenAI(
        base_url="http://localhost:1234/v1",
        api_key="not-needed",
        model="local-model",
        temperature=0
    )

# Initialize ChemRxiv client
chemrxiv_client = chemrxiv.Client()

# --- STATE DEFINITION ---

class State(TypedDict):
    messages: Annotated[List, add_messages]
    initial_pdf_path: str
    initial_info: str
    key_terms: str
    search_results: List[Dict]
    paper_dois_to_download: List[str]
    downloaded_papers: List[str]
    extracted_infos: List[Dict]
    report: str
    search_attempts: int
    review_feedback: str
    revision_attempts: int
    approved: bool

# --- TOOLS & HELPERS ---

# For a more robust, cloud-native solution, the local PDF tools below could be replaced
# with a call to the Azure AI Document Intelligence service 
def pdf_extraction_tool(pdf_path: str) -> str:
    """Extracts markdown text from ONLY THE FIRST PAGE of a PDF."""
    try:
        clean_path = pdf_path.strip().strip("'\"")
        print(f"   [Tool] Extracting FIRST PAGE ONLY from: {clean_path}")
        return to_markdown(clean_path, pages=[0])
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"

def full_pdf_extraction_tool(pdf_path: str) -> str:
    """Extracts all markdown text from a PDF document."""
    try:
        clean_path = pdf_path.strip().strip("'\"")
        print(f"   [Tool] Extracting FULL TEXT from: {clean_path}")
        return to_markdown(clean_path)
    except Exception as e:
        return f"Error extracting full PDF: {str(e)}"

def strip_references(text: str) -> str:
    """Removes the references/bibliography section from the end of a paper."""
    parts = re.split(r'\n\s*(?:References|Bibliography)\s*\n', text, maxsplit=1, flags=re.IGNORECASE)
    return parts[0]

def distill_long_text(text: str) -> str:
    """For very long texts, extracts the first two sentences of each paragraph."""
    print("   [Helper] Distilling long text...")
    paragraphs = text.split('\n\n')
    distilled_paragraphs = [' '.join(re.split(r'(?<=[.!?])\s+', p.strip())[:2]) for p in paragraphs if p.strip()]
    return '\n\n'.join(distilled_paragraphs)

def chemrxiv_search_tool(query: str) -> List[Dict]:
    """Searches ChemRxiv and returns a list of paper dictionaries."""
    print(f"   [Tool] Searching ChemRxiv for: '{query}'")
    try:
        search = chemrxiv.Search(term=query, limit=5, sort=chemrxiv.SortCriterion.PUBLISHED_DATE_DESC)
        return [{"title": p.title, "abstract": p.abstract, "doi": p.doi} for p in chemrxiv_client.results(search)]
    except Exception:
        return []

def chemrxiv_download_tool(doi: str, directory: str = "downloads") -> str:
    """Downloads a PDF from ChemRxiv given a DOI."""
    print(f"   [Tool] Attempting download for DOI: {doi}")
    try:
        os.makedirs(directory, exist_ok=True)
        paper = chemrxiv_client.item_by_doi(doi)
        filename = os.path.join(directory, f"{paper.doi.replace('/', '_')}.pdf")
        paper.download_pdf(filename=filename)
        return f"Success: {filename}"
    except Exception as e:
        return f"Error downloading {doi}: {str(e)}"

def write_file_tool(file_path: str, content: str) -> str:
    """Deterministic function to write content to a file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote content to {file_path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"


# --- WORKFLOW NODES ---

def extract_initial_info(state: State):
    print(f"--- Node: extract_initial_info ---")
    # Call the tool directly. It's more efficient and explicit.
    pdf_path = state['initial_pdf_path']
    info = pdf_extraction_tool(pdf_path)
    return {"initial_info": info}

def extract_key_terms(state: State):
    print(f"--- Node: extract_key_terms ---")
    prompt = f"Extract the {NUM_TOPICS_TO_EXTRACT} most important topics from this text. Return a comma-separated list. Text: {state['initial_info']}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"key_terms": response.content.strip()}

def search_for_papers(state: State):
    print(f"--- Node: search_for_papers (Attempt #{state['search_attempts'] + 1}) ---")
    all_results, seen_dois = [], set()
    for term in state['key_terms'].split(','):
        term = term.strip().lower()
        if not term: continue
        for result in chemrxiv_search_tool(term):
            if result.get('doi') and result['doi'] not in seen_dois:
                all_results.append(result)
                seen_dois.add(result['doi'])
    return {"search_results": all_results}

def filter_relevant_papers(state: State):
    print(f"--- Node: filter_relevant_papers ---")
    if not state['search_results']: return {}
    current_dois = set(state.get('paper_dois_to_download', []))
    for paper in state['search_results']:
        if paper['doi'] in current_dois: continue
        prompt = f"Is this paper somewhat related to '{state['key_terms']}'?\nTitle: {paper['title']}\nAbstract: {paper['abstract']}\nAnswer ONLY with YES or NO."
        response = llm.invoke([HumanMessage(content=prompt)])
        if 'yes' in response.content.strip().lower():
            current_dois.add(paper['doi'])
    return {"paper_dois_to_download": list(current_dois)}

def generalize_search_terms(state: State):
    print(f"--- Node: generalize_search_terms ---")
    prompt = f"The search terms '{state['key_terms']}' found too few papers. Generate two new, broader, related terms. Return ONLY a comma-separated list."
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"key_terms": response.content.strip(), "search_attempts": state['search_attempts'] + 1}

def download_papers(state: State):
    print(f"--- Node: download_papers ---")
    dois_to_download = state['paper_dois_to_download'][:NUM_PAPERS_TO_DOWNLOAD]
    # Create a quick lookup map from DOI to Title from the search results
    doi_to_title_map = {p['doi']: p['title'] for p in state['search_results']}
    
    downloaded_files_info = [] 
    if not dois_to_download: return {"downloaded_papers": []}
    
    for doi in dois_to_download:
        result = chemrxiv_download_tool(doi)
        if result.startswith("Success"):
            file_path = result.split(": ")[1]
            paper_title = doi_to_title_map.get(doi, "Unknown Title") # Fallback
            downloaded_files_info.append({"path": file_path, "title": paper_title, "doi": doi})
            
    return {"downloaded_papers": downloaded_files_info}
    
    
def summarize_papers_node(state: State):
    print(f"--- Node: summarize_papers_node ---")
    if not state['downloaded_papers']: return {"extracted_infos": []}    
    summaries = []
    # paper_info is now a dictionary
    for paper_info in state['downloaded_papers']:
        paper_path = paper_info['path']
        print(f"--- Summarizing: {paper_path} ({paper_info['title']}) ---")
        full_text = full_pdf_extraction_tool(paper_path)
        if "Error" in full_text:
            summaries.append({
                "path": paper_path,
                "title": paper_info['title'],
                "content": "Error: Could not extract full text."
            })
            continue
        text_for_summarization = strip_references(full_text)
        word_count = len(text_for_summarization.split())
        if word_count > 3000:
            text_for_summarization = distill_long_text(text_for_summarization)
        prompt = f"""You are an expert scientific analyst. Your task is to write a high-quality, comprehensive summary (around 300-400 words) of the following scientific paper text.

        Focus ONLY on the core scientific content. Pay close attention to the following sections if they are present:
        - **Abstract**: The overall summary.
        - **Introduction**: The background, problem statement, and goals.
        - **Methods/Methodology**: How the research was conducted.
        - **Results**: The key findings and data.
        - **Discussion/Conclusion**: The interpretation of the results and their implications.

        **Crucially, you must IGNORE any 'References' or 'Bibliography' sections.** Base your summary on the paper's own research, not the works it cites.

        TEXT:
        {text_for_summarization}
        """
        summary_response = llm.invoke([HumanMessage(content=prompt)])
        summaries.append({
            "path": paper_path,
            "title": paper_info['title'],
            "content": summary_response.content
        })        
    return {"extracted_infos": summaries}
    
    
def compile_report(state: State):
    print(f"--- Node: compile_report (Revision Attempt #{state['revision_attempts']}) ---")
    
    # 1. Build the structured data for the report
    report_data = f"## Initial Paper Analysis\n{state['initial_info']}\n\n"
    if state['extracted_infos']:
        report_data += f"## Analysis of {len(state['extracted_infos'])} Related Papers (Summaries)\n"
        for i, info in enumerate(state['extracted_infos']):
            # Use the paper's title instead of its path
            report_data += f"### Summary of '{info['title']}'\n{info['content']}\n\n"
    else:
       report_data += "## No Relevant Related Papers Were Found.\n"

    # 2. Use a focused LLM call to generate the creative parts
    if state.get('review_feedback'):
        print("Revising report based on feedback...")
        synthesis_prompt = (
            f"You are a scientific editor. Please revise and improve the 'Introduction', 'Analysis' and 'Conclusion' for the report below based on the provided feedback. "
            f"Return ONLY the new 'Introduction', 'Analysis' and 'Conclusion' sections, each under its respective markdown heading (e.g., '# Introduction').\n\n"
            f"FEEDBACK:\n{state['review_feedback']}\n\n"
            f"REPORT DATA:\n{report_data}"
        )
    else:
        print("Compiling initial report...")
        synthesis_prompt = (
            f"You are a scientific editor. Please write a high-level 'Introduction', 'Analysis' and 'Conclusion' for the following collection of research summaries about '{state['key_terms']}'. "
            f"The introduction should set the stage for the topic, the analysis should give a general overview plus interconnections and commons themes between the summaries and the conclusion should synthesize the key findings from all papers. "
            f"Return ONLY the 'Introduction', 'Analysis' and 'Conclusion' sections, each under its respective markdown heading (e.g., '# Introduction').\n\n"
            f"RESEARCH SUMMARIES:\n{report_data}"
        )

    print("Asking LLM to synthesize Introduction, Analysis and Conclusion...")
    synthesis_response = llm.invoke([HumanMessage(content=synthesis_prompt)]).content

    # 3. Assemble the final report in Python
    final_report_content = (
        f"# Literature Review: {state['key_terms']}\n\n"
        f"{synthesis_response}\n\n"  # This contains the LLM-generated Intro Analysis and Conclusion
        f"{report_data}"             # This contains the structured data
    )

    # 4. Save the report deterministically
    save_result = write_file_tool("final_report.md", final_report_content)
    print(save_result)
    
    return {"report": save_result}

def review_report_node(state: State):
    print(f"--- Node: review_report_node ---")
    try:
        with open("final_report.md", 'r', encoding='utf-8') as f:
            report_content = f.read()
    except FileNotFoundError:
        return {"approved": False, "review_feedback": "Agent failed to write report file."}
    prompt = f"""Review the report based on structure (intro, analysis, conclusions) and length (4 paragraphs, 4 sentences each). Provide concrete suggestions. End your review with a single word on a new line: APPROVED or NEEDS_REVISION.\n\nReport:\n{report_content}"""
    response_content = llm.invoke([HumanMessage(content=prompt)]).content
    print(f"Reviewer Feedback:\n{response_content}")
    return {"review_feedback": response_content, "approved": "APPROVED" in response_content, "revision_attempts": state['revision_attempts'] + 1}

def should_generalize_and_retry(state: State) -> str:
    if state['search_attempts'] >= 2: return "proceed_to_download"
    if len(state.get('paper_dois_to_download', [])) < 3: return "retry_search"
    return "proceed_to_download"

def should_end_or_revise(state: State) -> str:
    if state.get('approved') or state['revision_attempts'] >= 2: return END
    return "revise_report"

# --- GRAPH DEFINITION ---

workflow = StateGraph(State)

workflow.add_node("extract_initial_info", extract_initial_info)
workflow.add_node("extract_key_terms", extract_key_terms)
workflow.add_node("search_for_papers", search_for_papers)
workflow.add_node("filter_relevant_papers", filter_relevant_papers)
workflow.add_node("generalize_search_terms", generalize_search_terms)
workflow.add_node("download_papers", download_papers)
workflow.add_node("summarize_papers", summarize_papers_node)
workflow.add_node("compile_report", compile_report)
workflow.add_node("review_report", review_report_node)

workflow.set_entry_point("extract_initial_info")
workflow.add_edge("extract_initial_info", "extract_key_terms")
workflow.add_edge("extract_key_terms", "search_for_papers")
workflow.add_edge("search_for_papers", "filter_relevant_papers")
workflow.add_conditional_edges("filter_relevant_papers", should_generalize_and_retry, {"proceed_to_download": "download_papers", "retry_search": "generalize_search_terms"})
workflow.add_edge("generalize_search_terms", "search_for_papers")
workflow.add_edge("download_papers", "summarize_papers")
workflow.add_edge("summarize_papers", "compile_report")
workflow.add_edge("compile_report", "review_report")
workflow.add_conditional_edges("review_report", should_end_or_revise, {"revise_report": "compile_report", END: END})

app = workflow.compile()

# --- EXECUTION ---

if __name__ == "__main__":
    initial_pdf_path = PDF_PAPER_PATH
    
    initial_state = {
        "messages": [], "initial_pdf_path": initial_pdf_path,
        "initial_info": "", "key_terms": "", "search_results": [],
        "paper_dois_to_download": [], "downloaded_papers": [], "extracted_infos": [],
        "report": "", "search_attempts": 0,
        "review_feedback": "", "revision_attempts": 0, "approved": False
    }
    
    result = app.invoke(initial_state)
    print("\n--- Workflow Completed ---")
    print(f"Final report saved. Check 'final_report.md' and the 'downloads' directory.")
