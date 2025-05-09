import os
from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task, LLM
from crewai_tools import PDFSearchTool, WebsiteSearchTool
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st
import tempfile
from apify_client import ApifyClient
from crewai.tools import BaseTool
from crewai_tools import ApifyActorsTool

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("api_key")

# --- LLM and Embeddings ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
llm = LLM(model="groq/llama-3.3-70b-versatile", api_key=groq_api_key, temperature=0.7, max_tokens=500)

embedder = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

# --- Streamlit UI ---
st.title("CV Analysis Chatbot")

uploaded_file = st.file_uploader("Upload your CV (PDF)", type=["pdf"])

if uploaded_file:
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    # Define tools dynamically based on uploaded PDF
    pdf_search_tool = PDFSearchTool(
        pdf=pdf_path,
        config=dict(
            llm=dict(provider="groq", config=dict(model="llama-3.3-70b-versatile")),
            embedder=dict(provider="huggingface", config=dict(model=EMBEDDING_MODEL)),
        ),
    )

    web_search_tool = WebsiteSearchTool(
        config=dict(
            llm=dict(provider="groq", config=dict(model="llama-3.3-70b-versatile")),
            embedder=dict(provider="huggingface", config=dict(model=EMBEDDING_MODEL)),
        )
    )

    # Scrapes Google for “<title> jobs in <city>” keywords
    job_search_tool = ApifyActorsTool(
        name="job_search",
        description="Fetch recent job listings by scraping Google Search results",
        actor_name="apify/google-search-scraper",      # free SERP scraper actor
        apify_token=os.getenv("APIFY_API_TOKEN"),         # your Apify token (free tier available)
        default_input={                                # base inputs; we'll override 'queries' per-run
            "numResults": 20,                          # fetch up to 20 results
            "safe": "active",
        },
    )
    # --- Agent and Task ---
    cv_analysis_agent = Agent(
        role="CV Analysis Agent",
        goal="Understand and analyze the uploaded CV, answering questions and offering suggestions for improvement. The question is {user_input}",
        allow_delegation=True,
        verbose=True,
        backstory="""
            This agent is skilled at understanding resume content. It can answer questions 
            based on the CV, identify strengths and weaknesses, and suggest ways to improve 
            the document. If necessary, it uses online search to provide relevant, updated 
            advice.
        """,
        tools=[pdf_search_tool, web_search_tool],
        llm=llm,
    )

    cv_question_task = Task(
        description="""
            Answer the user's question based on the uploaded CV. 
            Use the CV content to provide a detailed and accurate answer.
            If the question relates to improving the CV or identifying strengths/weaknesses,
            analyze the text and use web search for suggestions if needed using the web search tool.

            Question:
            {user_input}
        """,
        expected_output="""
            An informative, CV-based answer that may include suggestions, feedback, or 
            skill improvement tips depending on the question context.
        """,
        tools=[pdf_search_tool, web_search_tool],
        agent=cv_analysis_agent,
    )
    
    # New Agent: Skill Extraction Agent
    skill_agent = Agent(
        role="Skill Extraction Agent",
        goal="Extract technical and soft skills from the CV.",
        allow_delegation=True,
        verbose=True,
        backstory="""
            This agent is skilled at Skill Extraction from the CV. it can identify and categorize
            skills into technical, soft, tools, and languages. 
        """,
        tools=[pdf_search_tool],
        llm=llm,
    )

    # New Agent: Improvement Recommender Agent
    improvement_agent = Agent(
        role="Improvement Suggestion Agent",
        goal="Provide suggestions to improve the CV for job search or academic roles.",
        allow_delegation=True,
        verbose=True,
        backstory="""
            This agent is skilled at providing suggestions for improving CVs. It can analyze the CV content
            and suggest ways to enhance it based on job market trends or missing elements.""",
        tools=[pdf_search_tool, web_search_tool],
        llm=llm,
    )

    # Add new Tasks:
    extraction_task = Task(
        description="Extract and list the skills and experience keywords from the CV.",
        expected_output="A categorized list of skills (technical, soft, tools, languages).",
        tools=[pdf_search_tool],
        agent=skill_agent,
    )

    suggestion_task = Task(
        description="Suggest improvements based on missing elements or job market trends.",
        expected_output="Customized suggestions for improving the CV.",
        tools=[web_search_tool, pdf_search_tool],
        agent=improvement_agent,
    )
    job_search_agent = Agent(
        role="Job Search Agent",
        goal="Take user keywords + location, run a Google search for recent job postings, "\
            "and return a concise, filtered Markdown list of the top opportunities.",
        backstory=(
            "This agent crafts a Google query like “<keywords> jobs in <city>,” "
            "invokes the job_search tool, filters out non-job links, and formats "
            "the best 5–10 listings with title, company (if available), location, and URL."
        ),
        tools=[job_search_tool],
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )

    search_jobs_task = Task(
        description=(
            "Inputs:\n"
            "- `keywords`: what role or field (e.g. “Data Scientist”)\n"
            "- `city`: location (e.g. “Berlin, Germany”)\n\n"
            "Run the job_search tool with `queries` = f\"{keywords} jobs in {city}\". "
            "Return a Markdown list of the top 5–10 valid job postings, each with:\n"
            "  - **Title**\n"
            "  - *Company/location snippet*\n"
            "  - [Link](URL)\n"
        ),
        expected_output="Markdown bullet-list of job postings.",
        tools=[job_search_tool],
        agent=job_search_agent,
)
    
    crew = Crew(
        #tasks=[cv_question_task,extraction_task,suggestion_task],
        #agents=[cv_analysis_agent, skill_agent, improvement_agent],
        tasks=[cv_question_task],
        agents=[cv_analysis_agent],
        process=Process.sequential,
        memory=False,
        verbose=True
    )

    # --- Chat Memory ---
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    for message in st.session_state.conversation_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    user_input = st.chat_input("Ask something about your CV...")

    if user_input:
        st.session_state.conversation_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Analyzing your CV..."):
            result = crew.kickoff(inputs={"user_input": user_input})

        with st.chat_message("assistant"):
            st.markdown(result)

        st.session_state.conversation_history.append({"role": "assistant", "content": result})

else:
    st.info("Please upload a PDF file to start the analysis.")
