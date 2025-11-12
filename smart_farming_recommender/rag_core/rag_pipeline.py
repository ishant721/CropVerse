import warnings
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List, Dict
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph
from langchain_community.tools.tavily_search import TavilySearchResults
from .disease_predictor import predict_disease
import base64
import tempfile

# Suppress LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning, module='langchain')

# --- 1. Load environment and initialize components ---
load_dotenv()

# Check for GOOGLE_API_KEY
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it in your .env file.")

# Check for TAVILY_API_KEY
if not os.getenv("TAVILY_API_KEY"):
    print("Warning: TAVILY_API_KEY environment variable not set. Web search functionality will be limited.")

# Initialize LLM
try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, convert_system_message_to_human=True)
    # Test if the model is available
    llm.invoke("Hello")
except Exception as e:
    raise e # If gemini-pro-vision fails, something is seriously wrong.
# Initialize Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load Vector Store
app_dir = os.path.dirname(os.path.abspath(__file__))
persist_directory = os.path.join(app_dir, 'chroma_db')
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
retriever = vectorstore.as_retriever()

# Initialize Tavily Search Tool
tavily_tool = TavilySearchResults(max_results=5)

# --- 2. Define the State for our LangGraph Agent ---
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: The user's question, now a list of message parts (text and/or image).
        generation: The LLM's generated answer.
        documents: The list of documents retrieved from the knowledge base.
        web_search_results: The results from web search.
        iteration: The number of cycles completed.
        decision: The decision made by the grader (e.g., "continue", "web_search", "finish", "re-try", "relevant", "irrelevant").
        classification: The classification of the user's question (e.g., "farming", "political", "other").
    """
    question: List[Dict] # Changed from str to List[Dict]
    generation: str
    documents: List[str]
    web_search_results: List[str]
    iteration: int
    decision: str

# --- 3. Define the Nodes of the Graph ---

def retrieve_documents(state):
    """
    Retrieves documents from the vector store.

    Args:
        state (dict): The current graph state.

    Returns:
        dict: New state with documents.
    """
    print("---RETRIEVING DOCUMENTS---")
    messages = state["question"] # Now it's a list of messages
    
    # Extract text content for retrieval
    text_question = ""
    for msg_part in messages:
        if msg_part["type"] == "text":
            text_question += msg_part["text"] + " "
    
    documents = retriever.invoke(text_question.strip())
    return {"documents": documents, "question": messages} # Pass the full messages list back

def grade_documents(state):
    """
    Grades the relevance of retrieved documents to the user question.

    Args:
        state (dict): The current graph state.

    Returns:
        dict: New state with a decision ("continue" or "web_search").
    """
    print("---CHECKING DOCUMENT RELEVANCE---")
    messages = state["question"] # Now it's a list of messages

    # Extract text content for grading
    text_question = ""
    for msg_part in messages:
        if msg_part["type"] == "text":
            text_question += msg_part["text"] + " "

    documents = state["documents"]

    # LLM call to grade documents
    prompt = ChatPromptTemplate.from_messages([
        ("system", 'You are a grader assessing if a retrieved document is sufficient to answer a user\'s question. Grade it as "yes" if the document contains enough detailed information to provide a comprehensive answer. Otherwise, grade it as "no". Provide the binary score as a JSON with a single key "score".'),
        ("user", "User question: {question}\n\nRetrieved document:\n\n{document_content}\n\nIs the document sufficient to answer the question?"),
    ])
    
    grader_chain = prompt | llm | JsonOutputParser()
    
    # Grade each document
    relevant_docs = []
    for doc in documents:
        result = grader_chain.invoke({"document_content": doc.page_content, "question": text_question.strip()})
        grade = result.get("score", "no")
        if grade.lower() == "yes":
            relevant_docs.append(doc)
    
    if relevant_docs:
        print("---DECISION: DOCUMENTS ARE RELEVANT, CONTINUE---")
        return {"documents": relevant_docs, "question": messages, "decision": "continue"}
    else:
        print("---DECISION: NO RELEVANT DOCUMENTS, WEB SEARCH---")
        return {"question": messages, "decision": "web_search"}

def web_search(state):
    """
    Performs a web search using Tavily.

    Args:
        state (dict): The current graph state.

    Returns:
        dict: New state with web search results.
    """
    print("---WEB SEARCH---")
    messages = state["question"] # Now it's a list of messages

    # Extract text content for web search
    text_question = ""
    for msg_part in messages:
        if msg_part["type"] == "text":
            text_question += msg_part["text"] + " "

    web_search_results = tavily_tool.invoke({"query": text_question.strip()})
    return {"web_search_results": web_search_results, "question": messages}

def generate_answer(state):
    """
    Generates an answer using the retrieved documents and/or web search results.

    Args:
        state (dict): The current graph state.

    Returns:
        dict: New state with the generated answer.
    """
    print("---GENERATING ANSWER---")
    messages = state["question"] # Now it's a list of messages
    documents = state["documents"]
    web_search_results = state.get("web_search_results", [])
    
    context_text = ""
    if documents:
        context_text += "Knowledge Base Documents:\n" + "\n\n".join([doc.page_content for doc in documents])
    if web_search_results:
        context_text += "\n\nWeb Search Results:\n" + "\n\n".join([str(s) for s in web_search_results])

    if not context_text:
        generation = "I can only help you with farmer related queries."
    else:
        # Create a system message with the context
        system_message = SystemMessage(
            content=(
                "You are an agricultural expert. Use the following context to answer the question. "
                "Format your answer in a highly structured and presentable way using Markdown. "
                "Use clear, hierarchical headings (e.g., ## for main sections, ### for sub-sections), "
                "bullet points for lists, and bold text for key terms or important details. "
                "Ensure the output is easy to read and well-organized. "
                "If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n"
                f"Context:\n{context_text}"
            )
        )
        
        # Create a human message with the user's input (text and images)
        human_message = HumanMessage(content=messages)
        
        # Invoke the LLM with the full list of messages
        full_messages = [system_message, human_message]
        generation = llm.invoke(full_messages).content
    
    return {"documents": documents, "web_search_results": web_search_results, "question": messages, "generation": generation}

def grade_generation(state):
    """
    Grades the generated answer based on the retrieved documents. This is our 'Verifier'.

    Args:
        state (dict): The current graph state.

    Returns:
        dict: The original state, as this is a read-only check.
    """
    print("---CHECKING IF ANSWER IS GROUNDED IN DOCUMENTS---")
    messages = state["question"] # Now it's a list of messages

    # Extract text content for grading
    text_question = ""
    for msg_part in messages:
        if msg_part["type"] == "text":
            text_question += msg_part["text"] + " "

    documents = state["documents"]
    generation = state["generation"]
    web_search_results = state.get("web_search_results", [])

    # Combine all context for grading
    all_context = ""
    if documents:
        all_context += "\n\n".join([doc.page_content for doc in documents])
    if web_search_results:
        all_context += "\n\n".join([str(s) for s in web_search_results])

    if not all_context: # If no context was available, it cannot be grounded
        print("---DECISION: NO CONTEXT AVAILABLE, GENERATION NOT GROUNDED, FINISH---")
        return {"decision": "finish", "generation": "I can only help you with farmer related queries."}

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", 'You are a grader assessing whether an answer is grounded in / supported by a set of facts. Give a binary "yes" or "no" score to indicate whether the answer is grounded in the provided facts. Provide the binary score as a JSON with a single key "score".'),
            ("user", "Retrieved facts:\n\n{facts}\n\nGenerated answer:\n{generation}\n\nAnswer the question: Is the generated answer grounded in the retrieved facts?"),
        ]
    )
    
    grader_chain = prompt | llm | JsonOutputParser()
    
    result = grader_chain.invoke({"facts": all_context, "generation": generation})
    grade = result.get("score", "no")
    
    if grade.lower() == "yes":
        print("---DECISION: GENERATION IS GROUNDED, FINISH---")
        return {"decision": "finish"}
    else:
        print("---DECISION: GENERATION NOT GROUNDED, FINISH---")
        return {"decision": "finish", "generation": "I can only help you with farmer related queries."}

def transform_query(state):
    """
    Transforms the query to produce a better response.

    Args:
        state (dict): The current graph state.

    Returns:
        dict: New state with a transformed query.
    """
    print("---TRANSFORMING QUERY---")
    messages = state["question"] # Now it's a list of messages

    # Extract text content for query transformation
    text_question = ""
    for msg_part in messages:
        if msg_part["type"] == "text":
            text_question += msg_part["text"] + " "

    documents = state["documents"]
    web_search_results = state.get("web_search_results", [])
    iteration = state.get("iteration", 0) + 1

    if iteration > 3:
        print("---MAX ITERATIONS REACHED, FINISHING---")
        return {"question": messages} # Return the original messages

    context = ""
    if documents:
        context += "Knowledge Base Documents:\n" + "\n\n".join([doc.page_content for doc in documents])
    if web_search_results:
        context += "\n\n".join([str(s) for s in web_search_results])

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a query transformation expert. Your task is to rewrite the user's question to be more specific and easier to answer, based on the previously retrieved information (documents and/or web search results). Do not generate an answer, only a better question."),
        ("user", "Original question: {question}\n\nPreviously retrieved information (may be irrelevant):\n{context}\n\nRewrite the question to improve the chances of retrieving relevant information."),
    ])
    
    transform_chain = prompt | llm | StrOutputParser()
    better_question_text = transform_chain.invoke({"question": text_question.strip(), "context": context})
    
    # Reconstruct the messages with the transformed text question
    transformed_messages = []
    for msg_part in messages:
        if msg_part["type"] == "text":
            transformed_messages.append({"type": "text", "text": better_question_text})
        else:
            transformed_messages.append(msg_part) # Keep image parts as they are
            
    return {"question": transformed_messages, "iteration": iteration}

# --- 4. Build the Graph ---

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve_documents", retrieve_documents)
# grade_documents node is removed
workflow.add_node("web_search", web_search)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("grade_generation", grade_generation)
workflow.add_node("transform_query", transform_query)

# Set the entrypoint
workflow.set_entry_point("retrieve_documents")

# Add edges
workflow.add_edge("retrieve_documents", "web_search")
workflow.add_edge("web_search", "generate_answer")
workflow.add_edge("generate_answer", "grade_generation")

# Add conditional edges for generation grading
workflow.add_conditional_edges(
    "grade_generation",
    lambda x: x["decision"],
    {
        "finish": END,
        "re-try": "transform_query",
    },
)
workflow.add_edge("transform_query", "retrieve_documents")

# Compile the graph
app = workflow.compile()

# --- 5. Define the main pipeline function ---

def get_rag_pipeline():
    """
    Returns the compiled LangGraph agent.
    """
    return app

def perform_tavily_search(query: str) -> List[Dict]:
    """
    Performs a web search using Tavily and returns the results.
    """
    if not query:
        return []
    print(f"---PERFORMING TAVILY SEARCH FOR: {query}---")
    try:
        results = tavily_tool.invoke({"query": query})
        return results
    except Exception as e:
        print(f"Error during Tavily search for query '{query}': {e}")
        return []

from IPython.display import Image, display

# Example usage (for testing purposes)
if __name__ == "__main__":
    # Generate and save the architecture diagram
    try:
        img_data = app.get_graph().draw_mermaid_png()
        with open("architecture.png", "wb") as f:
            f.write(img_data)
        print("Architecture diagram saved as architecture.png")
    except Exception as e:
        print(f"Error generating architecture diagram: {e}")
        print("Please ensure you have the 'pygraphviz' or 'pydot' package installed for graph visualization.")

    # Ensure you have a chroma_db directory with ingested data
    # Run: python smart_farming_recommender/rag_core/data_ingest.py first
    
    # pipeline = get_rag_pipeline()
    
    # # The input to the graph is a dictionary with the key "question"
    # # inputs = {"question": "What are the common diseases of rice and how to prevent them?"}
    inputs = {"question": [{"type": "text", "text": "What are the common diseases of rice and how to prevent them?"}]}
    
    # for output in pipeline.stream(inputs):
    #     for key, value in output.items():
    #         print(f"Output from node '{key}':")
    #         print("---")
    #         print(value)
    #     print("\n---\n")
