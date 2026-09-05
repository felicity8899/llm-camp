import time
import sqlite3
import yaml
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks import get_openai_callback
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()

# --- 1. Load Configuration from YAML ---
def load_config(config_path="config.yaml"):
    """
    Automatically load the YAML configuration file and provide robust default fallback values.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # Fallback configurations if config.yaml is missing
        return {
            "retriever": {
                "search_type": "similarity",
                "k": 3,
                "embedding_model": "text-embedding-3-small",
                "persist_directory": "./chroma_db"
            },
            "llm": {
                "model_name": "gpt-4o-mini",
                "temperature": 0.0
            },
            "system": {
                "engine_type": "Standard RAG Chain",
                "sample_size": 20
            }
        }

# --- 2. Database Initialization & Operational Functions ---
def init_all_dbs():
    """Initialize SQLite tables for metrics logging and user/judge feedback."""
    conn = sqlite3.connect('metrics.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS llm_metrics (
                    conversation_id TEXT PRIMARY KEY, 
                    timestamp TEXT, 
                    response_time REAL, 
                    prompt_tokens INTEGER, 
                    completion_tokens INTEGER, 
                    total_tokens INTEGER, 
                    cost REAL, 
                    eval_passed INTEGER)''')
    c.execute('''CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    conversation_id TEXT, 
                    source TEXT NOT NULL, 
                    relevance TEXT, 
                    explanation TEXT, 
                    score INTEGER, 
                    timestamp TEXT NOT NULL)''')
    conn.commit()
    conn.close()

def save_feedback(conversation_id, source, relevance=None, explanation=None, score=None):
    """Save user or system feedback into the feedback database."""
    conn = sqlite3.connect('metrics.db')
    c = conn.cursor()
    c.execute(
        """INSERT INTO feedback (conversation_id, source, relevance, explanation, score, timestamp) 
           VALUES (?, ?, ?, ?, ?, ?)""",
        (conversation_id, source, relevance, explanation, score, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def format_docs(docs):
    """Concatenate retrieved documents into a single text block."""
    return "\n\n".join(doc.page_content for doc in docs)

# --- 3. Parameterized CFA Assistant Core ---
class CFAAssistant:
    def __init__(self):
        # 1. Initialize databases
        init_all_dbs()
        
        # 2. Load settings from YAML
        self.config = load_config()
        self.retriever_config = self.config["retriever"]
        self.llm_config = self.config["llm"]
        self.sys_config = self.config["system"]
        
        # 3. Dynamic setup of embedding and vectorstore
        embeddings = OpenAIEmbeddings(model=self.retriever_config["embedding_model"])
        vectorstore = Chroma(
            persist_directory=self.retriever_config["persist_directory"], 
            embedding_function=embeddings
        )
        
        # Dynamic search parameters configuration
        search_kwargs = {"k": self.retriever_config["k"]}
        if self.retriever_config["search_type"] == "mmr":
            # Set fetch_k to expand initial candidate pool for diversity filtering
            search_kwargs["fetch_k"] = max(20, self.retriever_config["k"] * 3)
            
        self.retriever = vectorstore.as_retriever(
            search_type=self.retriever_config["search_type"],
            search_kwargs=search_kwargs
        )
        
        # 4. Initialize LLM with YAML settings
        self.llm = ChatOpenAI(
            model=self.llm_config["model_name"], 
            temperature=self.llm_config["temperature"]
        )
        
        # 5. Define Standard RAG Chain
        system_prompt = (
            "You are an expert CFA Level II study assistant.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. DIRECT SCENARIO LINKING: You must directly apply the retrieved context to the specific question asked. "
            "Do NOT just summarize or introduce the theoretical method in general. You must explicitly link the variables, "
            "equations, and concepts from the retrieved context directly to the scenario, numbers, or choices in the user's question.\n\n"
            "2. NO DETACHED OVERVIEWS: Do not write generic introductory paragraphs about the financial topic. "
            "Immediately address the core problem. If a calculation is required, define the formula and compute the step-by-step answer "
            "using the exact numbers given in the question.\n\n"
            "3. COHERENCE & TRACEABILITY: Ensure your logical steps clearly demonstrate how the retrieved principles "
            "lead to your final conclusion. Do not skip logical links.\n\n"
            "Context:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        self.rag_chain = {"context": self.retriever | format_docs, "input": RunnablePassthrough()} | prompt | self.llm | StrOutputParser()
        
        # 6. Define LangGraph Tool-Calling Agent
        @tool
        def cfa_knowledge_search(query: str) -> str:
            """Search for CFA Level II curriculum concepts, formulas, and explanations."""
            docs = self.retriever.invoke(query)
            return "\n\n".join(doc.page_content for doc in docs)
            
        self.agent_executor = create_react_agent(self.llm, tools=[cfa_knowledge_search])
        
    def run_chat(self, prompt_text):
        call_time = datetime.now()
        conversation_id = call_time.strftime("%Y%m%d%H%M%S%f")
        
        start_time = time.time()
        
        # RAG Execution with operational metrics tracking
        with get_openai_callback() as cb:
            if self.sys_config["engine_type"] == "LangGraph Tool Agent":
                # Execute via multi-turn LangGraph Agent
                agent_system_msg = (
                    "You are a CFA AI Agent. Use your search tool to find information to answer the user. "
                    "If you need more info, you can search multiple times."
                )
                agent_result = self.agent_executor.invoke({
                    "messages": [
                        ("system", agent_system_msg),
                        ("user", prompt_text)
                    ]
                })
                response = agent_result["messages"][-1].content
                
                # Fetch context for the online judge evaluator (based on final tool call results or pure search)
                docs = self.retriever.invoke(prompt_text)
                context_str = format_docs(docs)
            else:
                # Execute via standard RAG chain
                docs = self.retriever.invoke(prompt_text)
                context_str = format_docs(docs)
                response = self.rag_chain.invoke(prompt_text)
                
            response_time = time.time() - start_time
            
        # AI Judge online evaluation
        judge_llm = ChatOpenAI(model=self.llm_config["model_name"], temperature=0)
        eval_chain = ChatPromptTemplate.from_messages([
            ("system", "You are a strict QA evaluator. Output ONLY 'PASS' or 'FAIL' followed by a short reason."),
            ("human", "Context:\n{context}\n\nAnswer:\n{answer}")
        ]) | judge_llm | StrOutputParser()
        
        eval_result = eval_chain.invoke({"context": context_str, "answer": response})
        is_pass = "PASS" in eval_result.upper()
        
        # Save metrics to sqlite db (llm_metrics table)
        conn = sqlite3.connect('metrics.db')
        c = conn.cursor()
        c.execute(
            "INSERT OR REPLACE INTO llm_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?)", 
            (conversation_id, call_time.isoformat(), response_time, cb.prompt_tokens, cb.completion_tokens, cb.total_tokens, cb.total_cost, 1 if is_pass else 0)
        )
        conn.commit()
        conn.close()
        
        # Save automated evaluation to feedback table
        judge_score = 1 if is_pass else -1
        save_feedback(
            conversation_id=conversation_id,
            source="judge",
            relevance="PASS" if is_pass else "FAIL",
            explanation=eval_result,
            score=judge_score
        )
        
        metrics_text = f"{'✅ AI-Pass' if is_pass else '⚠️ AI-Review'} | ⏱️ {response_time:.2f}s | 🪙 Tokens: {cb.total_tokens} | 💰 ${cb.total_cost:.6f}"
        return {
            "conversation_id": conversation_id,
            "response": response,
            "metrics_text": metrics_text
        }

# --- 4. Interactive Terminal CLI Loop ---
if __name__ == "__main__":
    print("=" * 60)
    print("🤖 Initializing CFA Level II CLI Assistant...")
    assistant = CFAAssistant()
    
    print("\n📊 CURRENT ACTIVE PARAMETERS (from config.yaml):")
    print(f" - LLM Engine       : {assistant.llm_config['model_name']}")
    print(f" - Active Pipeline  : {assistant.sys_config['engine_type']}")
    print(f" - Search Strategy  : {assistant.retriever_config['search_type']} (k={assistant.retriever_config['k']})")
    print("=" * 60)
    print("✅ Assistant is ready! Type your question below (type 'exit' or 'quit' to stop).\n")
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            if not user_input.strip():
                continue
                
            print("\n[System]: Thinking & Evaluating...")
            result = assistant.run_chat(user_input)
            
            print(f"\nAssistant:\n{result['response']}")
            print(f"\n[Metrics & AI Judge: {result['metrics_text']}]")
            
            # Request and save user manual feedback (Thumbs up / down)
            
            feedback_choice = input("\nRate this answer [Enter to skip | type '1' for 👍 | type '-1' for 👎]: ").strip()
            if feedback_choice in ["1", "-1"]:
                save_feedback(
                    conversation_id=result["conversation_id"],
                    source="user",
                    score=int(feedback_choice)
                )
                print(f"✅ Human feedback recorded ({'👍 Thumbs Up' if feedback_choice == '1' else '👎 Thumbs Down'})!")
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
