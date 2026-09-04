import time
import sqlite3
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks import get_openai_callback

load_dotenv()

def init_all_dbs():
    conn = sqlite3.connect('metrics.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS llm_metrics
                 (conversation_id TEXT PRIMARY KEY,
                  timestamp TEXT, response_time REAL, 
                  prompt_tokens INTEGER, completion_tokens INTEGER, 
                  total_tokens INTEGER, cost REAL, eval_passed INTEGER)''')
    c.execute('''CREATE TABLE IF NOT EXISTS feedback 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  conversation_id TEXT,
                  source TEXT NOT NULL,
                  relevance TEXT,
                  explanation TEXT,
                  score INTEGER,
                  timestamp TEXT NOT NULL)''')
    conn.commit()
    conn.close()

def save_feedback(conversation_id, source, relevance=None, explanation=None, score=None):
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
    return "\n\n".join(doc.page_content for doc in docs)

class CFAAssistant:
    def __init__(self):
        init_all_dbs()
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        system_prompt = (
            "You are an expert CFA Level II study assistant. "
            "Use the following retrieved context to answer the user's question. "
            "If you don't know the answer based on the context, just say that you don't know. "
            "CRITICAL INSTRUCTIONS:\n"
            "1. Think step-by-step before providing the final answer.\n"
            "2. If mathematical calculation is needed, explicitly write out the formula and your step-by-step computation.\n\n"
            "Context:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
        self.rag_chain = {"context": self.retriever | format_docs, "input": RunnablePassthrough()} | prompt | self.llm | StrOutputParser()

    def run_chat(self, prompt_text):
        call_time = datetime.now()
        conversation_id = call_time.strftime("%Y%m%d%H%M%S%f")

        # 1. RAG execution & operational metrics tracking
        start_time = time.time()
        with get_openai_callback() as cb:
            docs = self.retriever.invoke(prompt_text)
            context_str = format_docs(docs)
            response = self.rag_chain.invoke(prompt_text)
        response_time = time.time() - start_time

        # 2. AI Judge evaluation (Online Evaluation)
        judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        eval_chain = ChatPromptTemplate.from_messages([
            ("system", "You are a strict QA evaluator. Output ONLY 'PASS' or 'FAIL' followed by a short reason."),
            ("human", "Context:\n{context}\n\nAnswer:\n{answer}")
        ]) | judge_llm | StrOutputParser()
        
        eval_result = eval_chain.invoke({"context": context_str, "answer": response})
        is_pass = "PASS" in eval_result.upper()

        # 3. Save to llm_metrics table (using .isoformat())
        conn = sqlite3.connect('metrics.db')
        c = conn.cursor()
        c.execute("INSERT OR REPLACE INTO llm_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                  (conversation_id, call_time.isoformat(), response_time, cb.prompt_tokens, cb.completion_tokens, cb.total_tokens, cb.total_cost, 1 if is_pass else 0))
        conn.commit()
        conn.close()

        # 4. Save to feedback table (source = "judge")
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


# --- Terminal CLI Mode (with User Evaluation) ---
if __name__ == "__main__":
    print("=" * 60)
    print("🤖 Initializing CFA Level II Assistant (CLI Mode with Feedback)...")
    assistant = CFAAssistant()
    print("✅ Assistant is ready! Type your question below (type 'exit' or 'quit' to stop).\n")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            if not user_input.strip():
                continue
            
            print("\nThinking & Evaluating...")
            result = assistant.run_chat(user_input)
            
            print(f"\nAssistant:\n{result['response']}")
            print(f"\n[Metrics & AI Judge: {result['metrics_text']}]")

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