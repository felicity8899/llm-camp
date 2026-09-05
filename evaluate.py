import json
import random
import argparse
import os 
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()

# --- 1. Basic Setup & LLM Initialization ---
def get_components(
    embedding_model="text-embedding-3-small",
    persist_directory="./chroma_db",
    k=3,
    search_type="similarity",
    model_name="gpt-4o-mini",
    temperature=0.0
):
    """
    Dynamically build Embedding, Vector Store retriever, and LLM instances via parameters.
    """
    embeddings = OpenAIEmbeddings(model=embedding_model)
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    
    search_kwargs = {"k": k}
    if search_type == "mmr":
        search_kwargs["fetch_k"] = max(20, k * 3)
        
    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )
    
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    return retriever, llm

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- 2. Build Two Different QA Systems: RAG vs Agent ---
def setup_rag_chain(retriever, llm):
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
        ("human", "{input}"),
    ])
    return {"context": retriever | format_docs, "input": RunnablePassthrough()} | prompt | llm | StrOutputParser()

def setup_agent(retriever, llm):
    @tool
    def cfa_knowledge_search(query: str) -> str:
        """Search for CFA Level II curriculum concepts, formulas, and explanations."""
        docs = retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in docs)
        
    return create_react_agent(llm, tools=[cfa_knowledge_search])

# --- 3. Core Evaluation Pipeline ---
def evaluate_pipeline(
    sample_size=20,
    embedding_model="text-embedding-3-small",
    persist_directory="./chroma_db",
    k=3,
    search_type="similarity",
    model_name="gpt-4o-mini",
    temperature=0.0
):
    retriever, llm = get_components(
        embedding_model=embedding_model,
        persist_directory=persist_directory,
        k=k,
        search_type=search_type,
        model_name=model_name,
        temperature=temperature
    )
    
    rag_chain = setup_rag_chain(retriever, llm)
    agent_executor = setup_agent(retriever, llm)
    
    print("=" * 60)
    print(f"📊 STARTING EVALUATION WITH CONFIGURATION:")
    print(f" - LLM Model Name      : {model_name}")
    print(f" - Embedding Model     : {embedding_model}")
    print(f" - Search Strategy     : {search_type} (k={k})")
    print(f" - Temperature         : {temperature}")
    print(f" - Evaluation Sample   : {sample_size}")
    print("=" * 60)
    
    # Try to load the test dataset
    try:
        with open("data/cfa_questions.json", "r", encoding="utf-8") as f:
            all_questions = json.load(f)
    except FileNotFoundError:
        # Compatibility fallback: if not found in the current path, try to look in scratch or artifacts
        print("[Warning] data/cfa_questions.json not found, falling back to dummy questions for testing.")
        all_questions = [
            {
                "question_id": 1,
                "question_text": "Determine the type of regression model you should use.",
                "explanation": "You should use a multiple linear regression model since the dependent variable is continuous and there is more than one explanatory variable."
            }
        ]
        
    test_sample = random.sample(all_questions, min(len(all_questions), sample_size))
    
    q_gen_prompt = ChatPromptTemplate.from_template(
        "Based on this CFA explanation, generate a realistic, tricky student question "
        "that requires this exact knowledge to answer.\n"
        "Just output the question.\n\n"
        "Context: {context}"
    )
    q_gen_chain = q_gen_prompt | llm | StrOutputParser()
    
    judge_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a strict CFA grader.\n"
                  "Compare the Standard RAG answer and the Agent answer against the Expected Knowledge.\n\n"
                  "Evaluate both and strictly output in this format:\\n"
                  "RAG VERDICT: [CORRECT/INCORRECT] - [1 short reason]\\n"
                  "AGENT VERDICT: [CORRECT/INCORRECT] - [1 short reason]"),
        ("human", "Question: {question}\nExpected Knowledge: {expected}\n\n"
                  "--- Standard RAG Answer ---\n{rag_ans}\n\n"
                  "--- Agent Answer ---\n{agent_ans}")
    ])
    judge_chain = judge_prompt | llm | StrOutputParser()
    
    metrics = {"hits": 0, "mrr_sum": 0.0, "rag_correct": 0, "agent_correct": 0}
    
    for i, q in enumerate(test_sample, 1):
        print(f"\n--- Testing Sample {i}/{sample_size} ---")
        original_context = q['explanation']
        synthetic_question = q_gen_chain.invoke({"context": original_context})
        print(f"[Generated Query]: {synthetic_question}")
        
        # Simulate retrieval and computation
        try:
            docs = retriever.invoke(synthetic_question)
            hit = 0
            mrr = 0.0
            for rank, doc in enumerate(docs, 1):
                if original_context[:50] in doc.page_content or doc.page_content[:50] in original_context:
                    hit = 1
                    mrr = 1.0 / rank
                    break
        except Exception as e:
            print(f"[Error in Retrieval]: {e}")
            hit = 0
            mrr = 0.0
                
        metrics["hits"] += hit
        metrics["mrr_sum"] += mrr
        print(f"[Retrieval] Hit: {hit} | MRR: {mrr:.2f}")
        
        # Standard RAG answer
        rag_ans = rag_chain.invoke(synthetic_question)
        
        # Agent answer
        agent_system_msg = (
            "You are a CFA AI Agent. Use your search tool to find information to answer the user. "
            "If you need more info, you can search multiple times."
        )
        try:
            agent_result = agent_executor.invoke({
                "messages": [
                    ("system", agent_system_msg),
                    ("user", synthetic_question)
                ]
            })
            agent_ans = agent_result["messages"][-1].content
        except Exception as e:
            print(f"[Error in Agent execution]: {e}")
            agent_ans = "Error generating agent answer."
        
        # Judge evaluation / Scoring
        evaluation = judge_chain.invoke({
            "question": synthetic_question,
            "expected": original_context,
            "rag_ans": rag_ans,
            "agent_ans": agent_ans
        })
        print(f"[Judge Evaluation]:\n{evaluation}")
        
        if "RAG VERDICT: CORRECT" in evaluation.upper():
            metrics["rag_correct"] += 1
        if "AGENT VERDICT: CORRECT" in evaluation.upper():
            metrics["agent_correct"] += 1
        
    # --- 4. Aggregate Results and Automatically Output JSON Experiment Report ---
    hit_rate = (metrics['hits'] / sample_size) * 100
    mrr = metrics['mrr_sum'] / sample_size
    rag_accuracy = (metrics['rag_correct'] / sample_size) * 100
    agent_accuracy = (metrics['agent_correct'] / sample_size) * 100

    print("\n" + "#" * 30 + " FINAL METRICS " + "#" * 30)
    print(f"Total Queries Evaluated : {sample_size}")
    print(f"Retrieval Hit Rate      : {hit_rate:.1f}%")
    print(f"Retrieval MRR           : {mrr:.3f}")
    print(f"RAG Accuracy            : {rag_accuracy:.1f}%")
    print(f"Agent Accuracy          : {agent_accuracy:.1f}%")
   
    #Save the parameter as report
    report_dir = "reports"
    os.makedirs(report_dir, exist_ok=True)
    
    report_file = f"eval_report_{model_name}_{search_type}_k{k}.json".replace("/", "_")
    full_report_path = os.path.join(report_dir, report_file)
    
    report_data = {
        "parameters": {
            "model_name": model_name,
            "embedding_model": embedding_model,
            "search_type": search_type,
            "k": k,
            "temperature": temperature,
            "sample_size": sample_size
        },
        "metrics": {
            "hit_rate": hit_rate,
            "mrr": mrr,
            "rag_accuracy": rag_accuracy,
            "agent_accuracy": agent_accuracy
        }
    }
    with open(full_report_path, "w", encoding="utf-8") as rf:
        json.dump(full_report_path, rf, indent=4, ensure_ascii=False)
    print(f"\n[Success] Experiment metrics saved to: {report_file}")

# --- 5. Command Line Interface Parser ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tunable CFA Evaluation Pipeline for RAG vs Agent")
    
    parser.add_argument("--sample-size", type=int, default=20, help="Number of questions to test in this run")
    parser.add_argument("--k", type=int, default=3, help="Number of documents for retriever to fetch")
    parser.add_argument("--search-type", type=str, choices=["similarity", "mmr"], default="similarity", help="Retrieval algorithm strategy")
    parser.add_argument("--embedding-model", type=str, default="text-embedding-3-small", help="LangChain OpenAI embeddings model name")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM engine used for QA generation and grading")
    parser.add_argument("--temperature", type=float, default=0.0, help="Creative temperature of LLM (0.0 is ideal for strict QA)")
    parser.add_argument("--persist-directory", type=str, default="./chroma_db", help="Path to your chroma database folder")
    
    args = parser.parse_args()
    
    evaluate_pipeline(
        sample_size=args.sample_size,
        embedding_model=args.embedding_model,
        persist_directory=args.persist_directory,
        k=args.k,
        search_type=args.search_type,
        model_name=args.model,
        temperature=args.temperature
    )