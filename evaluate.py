import json
import random
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
def get_components():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return retriever, llm

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- 2. Build Two Different QA Systems: RAG vs Agent ---
def setup_rag_chain(retriever, llm):
    system_prompt = (
        "You are an expert CFA Level II study assistant. "
        "Use the following retrieved context to answer the user's question.\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. Think step-by-step before providing the final answer.\n"
        "2. If mathematical calculation is needed, explicitly write out the formula and your step-by-step computation.\n\n"
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

    # Removed the state_modifier that caused errors, creating a basic agent directly
    return create_react_agent(llm, tools=[cfa_knowledge_search])

# --- 3. Core Evaluation Pipeline ---
def evaluate_pipeline(sample_size=20):
    retriever, llm = get_components()
    rag_chain = setup_rag_chain(retriever, llm)
    agent_executor = setup_agent(retriever, llm)
    
    print(f"Loading data & Sampling {sample_size} questions...\n")
    with open("data/cfa_questions.json", "r", encoding="utf-8") as f:
        all_questions = json.load(f)
    test_sample = random.sample(all_questions, sample_size)

    q_gen_prompt = ChatPromptTemplate.from_template(
        "Based on this CFA explanation, generate a realistic, tricky student question "
        "that requires this exact knowledge to answer. Just output the question.\n\nContext: {context}"
    )
    q_gen_chain = q_gen_prompt | llm | StrOutputParser()

    judge_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a strict CFA grader. Compare the Standard RAG answer and the Agent answer against the Expected Knowledge.\n"
                   "Evaluate both and strictly output in this format:\n"
                   "RAG VERDICT: [CORRECT/INCORRECT] - [1 short reason]\n"
                   "AGENT VERDICT: [CORRECT/INCORRECT] - [1 short reason]"),
        ("human", "Question: {question}\nExpected Knowledge: {expected}\n\n"
                  "--- Standard RAG Answer ---\n{rag_ans}\n\n"
                  "--- Agent Answer ---\n{agent_ans}")
    ])
    judge_chain = judge_prompt | llm | StrOutputParser()

    metrics = {"hits": 0, "mrr_sum": 0.0, "rag_correct": 0, "agent_correct": 0}

    print("=" * 60)
    for i, q in enumerate(test_sample, 1):
        print(f"--- Testing Sample {i}/{sample_size} ---")
        original_context = q['explanation']
        
        synthetic_question = q_gen_chain.invoke({"context": original_context})
        print(f"[Generated Query]: {synthetic_question}")

        docs = retriever.invoke(synthetic_question)
        hit = 0
        mrr = 0.0
        for rank, doc in enumerate(docs, 1):
            if original_context[:50] in doc.page_content or doc.page_content[:50] in original_context:
                hit = 1
                mrr = 1.0 / rank
                break
        
        metrics["hits"] += hit
        metrics["mrr_sum"] += mrr
        print(f"[Retrieval] Hit: {hit} | MRR: {mrr:.2f}")

        rag_ans = rag_chain.invoke(synthetic_question)
        
        # Stable approach: Dynamically inject System Prompt for each query
        agent_system_msg = (
            "You are a CFA AI Agent. Use your search tool to find information to answer the user. "
            "If you need more info, you can search multiple times."
        )
        agent_result = agent_executor.invoke({
            "messages": [
                ("system", agent_system_msg),
                ("user", synthetic_question)
            ]
        })
        agent_ans = agent_result["messages"][-1].content

        evaluation = judge_chain.invoke({
            "question": synthetic_question,
            "expected": original_context,
            "rag_ans": rag_ans,
            "agent_ans": agent_ans
        })
        
        print("\n[Judge Evaluation]:")
        print(evaluation)
        
        if "RAG VERDICT: CORRECT" in evaluation.upper(): metrics["rag_correct"] += 1
        if "AGENT VERDICT: CORRECT" in evaluation.upper(): metrics["agent_correct"] += 1
        print("=" * 60)

    print("\n" + "#" * 30 + " FINAL METRICS " + "#" * 30)
    print(f"Total Queries Evaluated : {sample_size}")
    print(f"Retrieval Hit Rate      : {(metrics['hits'] / sample_size) * 100:.1f}%")
    print(f"Retrieval MRR           : {metrics['mrr_sum'] / sample_size:.3f}")
    print(f"RAG Accuracy            : {(metrics['rag_correct'] / sample_size) * 100:.1f}%")
    print(f"Agent Accuracy          : {(metrics['agent_correct'] / sample_size) * 100:.1f}%")

if __name__ == "__main__":
    evaluate_pipeline(sample_size=40)