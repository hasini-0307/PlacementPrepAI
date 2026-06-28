from src.chain import create_chain
from src.process_documents import process_documents
from src.memory import get_chat_history
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from src.ats_analyzer import analyze_resume
from src.skill_gap_analyzer import analyze_skill_gap
from src.roadmap_generator import generate_roadmap
from collections import OrderedDict
from src.interview_generator import generate_interview
from src.interview_agent import (
    start_interview,
    evaluate_and_continue
)
from src.reranker import Reranker
from src.context_guard import has_sufficient_context
from src.langfuse_client import langfuse
import re
class RAGPipeline:

    def __init__(self):

        self.vectorstore = None
        self.chunks = None
        self.retriever = None
        self.prompt = None
        self.llm = None
        self.parser = None
        self.chat_history = get_chat_history()
        self.reranker = Reranker()
        self.memory_window = 8
        recent_messages = self.chat_history.messages[-self.memory_window:]
        self.retrieval_cache = OrderedDict()
        self.max_cache_size = 100

    def normalize_question(self, question):

        question = question.lower()

        question = re.sub(r"[^\w\s]", "", question)

        question = " ".join(question.split())

        return question
                


    def load_documents(self, pdf_paths):

        self.vectorstore = None
        self.retriever = None

        self.vectorstore, self.chunks = process_documents(pdf_paths)
        self.retriever, self.prompt, self.llm, self.parser = create_chain(
            self.vectorstore,
            self.chunks
        )

    def ats_analysis(self):
     
    

     docs = self.retriever.invoke(
        "Provide complete information about the candidate."
    )

     context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

     return analyze_resume(context)
    
    def skill_gap_analysis(self):

     docs = self.retriever.invoke(
        "Provide complete information about the candidate and job requirements."
    )

     context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

     return analyze_skill_gap(context)
    

    def roadmap(self, goal):

     docs = self.retriever.invoke(
        "Provide complete information about the candidate."
    )

     context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

     return generate_roadmap(
        context,
        goal
    )


    def interview_questions(self, role):

     docs = self.retriever.invoke(
        "Provide complete information about the candidate."
    )

     context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

     return generate_interview(
        context,
        role
    )
    

    def start_interview(self, role):

     docs = self.retriever.invoke(
        "Provide complete information about the candidate."
    )

     context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

     return start_interview(
        context,
        role
    )


    def continue_interview(
        self,
        role,
        question,
        answer,
        history
):

     history_text = ""

     for item in history:

        history_text += (
            f"Question: {item['question']}\n"
            f"Answer: {item['answer']}\n\n"
        )

     result = evaluate_and_continue(
        role,
        question,
        answer,
        history_text
    )

     return result


    def ask(self, question):

        with langfuse.start_as_current_observation(
            name=f"Q: {question[:50]}",
            input={
                "question": question
            },
            metadata={
                "project": "PlacementPrepAI",
                "retrieval": "Hybrid+MultiQuery",
                "reranker": "CrossEncoder"
            }
        ) as query_obs:   

            if self.vectorstore is None:

                return iter(
                    "Please upload and process PDF documents first."
                , [],{})

            with langfuse.start_as_current_observation(
                name="retrieval"
            ) as retrieval_obs:

                cache_key = self.normalize_question(question)
               

                if cache_key in self.retrieval_cache:

                    docs = self.retrieval_cache[cache_key]

                    # Move this entry to the end (most recently used)
                    self.retrieval_cache.move_to_end(cache_key)

                    cache_hit = True

                else:

                    docs = self.retriever.invoke(question)

                    self.retrieval_cache[cache_key] = docs

                    # Remove the oldest item if cache exceeds limit
                    if len(self.retrieval_cache) > self.max_cache_size:

                        oldest_key, _ = self.retrieval_cache.popitem(last=False)

                        print(f"🗑️ Evicted cache entry: {oldest_key}")

                    cache_hit = False

                retrieval_obs.create_event(
                name="cache_status",
                input={
                    "status": "HIT" if cache_hit else "MISS"
                }
            )
                


            with langfuse.start_as_current_observation(
                name="reranking"
            ) as rerank_obs:

                docs, scores = self.reranker.rerank(
                    question,
                    docs,
                    top_k=5
                )

                avg_score = round(
                    sum(scores) / len(scores),
                    2
                )

                rerank_obs.update(
                    output={
                        "avg_score": avg_score,
                        "max_score": round(max(scores), 2),
                        "scores": [round(s, 2) for s in scores]
                    }
                )

                   
            
            print("\nScores:")
            print(scores)

            avg_score = sum(scores)/len(scores)

            print("Average score:", avg_score)
            print("Max score:", max(scores))
            if len(scores) == 0:

                return (
            "I couldn't find enough information in the uploaded documents.",
            [],{}
        )

            # Hallucination guard disabled temporarily
            if not has_sufficient_context(docs):
                return (
                "I couldn't find enough information in the uploaded documents.",
                [],{}
            )
            print("\nRetrieved docs:", len(docs))

        

            context = "\n\n".join(
                doc.page_content
                for doc in docs
            )

            
            recent_messages = self.chat_history.messages[-self.memory_window:]

            history = "\n".join(
                f"{msg.type}: {msg.content}"
                for msg in recent_messages
    )

            messages = self.prompt.invoke(
                {
                    "history": history,
                    "context": context,
                    "question": question
                }
            )

            try:

                with langfuse.start_as_current_observation(
                    name="generation"
                ):

                    response = self.llm.stream(messages)

            except Exception:

                return iter(
                    "⚠️ LLM unavailable or rate limit exceeded. Please try again later."
                , [],{})

            pages = sorted(
                set(
                    doc.metadata["page"] + 1
                    for doc in docs
                )
            )
            avg_score = round(
            sum(scores) / len(scores),
            2
        )
            metadata = {
            "num_chunks": len(docs),
            "avg_score": avg_score,
            "max_score": max(scores),
            "pages": pages,
            "retrieval": "Hybrid + MultiQuery",
            "reranker": "CrossEncoder"
    }
            
            
            query_obs.update(
            output={
                "pages": pages,
                "num_chunks": len(docs),
                "avg_score": avg_score
            }
        )


            langfuse.flush()
            return response, pages, metadata