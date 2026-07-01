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
from src.interview_agent import start_interview, evaluate_and_continue
from src.reranker import Reranker
from src.context_guard import has_sufficient_context
from src.langfuse_client import langfuse
from src.logger import logger
import re
from src.utils import normalize_question

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
        recent_messages = self.chat_history.messages[-self.memory_window :]
        self.retrieval_cache = OrderedDict()
        self.max_cache_size = 100

    

    def load_documents(self, pdf_paths):

        self.vectorstore = None
        self.retriever = None

        self.vectorstore, self.chunks = process_documents(pdf_paths)
        self.retriever, self.prompt, self.llm, self.parser = create_chain(
            self.vectorstore, self.chunks
        )

    def ats_analysis(self):

        logger.info("ATS analysis started")

        try:

            docs = self.retriever.invoke(
                "Provide complete information about the candidate."
            )

            logger.info("Retrieved %d documents for ATS analysis", len(docs))

            context = "\n\n".join(doc.page_content for doc in docs)

            result = analyze_resume(context)

            logger.info("ATS analysis completed successfully")

            return result

        except Exception:

            logger.exception("ATS analysis failed")

            raise

    def skill_gap_analysis(self):

        logger.info("Skill Gap Analysis started")

        try:

            docs = self.retriever.invoke(
                "Provide complete information about the candidate and job requirements."
            )

            logger.info("Retrieved %d documents for Skill Gap Analysis", len(docs))

            context = "\n\n".join(doc.page_content for doc in docs)

            result = analyze_skill_gap(context)

            logger.info("Skill Gap Analysis completed successfully")

            return result

        except Exception:

            logger.exception("Skill Gap Analysis failed")

            raise

    def roadmap(self, goal):

        logger.info("Roadmap generation started | Goal: %s", goal)

        try:

            docs = self.retriever.invoke(
                "Provide complete information about the candidate."
            )

            logger.info("Retrieved %d documents for Roadmap Generation", len(docs))

            context = "\n\n".join(doc.page_content for doc in docs)

            result = generate_roadmap(context, goal)

            logger.info("Roadmap generated successfully | Goal: %s", goal)

            return result

        except Exception:

            logger.exception("Roadmap generation failed | Goal: %s", goal)

            raise

    def interview_questions(self, role):

        docs = self.retriever.invoke(
            "Provide complete information about the candidate."
        )

        context = "\n\n".join(doc.page_content for doc in docs)

        return generate_interview(context, role)

    def start_interview(self, role):

        logger.info("Interview started | Role: %s", role)

        docs = self.retriever.invoke(
            "Provide complete information about the candidate."
        )

        context = "\n\n".join(doc.page_content for doc in docs)

        result = start_interview(context, role)

        logger.info("Interview initialized successfully | Role: %s", role)

        return result

    def continue_interview(self, role, question, answer, history):
        logger.info("Evaluating interview response")

        history_text = ""

        for item in history:

            history_text += (
                f"Question: {item['question']}\n" f"Answer: {item['answer']}\n\n"
            )

        result = evaluate_and_continue(role, question, answer, history_text)
        logger.info("Interview response evaluated successfully")
        return result

    def ask(self, question):

        with langfuse.start_as_current_observation(
            name=f"Q: {question[:50]}",
            input={"question": question},
            metadata={
                "project": "PlacementPrepAI",
                "retrieval": "Hybrid+MultiQuery",
                "reranker": "CrossEncoder",
            },
        ) as query_obs:

            if self.vectorstore is None:

                return iter("Please upload and process PDF documents first.", [], {})

            with langfuse.start_as_current_observation(
                name="retrieval"
            ) as retrieval_obs:

                cache_key = normalize_question(question)

                if cache_key in self.retrieval_cache:

                    docs = self.retrieval_cache[cache_key]

                    # Move this entry to the end (most recently used)
                    self.retrieval_cache.move_to_end(cache_key)

                    cache_hit = True
                    logger.info("Retrieval Cache HIT")

                else:

                    docs = self.retriever.invoke(question)

                    self.retrieval_cache[cache_key] = docs

                    # Remove the oldest item if cache exceeds limit
                    if len(self.retrieval_cache) > self.max_cache_size:

                        oldest_key, _ = self.retrieval_cache.popitem(last=False)

                        print(f"🗑️ Evicted cache entry: {oldest_key}")

                    cache_hit = False
                    logger.info("Retrieval Cache MISS")

                retrieval_obs.create_event(
                    name="cache_status",
                    input={"status": "HIT" if cache_hit else "MISS"},
                )

            with langfuse.start_as_current_observation(name="reranking") as rerank_obs:

                docs, scores = self.reranker.rerank(question, docs, top_k=5)

                avg_score = round(sum(scores) / len(scores), 2)

                rerank_obs.update(
                    output={
                        "avg_score": avg_score,
                        "max_score": round(max(scores), 2),
                        "scores": [round(s, 2) for s in scores],
                    }
                )

            avg_score = sum(scores) / len(scores)

            logger.info(
                "Reranking completed | Avg Score: %.2f | Max Score: %.2f",
                avg_score,
                max(scores),
            )

            logger.debug("Scores: %s", scores)
            if len(scores) == 0:

                return (
                    "I couldn't find enough information in the uploaded documents.",
                    [],
                    {},
                )

            # Hallucination guard disabled temporarily
            if not has_sufficient_context(docs):
                return (
                    "I couldn't find enough information in the uploaded documents.",
                    [],
                    {},
                )
            logger.info("Retrieved %d documents", len(docs))

            context = "\n\n".join(doc.page_content for doc in docs)

            recent_messages = self.chat_history.messages[-self.memory_window :]

            history = "\n".join(f"{msg.type}: {msg.content}" for msg in recent_messages)

            messages = self.prompt.invoke(
                {"history": history, "context": context, "question": question}
            )

            try:

                logger.info("LLM generation started")

                with langfuse.start_as_current_observation(name="generation"):

                    response = self.llm.stream(messages)

                logger.info("LLM generation initialized successfully")

            except Exception as e:

                logger.exception("LLM generation failed")

                return (
                    iter(
                        "⚠️ LLM unavailable or rate limit exceeded. Please try again later."
                    ),
                    [],
                    {},
                )

            pages = sorted(set(doc.metadata["page"] + 1 for doc in docs))
            avg_score = round(sum(scores) / len(scores), 2)
            metadata = {
                "num_chunks": len(docs),
                "avg_score": avg_score,
                "max_score": max(scores),
                "pages": pages,
                "retrieval": "Hybrid + MultiQuery",
                "reranker": "CrossEncoder",
            }

            query_obs.update(
                output={"pages": pages, "num_chunks": len(docs), "avg_score": avg_score}
            )

            langfuse.flush()
            return response, pages, metadata
