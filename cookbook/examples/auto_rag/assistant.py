from phi.llm import LLM
from phi.assistant import Assistant
from phi.knowledge import AssistantKnowledge
from phi.document import Document
from phi.document.reader.pdf import PDFReader
from phi.document.reader.website import WebsiteReader
from phi.utils.log import logger
from phi.tools.duckduckgo import DuckDuckGo
from phi.vectordb.pgvector import PgVector2
from phi.storage.assistant.postgres import PgAssistantStorage
from typing import Optional  # Import Optional type hint

from phi.embedder.base import Embedder
from pydantic import BaseModel

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"


class LocalEmbedder(Embedder, BaseModel):
    dimensions: int = 1536

    def get_embedding_and_usage(self, text: str):
        # Implement the logic to generate embeddings for the input text using a local model
        # Return the embedding vector and usage information
        embedding = [0.5] * self.dimensions  # Placeholder values
        usage = "Local Embedder"
        return embedding, usage


def get_auto_rag_assistant(
    llm_model: str = "llama3:3b",
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Assistant:
    """Get an Auto RAG Assistant with local LLM models and a custom embedder."""

    llm_instance = LLM(model=llm_model)

    description = "You are a helpful Assistant called 'AutoRAG' and your goal is to assist the user in the best way possible."
    instructions = [
        "Given a user query, first ALWAYS search your knowledge base using the `search_knowledge_base` tool to see if you have relevant information.",
        "If you don't find relevant information in your knowledge base, use the `duckduckgo_search` tool to search the internet.",
        "If you need to reference the chat history, use the `get_chat_history` tool.",
        "If the user's question is unclear, ask clarifying questions to get more information.",
        "Carefully read the information you have gathered and provide a clear and concise answer to the user.",
        "Do not use phrases like 'based on my knowledge' or 'depending on the information'.",
    ]

    return Assistant(
        name="auto_rag_assistant",
        run_id=run_id,
        user_id=user_id,
        llm=llm_instance,
        storage=PgAssistantStorage(
            table_name="auto_rag_assistant_llm", db_url=db_url),
        knowledge_base=AssistantKnowledge(
            vector_db=PgVector2(
                db_url=db_url,
                collection="auto_rag_documents_llm",
                # Pass dimensions to LocalEmbedder
                embedder=LocalEmbedder(dimensions=1536),
            ),
            num_documents=3,
        ),
        description=description,
        instructions=instructions,
        show_tool_calls=True,
        search_knowledge=True,
        read_chat_history=True,
        tools=[DuckDuckGo()],
        markdown=True,
        add_chat_history_to_messages=True,
        add_datetime_to_instructions=True,
        debug_mode=debug_mode,
    )
