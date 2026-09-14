"""RAG Agent module for retrieval-augmented generation tasks."""
import argparse
import json
from pathlib import Path
from typing import Any

import boto3
from langchain_aws import BedrockEmbeddings

from my_agent.chunker import chunk_docx, chunk_pdf
from my_agent.config import get_settings
from my_agent.db_control import EmbeddingDBControl
from my_agent.tools import ToolRegistry
from my_agent.tools.db_tools import create_search_by_sections_tool

settings = get_settings()


def build_bedrock_runtime_client() -> boto3.client: # type: ignore
    """Build a Boto3 client for AWS Bedrock runtime.

    Returns:
        boto3.client: A Boto3 client for AWS Bedrock runtime.

    """
    if settings.aws_profile:
        session = boto3.Session(profile_name=settings.aws_profile, region_name=settings.aws_region)
    else:
        session = boto3.Session(region_name=settings.aws_region)

    return session.client("bedrock-runtime")


def ask_claude(prompt: str, bedrock_client: boto3.client) -> str: # type: ignore
    """Ask Claude a question and return the answer.

    Args:
        prompt (str): The question to ask Claude.
        bedrock_client (boto3.client): The Boto3 client for AWS Bedrock runtime.

    Returns:
        str: The answer from Claude.

    """
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "temperature": 0.2,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            },
        ],
    }

    response = bedrock_client.invoke_model(
        modelId=settings.bedrock_model_id,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json"
    )

    payload = json.loads(response["body"].read())
    content_blocks = payload.get("content", [])
    text_parts = [block.get("text", "") for block in content_blocks if block.get("type") == "text"]
    return "\n".join([t for t in text_parts if t]).strip()


def run_agent_loop(
    prompt: str,
    bedrock_client: boto3.client,  # type: ignore
    tool_registry: ToolRegistry | None = None,
    system_prompt: str | None = None,
    max_turns: int = 5,
) -> str:
    """Run an agent loop with Claude on Bedrock, executing tool calls when requested.

    Args:
        prompt (str): User prompt/question.
        bedrock_client (boto3.client): Boto3 client for Bedrock runtime.
        tool_registry (ToolRegistry, optional): Registry containing available tools.
        system_prompt (str, optional): System instructions for Claude.
        max_turns (int, optional): Maximum tool execution turns allowed. Defaults to 5.

    Returns:
        str: Final text answer from Claude.

    """
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    ]

    for turn in range(max_turns):
        body: dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "temperature": 0.2,
            "messages": messages,
        }

        if system_prompt:
            body["system"] = system_prompt

        if tool_registry and tool_registry.to_bedrock_tools():
            body["tools"] = tool_registry.to_bedrock_tools()

        response = bedrock_client.invoke_model(
            modelId=settings.bedrock_model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )

        payload = json.loads(response["body"].read())
        content_blocks = payload.get("content", [])
        stop_reason = payload.get("stop_reason")

        # Append assistant turn to history
        messages.append({"role": "assistant", "content": content_blocks})

        if stop_reason == "tool_use" and tool_registry:
            tool_results = []
            for block in content_blocks:
                if block.get("type") == "tool_use":
                    tool_use_id = block.get("id")
                    tool_name = block.get("name")
                    tool_input = block.get("input", {})
                    print(f"\n[Agente Turno {turn + 1}] Executando ferramenta '{tool_name}' com argumentos: {tool_input}")

                    try:
                        result = tool_registry.execute(tool_name, tool_input)
                    except Exception as err:
                        result = f"Erro ao executar a ferramenta {tool_name}: {err}"

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": str(result),
                        }
                    )

            # Append user tool_results turn to history
            messages.append({"role": "user", "content": tool_results})
        else:
            # Reached end_turn or no tool invoked
            text_parts = [
                block.get("text", "")
                for block in content_blocks
                if block.get("type") == "text"
            ]
            return "\n".join([t for t in text_parts if t]).strip()

    # Fallback if max_turns reached
    text_parts = [
        block.get("text", "")
        for block in content_blocks
        if block.get("type") == "text"
    ]
    return "\n".join([t for t in text_parts if t]).strip()


def load_docs() -> list[tuple[Path, list]]:
    """Load and chunk each configured document, retaining its source path."""
    # Get the grandparent directory of the current file, which is the project root.
    current_dir = Path(__file__).parents[2]

    pdf_path = current_dir / "data" / "normas" / "Abnt_nbr_10520_2023.pdf"
    docx_path = current_dir / "data" / "articles" / "Proposta_Preliminar_de_Pesquisa_ME_LEVI_V2.docx"

    return [(pdf_path, chunk_pdf(pdf_path)), (docx_path, chunk_docx(docx_path))]


def add_documents_to_rag(
    documents: list[tuple[Path, list]], embedding_db_control: EmbeddingDBControl
) -> None:
    """Replace the indexed chunks for every configured source document.

    Args:
        documents (list): Source paths paired with their chunked documents.
        embedding_db_control (EmbeddingDBControl): The embedding database control instance.

    """
    for source_path, chunks in documents:
        ids = embedding_db_control.replace_document(source_path, chunks)
        print(f"Indexed {len(ids)} chunks from {source_path.name}")


def build_rag_prompt(question: str, context_docs: list) -> str:
    """Build a prompt that grounds the question in the retrieved context chunks.

    Args:
        question (str): The user's question.
        context_docs (list): The most relevant chunks retrieved from the vector database.

    Returns:
        str: The prompt to send to the LLM.

    """
    context = "\n\n---\n\n".join(doc for doc in context_docs)
    return (
        "Contexto Inicial:\n"
        f"{context}\n\n"
        f"Pergunta: {question}"
    )


def answer_question(
    question: str,
    embedding_db_control: EmbeddingDBControl,
    bedrock_client: boto3.client, # type: ignore
    k: int = 7,
) -> str:
    """Answer a question using retrieval-augmented generation with tool execution capabilities.

    Args:
        question (str): The user's question.
        embedding_db_control (EmbeddingDBControl): The embedding database control instance.
        bedrock_client (boto3.client): The Boto3 client for AWS Bedrock runtime.
        k (int, optional): The number of context chunks to retrieve. Defaults to 7.

    Returns:
        str: The answer from the LLM, grounded in the retrieved context and tool executions.

    """
    context_docs = embedding_db_control.similarity_search(question, k=k)
    context_parsed = [
        f"Content: {ctxt.page_content}\nfrom: {ctxt.metadata.get('dl_meta', {}).get('origin', {}).get('filename', 'N/A')} "
        f"\nPage: {ctxt.metadata.get('dl_meta', {}).get('origin', {}).get('page_no', 'N/A')} "
        f"\nSection: {ctxt.metadata.get('dl_meta', {}).get('headings', ctxt.metadata.get('section', 'N/A'))}\n"
        for ctxt in context_docs
    ]

    registry = ToolRegistry()
    registry.register(create_search_by_sections_tool(embedding_db_control))

    system_prompt = (
        "Você é um agente especialista que responde perguntas relacionadas a normas e referências bibliográficas (como a NBR 10520:2023).\n"
        "Sempre que possível, tente referenciar corretamente as fontes no contexto (Seção, página, arquivo).\n"
        "Você recebeu um contexto inicial obtido via busca vetorial por similaridade.\n"
        "Se o contexto inicial for suficiente para responder completamente à pergunta, responda de forma concisa e direta.\n"
        "Se o contexto inicial for parcial ou insuficiente, mas citar ou referenciar seções específicas dos documentos "
        "onde a resposta pode estar, USE a ferramenta `search_by_sections` para buscar os trechos dessas seções antes de responder.\n"
        "Se mesmo após buscar nas seções solicitadas a informação não estiver disponível, informe que não encontrou a resposta nas fontes fornecidas."
    )

    prompt = build_rag_prompt(question, context_parsed)
    return run_agent_loop(
        prompt=prompt,
        bedrock_client=bedrock_client,
        tool_registry=registry,
        system_prompt=system_prompt,
    )


def main():
    """Execute the main RAG Agent workflow."""
    parser = argparse.ArgumentParser(description="Query or index the RAG document collection.")
    command_group = parser.add_mutually_exclusive_group()
    command_group.add_argument(
        "--reindex",
        action="store_true",
        help="Replace indexed chunks for the configured source documents.",
    )
    command_group.add_argument(
        "--reset-and-index",
        action="store_true",
        help="Delete every embedding, then index the configured source documents.",
    )
    args = parser.parse_args()

    # Initialize the Bedrock client and embeddings
    bedrock_client = build_bedrock_runtime_client()
    embeddings = BedrockEmbeddings(
        model_id=settings.embedding_model_id,
        client=bedrock_client,
    )
    print('AWS_PROFILE:', settings.aws_profile)
    print('AWS_REGION:', settings.aws_region)
    print(bedrock_client)

    # Initialize the embedding database control
    embedding_db_control = EmbeddingDBControl(
        embeddings=embeddings,
        model_name=settings.embedding_model_id,
    )

#     if args.reset_and_index:
#         deleted_count = embedding_db_control.delete_all_documents()
#         print(f"Deleted {deleted_count} chunks from the vector database")

#     if args.reindex or args.reset_and_index:
#         documents = load_docs()
#         add_documents_to_rag(documents, embedding_db_control)
#         return

#     example_question = """
# A seguinte citação está correta? explique caso não esteja:
# `A ironia seria uma forma implícita de heterogeneidade (AUTHIER-REVUZ, 1982)`
#     """
#     answer = answer_question(example_question, embedding_db_control, bedrock_client)
#     print("Question:", example_question)
#     print("Answer:", answer)

    source_path = Path("data/normas/Abnt_nbr_10520_2023.pdf")
    chunks = embedding_db_control.get_chunks_by_source(source_path)

    payload = [
        {
            "chunk_index": chunk.metadata["_chunk_index"],
            "content": chunk.page_content,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]
    with open("output.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
if __name__ == "__main__":
    main()

