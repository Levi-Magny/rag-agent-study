"""RAG Agent module for retrieval-augmented generation tasks."""
import os
import json

import boto3
from pathlib import Path
from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings
# Set the embedding model ID from the environment variable or use a default value
from my_agent.loaders import load_docx, load_pdf
from my_agent.chunker import chunk_pdf, chunk_docx
from my_agent.db_control import EmbeddingDBControl

load_dotenv()

AWS_PROFILE = os.getenv("AWS_PROFILE")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")


def build_bedrock_runtime_client() -> boto3.client: # type: ignore
    """Build a Boto3 client for AWS Bedrock runtime.

    Returns:
        boto3.client: A Boto3 client for AWS Bedrock runtime.

    """
    if AWS_PROFILE:
        session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    else:
        session = boto3.Session(region_name=AWS_REGION)

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
        modelId=BEDROCK_MODEL_ID,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json"
    )

    payload = json.loads(response["body"].read())
    content_blocks = payload.get("content", [])
    text_parts = [block.get("text", "") for block in content_blocks if block.get("type") == "text"]
    return "\n".join([t for t in text_parts if t]).strip()


def load_docs() -> list:
    """Load PDF and DOCX documents and print the number of pages loaded."""
    # Get the grandparent directory of the current file, which is the project root.
    current_dir = Path(__file__).parents[2]

    chunked_pdf = chunk_pdf(
        current_dir / "data" / "normas" / "Abnt_nbr_10520_2023.pdf"
    )
    chunked_docx = chunk_docx(
        current_dir / "data" / "articles" / "Proposta_Preliminar_de_Pesquisa_ME_LEVI_V2.docx"
    )

    chunks = chunked_pdf + chunked_docx

    return chunks


def add_documents_to_rag(chunks: list, embedding_db_control: EmbeddingDBControl) -> None:
    """Add document chunks to the RAG embedding database.

    Args:
        chunks (list): A list of document chunks to add.
        embedding_db_control (EmbeddingDBControl): The embedding database control instance.

    """
    embedding_db_control.add_documents(chunks)


def build_rag_prompt(question: str, context_docs: list) -> str:
    """Build a prompt that grounds the question in the retrieved context chunks.

    Args:
        question (str): The user's question.
        context_docs (list): The most relevant chunks retrieved from the vector database.

    Returns:
        str: The prompt to send to the LLM.

    """
    context = "\n\n---\n\n".join(doc.page_content for doc in context_docs)
    return (
        "Responda de forma concisa a pergunta usando apenas o contexto abaixo. "
        "Se a resposta não estiver no contexto, diga que não sabe.\n\n"
        f"Contexto:\n{context}\n\n"
        f"Pergunta: {question}"
    )


def answer_question(
    question: str,
    embedding_db_control: EmbeddingDBControl,
    bedrock_client: boto3.client, # type: ignore
    k: int = 4,
) -> str:
    """Answer a question using retrieval-augmented generation.

    Args:
        question (str): The user's question.
        embedding_db_control (EmbeddingDBControl): The embedding database control instance.
        bedrock_client (boto3.client): The Boto3 client for AWS Bedrock runtime.
        k (int, optional): The number of context chunks to retrieve. Defaults to 4.

    Returns:
        str: The answer from the LLM, grounded in the retrieved context.

    """
    context_docs = embedding_db_control.similarity_search(question, k=k)
    prompt = build_rag_prompt(question, context_docs)
    return ask_claude(prompt, bedrock_client)


def main():
    """Execute the main RAG Agent workflow."""
    # Initialize the Bedrock client and embeddings
    bedrock_client = build_bedrock_runtime_client()
    embeddings = BedrockEmbeddings(
        model_id=EMBEDDING_MODEL_ID,
        client=bedrock_client,
    )
    print('AWS_PROFILE:', AWS_PROFILE)
    print('AWS_REGION:', AWS_REGION)
    print(bedrock_client)

    # Initialize the embedding database control
    embedding_db_control = EmbeddingDBControl(
        embeddings=embeddings,
        model_name=EMBEDDING_MODEL_ID,
    )

    # Load and embed document chunks only if the table is still empty
    if embedding_db_control.status()["row_count"] == 0:
        chunks = load_docs()
        add_documents_to_rag(chunks, embedding_db_control)

    example_question = "Quem é Levi Magny?"
    answer = answer_question(example_question, embedding_db_control, bedrock_client)
    print("Question:", example_question)
    print("Answer:", answer)


if __name__ == "__main__":
    main()

