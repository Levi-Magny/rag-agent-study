"""RAG Agent module for retrieval-augmented generation tasks."""
import os
import json

import boto3
from langchain_aws import BedrockEmbeddings
# Set the embedding model ID from the environment variable or use a default value
from my_agent.loaders import load_docx, load_pdf
from my_agent.chunker import chunk_pdf, chunk_docx

AWS_PROFILE = os.getenv("AWS_PROFILE")
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
BEDROCK_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "amazon.titan-embeddings-text-v2:0")


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
        "temperature": 0.7,
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

    chunked_pdf = chunk_pdf(
        "C:\\Users\\lemma\\Documents\\Projetos-Pessoais\\RAG-Agent\\my_agent\\data\\normas\\Abnt_nbr_10520_2023.pdf"
    )
    chunked_docx = chunk_docx(
        "C:\\Users\\lemma\\Documents\\Projetos-Pessoais\\RAG-Agent\\my_agent\\data\\articles\\Proposta_Preliminar_de_Pesquisa_ME_LEVI_V2.docx"
    )

    chunks = chunked_pdf + chunked_docx

    return chunks


def embbed_chunks(chunks: list, embeddings) -> list:
    """Embed a list of document chunks using the provided embeddings.

    Args:
        chunks (list): A list of document chunks.
        embeddings: The embeddings object to use for embedding the chunks.

    Returns:
        list: A list of embeddings for the document chunks.

    """
    return [embeddings.embed(chunk.page_content) for chunk in chunks]


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

    # Load and embed document chunks
    chunks = load_docs()
    embedded_chunks = embbed_chunks(chunks, embeddings)


    # bedrock = boto3.client(
    #     "bedrock",
    #     region_name=AWS_REGION,
    # )

    # print(bedrock.list_foundation_models())
    # example_question = "Olá, meu nome é Levi, qual é o seu?"
    # example_answer = ask_claude(example_question, bedrock_client)
    # print("Question:", example_question)
    # print("Answer:", example_answer)


if __name__ == "__main__":
    main()

