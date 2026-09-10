# My Agent

This repository contains the implementation of a Retrieval-Augmented Generation (RAG) agent using AWS Bedrock and LangChain. The agent is capable of loading, chunking, and embedding PDF and DOCX documents, and then using these embeddings to answer questions via the Claude model on AWS Bedrock.

## Features
- Load PDF and DOCX documents.
- Chunk documents into manageable pieces while preserving section metadata.
- Embed document chunks using AWS Bedrock embeddings.
- Answer questions using the Claude model on AWS Bedrock.

## Usage

1. Clone the repository:

```bash
git clone https://github.com/your-username/my-agent.git
cd my-agent
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your AWS credentials and environment variables:

```bash
export AWS_PROFILE=your-aws-profile
export AWS_REGION=your-aws-region
export EMBEDDING_MODEL_ID=amazon.titan-embeddings-text-v2:0
```

4. Run the RAG agent:

```bash
python -m my_agent.my_rag
```