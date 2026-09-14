# My Agent

Agente de Retrieval-Augmented Generation (RAG) para consultar documentos PDF e DOCX com AWS Bedrock, Claude e PostgreSQL com pgvector. Ele transforma os arquivos em chunks estruturados, cria embeddings, persiste o resultado e recupera o contexto mais relevante antes de elaborar uma resposta.

## O que o agente faz hoje

- Carrega os documentos PDF e DOCX configurados em `data/normas` e `data/articles`.
- Preserva metadados estruturais fornecidos pelo conversor, como arquivo de origem, pagina e hierarquia de secoes.
- Agrupa unidades adjacentes da mesma secao e as divide em chunks configuraveis, com overlap em caracteres.
- Gera embeddings com Amazon Titan no AWS Bedrock e os armazena em PostgreSQL com a extensao pgvector.
- Substitui atomicamente os chunks de cada arquivo reindexado, evitando duplicatas para uma mesma origem.
- Recupera os chunks mais proximos de uma pergunta por distancia cosseno e tambem pode buscar chunks por titulo exato de secao.
- Envia o contexto recuperado ao Claude no Bedrock e permite que ele solicite a ferramenta de busca por secao durante a resposta.

## Componentes do pipeline

### Docling

[Docling](https://github.com/docling-project/docling) faz a conversao estruturada dos arquivos de entrada. Em [src/my_agent/loaders.py](src/my_agent/loaders.py), `DocumentConverter` recebe opcoes especificas para PDF e DOCX, enquanto `DoclingLoader`, da integracao `langchain-docling`, transforma a saida em documentos compatíveis com LangChain.

O resultado inclui `dl_meta`, usado pelo projeto para preservar a origem, a pagina e os titulos hierarquicos (`headings`). OCR fica desabilitado na configuracao atual; portanto, PDFs que sejam apenas imagens precisam de uma configuracao adicional para extrair texto.

### LangChain

LangChain define o formato comum `Document` usado em todas as etapas. Os carregadores retornam documentos, [src/my_agent/chunker.py](src/my_agent/chunker.py) enriquece seus metadados e usa `RecursiveCharacterTextSplitter` para produzir os chunks finais, e [src/my_agent/db_control.py](src/my_agent/db_control.py) os reconstrói ao consultar a base.

A interface `Embeddings` permite que o controle de banco permaneça independente do provedor. A aplicacao concreta e `BedrockEmbeddings`, de `langchain-aws`, criada no fluxo principal em [src/my_agent/my_rag.py](src/my_agent/my_rag.py) para gerar vetores com o modelo configurado no ambiente.

### SQLAlchemy

[SQLAlchemy](https://www.sqlalchemy.org/) e a camada ORM entre a aplicacao e o PostgreSQL. `EmbeddingRow`, em [src/my_agent/db_control.py](src/my_agent/db_control.py), mapeia a tabela `embeddings`: texto do chunk, vetor pgvector, metadados `JSONB`, modelo utilizado e a identidade/ordem da origem.

`EmbeddingDBControl` cria a engine a partir das configuracoes centralizadas, abre sessoes para gravar e consultar dados e faz a reindexacao de cada arquivo dentro de uma transacao. Na recuperacao vetorial, a expressao `embedding.cosine_distance(...)` do tipo `Vector` do pgvector ordena os registros pelo vetor mais proximo.

### Alembic

[Alembic](https://alembic.sqlalchemy.org/) versiona o schema do banco que o SQLAlchemy usa. [alembic/env.py](alembic/env.py) obtém a URL de conexao do mesmo `Settings` da aplicacao e usa `Base.metadata` para que `alembic revision --autogenerate` compare os modelos ORM com o banco.

As revisoes em [alembic/versions](alembic/versions) criam a extensao `vector` e a tabela `embeddings`, depois acrescentam `source_id` e `chunk_index`. Essas colunas identificam os chunks de cada arquivo e garantem uma ordem unica para a reindexacao idempotente.

## Uso

1. Instale as dependencias com [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

2. Configure as credenciais AWS e as variaveis de ambiente. Consulte [SETUP.md](SETUP.md) para criar o `.env`, iniciar o PostgreSQL/pgvector e aplicar as migracoes.

`CHUNK_SIZE` e `CHUNK_OVERLAP` são medidos em caracteres. O carregador preserva
as seções retornadas pelo Docling, agrupa unidades estruturais adjacentes da
mesma seção e aplica o overlap no chunk final. `CHUNK_OVERLAP` deve ser menor
que `CHUNK_SIZE`.

Cada execução de ingestão substitui, em uma transação, os vetores antes
associados a cada arquivo de origem. Assim, alterar e executar novamente o
script atualiza os chunks daquele arquivo sem duplicar os registros.

## Recuperar chunks por seção

Use `get_chunks_by_sections()` para recuperar todos os chunks que contenham uma
heading exata no metadata. Uma lista usa semântica de união: o chunk é retornado
quando tiver qualquer uma das headings solicitadas.

```python
chunks = embedding_db_control.get_chunks_by_sections(
	["6.1 Sistema autor-data", "7.2 Citação indireta"]
)
```

O método consulta tanto `metadata.section` quanto
`metadata.dl_meta.headings`, retorna os chunks na ordem de cada documento e é
independente de `similarity_search()`, que continua sendo a busca vetorial.

4. Reindexar somente os arquivos configurados, preservando outras origens:

```bash
uv run src/my_agent/my_rag.py --reindex
```

5. Limpar a base vetorial inteira e reindexar os arquivos configurados:

```bash
uv run src/my_agent/my_rag.py --reset-and-index
```

`--reset-and-index` é destrutivo: todos os embeddings existentes são removidos
antes de carregar os documentos configurados.

6. Consultar a base atual:

```bash
uv run src/my_agent/my_rag.py
```