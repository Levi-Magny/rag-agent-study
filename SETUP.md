# Setup do banco vetorial (PostgreSQL + pgvector)

Este guia prepara o banco local usado pelo agente. Ele assume que Python 3.11+, [uv](https://docs.astral.sh/uv/) e Docker Compose estao instalados. Os comandos devem ser executados na raiz do repositorio.

## 1. Configurar o ambiente

Crie o arquivo local de configuracao a partir do exemplo versionado:

```bash
cp .env.example .env
```

Revise `.env` antes de continuar. Para uma instalacao local, os valores padrao do banco funcionam sem alteracao. Configure `AWS_PROFILE` ou as credenciais AWS equivalentes e, quando necessario, ajuste `AWS_REGION`, `BEDROCK_MODEL_ID` e `EMBEDDING_MODEL_ID`.

O `.env` nao deve ser commitado: ele pode conter credenciais e configuracoes da maquina. O `.env.example` e seguro para versionamento e documenta apenas valores de desenvolvimento.

## 2. Subir e verificar o container

```bash
docker compose up -d
docker compose ps
```

O servico `pgvector` usa a imagem `pgvector/pgvector:pg16`, persiste seus dados no volume Docker `pgvector_data` e expoe a porta `5432` por padrao. Defina `POSTGRES_PORT` no `.env` para usar outra porta. Na primeira inicializacao de um volume novo, [docker/init-pgvector.sql](docker/init-pgvector.sql) cria a extensao `vector`.

Espere o estado do servico ficar `healthy` e valide a extensao:

```bash
docker exec rag-pgvector psql -U rag_user -d rag_db -c "SELECT extname FROM pg_extension;"
```

A extensao `vector` deve aparecer no resultado. Com as configuracoes padrao, a URL que a aplicacao monta e:

```
postgresql+psycopg://rag_user:rag_password@localhost:5432/rag_db
```

## 3. Migrações de schema (Alembic)

O schema do banco (extensão `vector` + tabela `embeddings`) é versionado com Alembic em [alembic/versions](alembic/versions). `EmbeddingDBControl` não cria mais tabelas automaticamente — é preciso aplicar as migrações antes do primeiro uso:

```bash
uv run alembic upgrade head
```

Execute o comando novamente ao atualizar para uma versao com novas migracoes. A revisao de identidade de origem preenche registros existentes e habilita a recarga idempotente: ao ingerir o mesmo arquivo, seus vetores anteriores sao removidos e substituidos na mesma transacao.

## 4. Configuracao de chunks

Os valores abaixo são opcionais no `.env` e são medidos em caracteres:

```dotenv
CHUNK_SIZE=2000
CHUNK_OVERLAP=300
```

O overlap deve ser menor que o tamanho do chunk. O carregamento agrupa as unidades estruturais adjacentes do Docling antes da divisao final, para que esses parametros tambem tenham efeito quando o documento contiver muitos paragrafos curtos.

## 5. Indexar e consultar documentos

Para substituir os chunks dos arquivos configurados, sem afetar outras origens
já presentes no banco, execute:

```bash
uv run src/my_agent/my_rag.py --reindex
```

Para remover todos os embeddings atuais e reconstruir a base somente com os
arquivos configurados, execute:

```bash
uv run src/my_agent/my_rag.py --reset-and-index
```

O segundo comando e destrutivo e deve ser usado apenas quando uma limpeza total for desejada. Para consultar a base sem reindexar, execute:

```bash
uv run src/my_agent/my_rag.py
```

Ao alterar o modelo ORM em [src/my_agent/db_control.py](src/my_agent/db_control.py), por exemplo ao incluir colunas ou alterar `EMBEDDING_DIM`, gere uma nova revisao e revise o arquivo antes de aplica-la:

```bash
uv run alembic revision --autogenerate -m "descrição da mudança"
uv run alembic upgrade head
```

Para reverter a última migração:

```bash
uv run alembic downgrade -1
```

## Problemas comuns no WSL

Esta secao so se aplica quando o daemon Docker roda dentro do WSL. Com Docker Desktop e integracao WSL habilitada, inicie o Docker Desktop no Windows e use `docker version` para conferir que as secoes `Client` e `Server` aparecem.

### O socket `/var/run/docker.sock` nao existe

Quando `docker compose up -d` retorna `failed to connect to the docker API`, o daemon nao esta em execucao. Em distribuicoes WSL sem systemd, tente:

```bash
sudo service docker start
docker version
```

### `ulimit: error setting limit (Invalid argument)`

O script `/etc/init.d/docker` pode falhar ao ajustar limites que o kernel WSL rejeita. Inicie o daemon diretamente e valide-o:

```bash
sudo -b sh -c 'dockerd > /tmp/dockerd.log 2>&1'
docker version
```

Se o daemon nao iniciar, consulte o motivo antes de tentar o Compose:

```bash
tail -n 100 /tmp/dockerd.log
```

### `iptables failed (exit status 4)` ou `CHAIN_ADD failed`

Alguns ambientes WSL nao oferecem os modulos necessarios para `iptables-nft`. Troque para o modo legacy, reinicie o daemon e repita a validacao:

```bash
sudo update-alternatives --set iptables /usr/sbin/iptables-legacy
sudo update-alternatives --set ip6tables /usr/sbin/ip6tables-legacy
sudo pkill dockerd
sudo -b sh -c 'dockerd > /tmp/dockerd.log 2>&1'
docker version
```

### `permission denied` ao executar `docker`

O daemon esta disponivel, mas o usuario atual nao tem permissao para acessar o socket. Adicione-o ao grupo `docker`, abra um novo terminal ou execute `newgrp docker` para atualizar a sessao:

```bash
sudo usermod -aG docker "$USER"
```

> Em WSL sem systemd, o `dockerd` precisa ser iniciado novamente apos reiniciar a distribuicao.
