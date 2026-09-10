# Setup do banco vetorial (Postgres + pgvector)

## 1. Subir o container

```bash
cp .env.example .env
docker compose up -d
docker compose ps
```

O serviço `pgvector` usa a imagem `pgvector/pgvector:pg16` e expõe a porta `5432` (configurável via `POSTGRES_PORT` no `.env`). A extensão `vector` é criada automaticamente pelo script [docker/init-pgvector.sql](docker/init-pgvector.sql) na primeira inicialização do volume.

Connection string para uso no LangChain (`langchain-postgres` / `psycopg`):

```
postgresql+psycopg://rag_user:rag_password@localhost:5432/rag_db
```

## 2. Problemas comuns no WSL (sem systemd)

Se `sudo systemctl start docker` falhar com `System has not been booted with systemd as init system`, o WSL não usa systemd. Use:

```bash
sudo service docker start
```

### Erro: `ulimit: error setting limit (Invalid argument)`

O script `/etc/init.d/docker` tenta ajustar limites de arquivos abertos que o kernel do WSL pode rejeitar. Se isso travar o início do serviço, suba o daemon manualmente:

```bash
sudo -b sh -c 'dockerd > /tmp/dockerd.log 2>&1'
```

### Erro: `iptables failed (exit status 4)` / `CHAIN_ADD failed`

O kernel do WSL não tem os módulos de netfilter necessários para `iptables-nft`. Troque para o modo legacy e reinicie o daemon:

```bash
sudo update-alternatives --set iptables /usr/sbin/iptables-legacy
sudo update-alternatives --set ip6tables /usr/sbin/ip6tables-legacy
sudo pkill dockerd
sudo -b sh -c 'dockerd > /tmp/dockerd.log 2>&1'
```

### Rodar `docker`/`docker compose` sem `sudo`

Adicione seu usuário ao grupo `docker` (é preciso abrir um novo terminal, ou rodar `newgrp docker`, para a mudança valer):

```bash
sudo usermod -aG docker "$USER"
```

> Como o WSL não tem systemd, o `dockerd` precisa ser iniciado manualmente (via `sudo service docker start` ou o comando `dockerd` acima) toda vez que a máquina WSL reiniciar.

## 3. Validar a instalação

```bash
docker exec rag-pgvector psql -U rag_user -d rag_db -c "SELECT extname FROM pg_extension;"
```

A extensão `vector` deve aparecer na lista.
