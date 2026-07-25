# Guia de usabilidade — pasta `llm/`

**Para quem:** time do Tech Challenge, quem for gravar o vídeo STAR ou validar o caso de uso “usuário cola histórico → recebe top 10”.  
**Pasta no repo:** [`llm/`](../llm/)

---

## 1. Objetivo

A pasta `llm/` é um **pipeline de texto** em volta do modelo já treinado:

1. Você informa um **histórico** (nome do filme + nota).  
2. O script encontra esses filmes no catálogo MovieLens/TMDB.  
3. Monta um **perfil cold-start** com os embeddings do PyTorch.  
4. Devolve **top K** filmes **ainda não vistos**, com **título**, **score** e **sinopse** (quando existir).

### O que **não** é

| Mito | Realidade |
|------|-----------|
| “É um ChatGPT / LLM generativo” | **Não.** O nome da pasta é só o “lado texto” da demo. |
| “Treina um modelo novo” | Usa `models/model.pth` + `enriched_metadata.parquet` já existentes. |
| “Usa userId do treino” | Simula usuário **novo** (sem `user_idx`), via média dos embeddings dos filmes curtidos. |

**Para que serve no desafio:** mostrar, de forma leiga, como uma pessoa se beneficia do recomendador (caso de uso / STAR), sem abrir o MLflow.

---

## 2. Estrutura da pasta

```
llm/
├── recommend_from_history.py          # script principal (CLI)
└── examples/
    ├── historico_exemplo.json         # entrada de exemplo (Ana)
    ├── historico_exemplo_recomendacoes.json   # saída gerada (não editar à mão)
    └── prompt_avaliacao_imparcial.md  # prompt para auditar a lista com outro LLM
    └── avaliacao_imparcial_gpt_ana.md # resposta formatada da auditoria (caso Ana)
```

Arquivos relacionados **fora** de `llm/` (testes de regressão do checkpoint):

| Arquivo | Papel |
|---------|--------|
| `tests/fixtures/inference_cases.json` | Casos golden (user_idx/movie_idx + score esperado) |
| `scripts/run_inference_cases.py` | Roda os golden na mão |
| `tests/integration/test_real_inference.py` | Mesmos golden via pytest |

---

## 3. Pré-requisitos

1. Ambiente Python do projeto instalado (`.venv` / `pip install -e ".[dev]"`).  
2. Checkpoint treinado: `models/model.pth`.  
3. Metadados enriquecidos: `data/processed/enriched_metadata.parquet`  
   (título + `overview` TMDB).  
4. Rodar a partir da **raiz do repositório**.

---

## 4. Como usar (fluxo feliz)

### 4.1 Rodar o exemplo da Ana

```bash
python llm/recommend_from_history.py --input llm/examples/historico_exemplo.json --k 10
```

**O que acontece:**

1. Lê o JSON de histórico.  
2. Resolve cada título no catálogo (busca flexível).  
3. Imprime no terminal: histórico reconhecido + top 10.  
4. Grava `llm/examples/<nome>_recomendacoes.json` ao lado do input.

### 4.2 Parâmetros CLI

| Flag | Default | Descrição |
|------|---------|-----------|
| `--input` | *(obrigatório)* | JSON com o histórico |
| `--k` | valor do JSON ou `10` | Quantidade de recomendações |
| `--model` | `models/model.pth` | Checkpoint PyTorch |
| `--metadata` | `data/processed/enriched_metadata.parquet` | Catálogo + sinopses |

### 4.3 Formato do JSON de entrada

```json
{
  "usuario": "Nome opcional (só aparece no relatório)",
  "k": 10,
  "historico": [
    { "titulo": "Toy Story", "nota": 5.0 },
    { "titulo": "Pulp Fiction", "nota": 4.5 }
  ]
}
```

Campos aceitos (sinônimos):

| Campo | Alternativas |
|-------|----------------|
| `historico` | `history` |
| `titulo` | `title` |
| `nota` | `rating` |
| `usuario` | `user` |

**Notas:** use escala MovieLens (~0,5 a 5). Filmes com nota maior pesam mais no perfil.

---

## 5. Como criar novos arquivos para teste

### Passo a passo

1. Copie o exemplo:

```bash
cp llm/examples/historico_exemplo.json llm/examples/historico_meu_teste.json
```

2. Edite `usuario`, `k` e a lista `historico` (títulos em português ou inglês; o match é aproximado).

3. Rode:

```bash
python llm/recommend_from_history.py --input llm/examples/historico_meu_teste.json --k 10
```

4. Confira:
   - terminal (lista legível);
   - `llm/examples/historico_meu_teste_recomendacoes.json` (saída estruturada).

5. *(Opcional)* Use o prompt de `prompt_avaliacao_imparcial.md` trocando histórico/top 10 pelos do seu teste, e peça a um LLM externo se as indicações fazem sentido.

### Dicas de bons testes manuais

| Tipo de teste | Ideia |
|---------------|--------|
| Gênero concentrado | Só animações infantis → espera lista mais “leve” |
| Crime/drama 90s | Pulp Fiction + Usual Suspects + Shawshank |
| Título ambíguo | `"Heat"` — pode casar 1972 em vez de 1995; anote no relatório |
| Título inexistente | Deve aparecer em “Não encontrados” e o resto seguir |
| Poucos filmes | 2–3 itens — perfil mais instável (esperado em cold start) |

### O que **não** colocar no JSON de entrada

- `userId` / `movieId` internos do treino (o script resolve por **nome**).  
- Sinopses (ele busca sozinho no parquet).  
- Listas gigantes desnecessárias (funciona, mas a demo fica ilegível).

---

## 6. Como interpretar a saída

### No terminal

- **Histórico reconhecido** — match feito (query → título oficial no catálogo).  
- **Não encontrados** — título sem correspondência suficiente.  
- **Top K** — `score` é similaridade de embedding (perfil × filme); **maior ≈ mais alinhado** ao perfil montado (não é nota de 0–5).

### No JSON `*_recomendacoes.json`

| Campo | Significado |
|-------|-------------|
| `historico_resolvido` | O que de fato entrou no perfil (com `movieId`, overview, etc.) |
| `nao_encontrados` | Queries que falharam |
| `recomendacoes[].titulo` | Nome no catálogo |
| `recomendacoes[].sinopse` | Overview TMDB ou “Sinopse indisponível.” |
| `recomendacoes[].score` | Similaridade do ranking |

---

## 7. Testes já feitos nesta pasta (sessão atual)

| Teste | Entrada | Resultado |
|-------|---------|-----------|
| Demo Ana | `historico_exemplo.json` (8 filmes) | Top 10 gerado; JSON de saída salvo |
| Match de títulos | Toy Story, Jumanji, Pulp Fiction, etc. | Todos encontrados |
| Caso ambíguo | `"Heat"` | Casou **Heat (1972)** — possível erro vs Heat (1995) |
| Sinopses | Vários do top 10 | Parte com overview; parte “indisponível” |
| Prompt imparcial | `prompt_avaliacao_imparcial.md` | Pronto para colar em LLM externo |

**Atenção:** a qualidade da lista depende do checkpoint e do match de títulos. Itens “estranhos” (ex.: show de Pearl Jam no meio de dramas) podem aparecer porque o modelo é **colaborativo puro**, sem filtro de gênero.

---

## 8. Testes automatizados ligados ao modelo (fora do fluxo texto)

Para validar que o **mesmo** `model.pth` não “mudou de comportamento”:

```bash
python scripts/run_inference_cases.py
pytest tests/integration/test_real_inference.py -v
```

Esses casos usam `user_idx` / `movie_idx` (não nomes). Se você **retreinar**, atualize `tests/fixtures/inference_cases.json`.

---

## 9. Limitações conhecidas

1. Cold start por embeddings de item ≠ usuário completo do treino.  
2. Match de título por similaridade de string — anos/homônimos podem errar.  
3. Sinopse depende do TMDB no parquet (`fetch_status` / coverage).  
4. Score não é “estrelas”; é ranking relativo.  
5. Pasta `llm/` é **demo/usabilidade**, não substitui `dvc repro` nem o Model Card.

---

## 10. Checklist rápido (STAR / apresentação)

- [ ] `models/model.pth` existe  
- [ ] `enriched_metadata.parquet` existe  
- [ ] Rodei `historico_exemplo.json` e mostrei o top 10 no terminal  
- [ ] Expliquei: “histórico → perfil → recomendações com sinopse”  
- [ ] (Opcional) Citei a auditoria com o prompt imparcial  

---

## 11. Referências

- Model Card: [MODEL_CARD.md](MODEL_CARD.md)  
- Relatório da sessão (treino/métricas): [RELATORIO_SESSAO_MODEL_CARD_TREINO_TESTES.md](RELATORIO_SESSAO_MODEL_CARD_TREINO_TESTES.md)  
- Script: [`llm/recommend_from_history.py`](../llm/recommend_from_history.py)
