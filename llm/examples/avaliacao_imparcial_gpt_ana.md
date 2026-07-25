# Avaliação imparcial das recomendações — Ana (exemplo)

**Fonte:** resposta de um modelo de linguagem ao prompt em [`prompt_avaliacao_imparcial.md`](prompt_avaliacao_imparcial.md)  
**Caso avaliado:** histórico + top 10 gerados por `llm/recommend_from_history.py`  
**Entrada:** [`historico_exemplo.json`](historico_exemplo.json)  
**Saída do modelo ML:** [`historico_exemplo_recomendacoes.json`](historico_exemplo_recomendacoes.json)

> Documento de **auditoria qualitativa** (humano/LLM). Não substitui métricas do Model Card (`RMSE`, `@K`, etc.).

---

## 1) Perfil inferido da Ana

*(com base apenas no histórico efetivamente usado pelo sistema)*

- Demonstra forte preferência por **dramas** e filmes de **crime** bem avaliados (*The Shawshank Redemption*, *The Usual Suspects*, *Pulp Fiction*).
- Também aprecia **ação comercial** (*GoldenEye*) e **aventuras familiares/fantasia** (*Toy Story*, *Jumanji*), sugerindo abertura a mais de um estilo.
- Parece valorizar filmes considerados **clássicos** ou muito influentes, em vez de obras de nicho.
- Não há evidência suficiente para concluir preferência por **guerra**, **ficção científica**, **documentários**, **musicais** ou **filmes experimentais**.
- O uso de **Heat (1972)** em vez de **Heat (1995)** torna o perfil artificialmente mais voltado para drama do que provavelmente deveria.

---

## 2) Avaliação de cada item do Top 10

| # | Título | Classificação | Justificativa |
|---|--------|---------------|---------------|
| 1 | **Band of Brothers (2001)** | Neutro | Obra bem avaliada e dramática, mas o histórico não indica interesse específico por guerra ou minisséries. |
| 2 | **One Shot (2004)** | Neutro | Sinopse indisponível; obra pouco conhecida no contexto da avaliação — recomendação incerta. |
| 3 | **The Silence of the Lambs (1991)** | Faz sentido | Quem gosta de *Usual Suspects* e *Pulp Fiction* frequentemente aprecia thrilers policiais de alta reputação. |
| 4 | **Pearl Jam: Immagine in Cornice… (2007)** | Fora do perfil | Nada no histórico sugere interesse por filme-concerto ou documentário musical. |
| 5 | **People on Sunday (1930)** | Fora do perfil | Filme mudo experimental é um salto grande sem evidência no histórico. |
| 6 | **Battlestar Galactica (2003)** | Neutro | Sci-fi pode agradar quem gosta de aventura, mas não há sinais claros; além disso é minissérie. |
| 7 | **We Stand Alone Together (2001)** | Fora do perfil | Documentário de veteranos parece derivar de *Band of Brothers*, não do gosto da usuária. |
| 8 | **Star Wars: Episode IV – A New Hope (1977)** | Faz sentido | Aventura clássica; conversa razoavelmente com Toy Story, Jumanji e GoldenEye. |
| 9 | **The Dark Knight (2008)** | Faz sentido | Mistura crime, drama e ação; compatível com Suspects, Pulp Fiction e GoldenEye. |
| 10 | **The Dawn Patrol (1938)** | Fora do perfil | Filme de guerra antigo sem suporte evidente no histórico. |

### Contagem rápida

| Classificação | Qtde |
|---------------|------|
| Faz sentido | 3 |
| Neutro | 3 |
| Fora do perfil | 4 |

---

## 3) Nota global de coerência da lista

### **5 / 10**

**Critério:** há algumas recomendações plausíveis (*Silence of the Lambs*, *The Dark Knight*, *Star Wars*), mas cerca de **metade** da lista parece vir de relações de embedding pouco úteis para este perfil (documentários, filme-concerto, cinema experimental, várias obras de guerra). A lista **não transmite uma linha clara** de preferências.

---

## 4) Principais falhas

1. **Erro de resolução de título (Heat 1972 vs 1995)** — maior fonte provável de distorção do perfil. Se a intenção era o filme de Michael Mann, o embedding usado representa praticamente outro gosto.
2. **Excesso de obras ligadas à guerra** — *Band of Brothers*, *We Stand Alone Together* e *The Dawn Patrol* sem apoio claro no histórico.
3. **Itens claramente de nicho** — *People on Sunday* e o show do Pearl Jam pouco justificáveis.
4. **Mistura de filmes, minisséries e documentários** — se o objetivo era “filmes”, a lista perde consistência.
5. **Pouca exploração do eixo crime/drama**, justamente o mais forte no histórico.

---

## 5) O que mudaria se o Heat correto fosse o de 1995

Com **Heat (1995)**, o perfil tenderia mais a **crime, thriller policial e ação adulta**.

**Mais esperado na lista:**

- thrilers policiais;
- filmes de assalto;
- dramas criminais;
- estilos próximos a Mann / Tarantino / Singer.

**Menos esperado:**

- filmes de guerra;
- documentários militares;
- obras experimentais antigas.

---

## 6) Veredito final

### **Parcialmente.**

A lista contém recomendações convincentes e compatíveis com o histórico, mas também vários itens pouco relacionados ao gosto inferido. O erro **Heat 1995 → 1972** provavelmente prejudicou o embedding do perfil; mesmo assim, a qualidade geral é **apenas mediana**, com inconsistências perceptíveis.

---

## Nota geral do modelo (visão do avaliador)

### **6 / 10**

| Dimensão | Nota | Comentário |
|----------|------|------------|
| Perfil inferido | 7/10 | Mesmo com o erro do Heat, capturou parte do gosto por clássicos, dramas e obras de alta reputação. |
| Qualidade das recomendações | 5/10 | Acertos claros (*Lambs*, *Dark Knight*, *Star Wars*) e ruído perceptível (Pearl Jam, *People on Sunday*, *Dawn Patrol*). |
| Ranking | 5/10 | Itens discutíveis aparecem acima de recomendações mais plausíveis. |
| Robustez ao cold start | 7/10 | Com só 8 avaliações, alguma imprecisão é natural; ainda assim houve relações interessantes. |

### O que essa nota significa?

Um **6/10** não indica que o modelo seja ruim: indica que **funciona**, mas ainda gera **muita recomendação de baixa relevância**.

Reação típica num streaming a essa lista:

| Reação | Estimativa |
|--------|------------|
| Gostaria de ver | 3–4 itens |
| Indiferente | 2–3 itens |
| Ignoraria | 3–4 itens |

Isso é **aceitável para protótipo / primeiro modelo colaborativo**, mas abaixo do esperado em sistemas maduros (Netflix, Spotify, Steam), que costumam manter taxa bem maior de relevância percebida.

### Se o erro do Heat não existisse

Expectativa do avaliador: cerca de **7 a 7,5 / 10**, mantendo a mesma arquitetura.  
O maior problema desta execução específica não foi só o colaborativo, e sim a **resolução de títulos** na entrada — que contaminou o perfil. Em modelos colaborativos, item importante identificado errado desloca as recomendações para a região errada do espaço de embeddings.

---

## Ações sugeridas (time)

1. Melhorar match de títulos (desempate por ano / popularidade / gênero).  
2. Filtrar tipos de mídia (filme vs minissérie vs documentário vs show).  
3. Opcional: re-ranquear com sinal de conteúdo (gênero / overview) além do embedding colaborativo.  
4. Reexecutar o caso da Ana após o fix do *Heat* e comparar esta auditoria.

---

*Arquivo gerado para versionar a auditoria qualitativa junto da pasta `llm/`.*
