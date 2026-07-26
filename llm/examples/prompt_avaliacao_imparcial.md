# Prompt — avaliação imparcial das recomendações

Copie o bloco abaixo e cole em um modelo de linguagem (ChatGPT, Claude, Gemini, etc.).

---

## PROMPT (copiar a partir daqui)

```text
Você é um avaliador IMPARCIAL de sistemas de recomendação de filmes.

OBJETIVO
Julgar se as 10 indicações feitas por um modelo de ML fazem sentido em relação ao histórico de gosto da usuária "Ana". Não defenda o modelo nem a usuária: seja crítico, equilibrado e explícito sobre incertezas.

CONTEXTO TÉCNICO (para calibrar a expectativa, não para desculpar o modelo)
- O modelo é colaborativo (embeddings PyTorch): aprende padrões de “quem gostou de X também gostou de Y”, sem usar sinopse/gênero no score.
- A usuária é cold start: o perfil foi montado só com a média ponderada dos embeddings dos filmes do histórico.
- Houve possível erro de resolução de título: a busca por "Heat" casou com Heat (1972), um drama/erótico de Hollywood, e NÃO com Heat (1995) de Michael Mann (assalto/policial). Isso pode distorcer o perfil.

REGRAS DE AVALIAÇÃO
1. Não invente fatos sobre filmes que você não conheça; se não souber, diga “incerto”.
2. Separe: (a) coerência de gosto/gênero/tom; (b) qualidade aparente da lista; (c) itens claramente fora do perfil.
3. Considere o histórico RESOLVIDO pelo sistema (não só o que a usuária digitou).
4. Não use elogios vazios. Se a lista for mediocre ou inconsistente, diga isso.
5. Não assuma demografia da Ana além do que o histórico sugere.

HISTÓRICO QUE A USUÁRIA DIGITOU
- Toy Story — nota 5.0
- Jumanji — nota 4.0
- Heat — nota 5.0
- GoldenEye — nota 4.5
- Usual Suspects — nota 5.0
- Pulp Fiction — nota 4.5
- Shawshank Redemption — nota 5.0
- Forrest Gump — nota 4.0

HISTÓRICO EFETIVAMENTE USADO PELO MODELO (após match no catálogo)
1. Toy Story (1995) — 5.0 — animação/aventura familiar
2. Jumanji (1995) — 4.0 — fantasia/aventura
3. Heat (1972) — 5.0 — drama (NÃO é o thriller policial de 1995)
4. GoldenEye (1995) — 4.5 — ação/espionagem (Bond)
5. The Usual Suspects (1995) — 5.0 — crime/mistério
6. Pulp Fiction (1994) — 4.5 — crime/cult
7. The Shawshank Redemption (1994) — 5.0 — drama carcerário
8. Forrest Gump (1994) — 4.0 — drama/comédia dramática

TOP 10 GERADO PELO MODELO (rank, título, score de similaridade de embedding, sinopse resumida)
1. Band of Brothers (2001) — score 0.970 — sinopse indisponível (minissérie WWII / Easy Company)
2. One Shot (2004) — score 0.961 — sinopse indisponível
3. The Silence of the Lambs (1991) — score 0.958 — thriller psicológico / FBI / Hannibal Lecter
4. Pearl Jam: Immagine in Cornice - Live in Italy 2006 (2007) — score 0.956 — filme de show de rock
5. People on Sunday / Menschen am Sonntag (1930) — score 0.954 — experimental/documental mudo alemão
6. Battlestar Galactica (2003) — score 0.954 — sinopse indisponível (scifi)
7. We Stand Alone Together (2001) — score 0.950 — documentário sobre veteranos de Band of Brothers
8. Star Wars: Episode IV - A New Hope (1977) — score 0.949 — scifi/aventura espacial
9. The Dark Knight (2008) — score 0.948 — super-herói/crime (Batman vs Coringa)
10. The Dawn Patrol (1938) — score 0.948 — drama de guerra aérea WWI

FORMATO OBRIGATÓRIO DA RESPOSTA
1) Perfil inferido da Ana (3–6 bullets), com base só no histórico efetivo.
2) Para CADA item do top 10: classificação em {Faz sentido / Neutro / Fora do perfil} + 1 frase justificando.
3) Nota global de coerência da lista: inteiro de 0 a 10 + justifico.
4) Principais falhas (ex.: erro Heat 1972 vs 1995, itens irrelevantes, falta de diversidade útil, etc.).
5) O que mudaria na lista se o Heat correto fosse o de 1995 (hipótese breve).
6) Veredito final em 2–3 frases: “as indicações fazem sentido?” — sim / parcialmente / não — sem marketing.

Seja direto. Português do Brasil.
```

---

Fim do prompt.
