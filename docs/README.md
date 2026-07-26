# 📚 Documentação Técnica do Projeto

Esta pasta contém a documentação técnica, especificação do modelo e guias de reprodução do sistema de recomendação.

---

## 📑 Índice da Documentação

| Documento | Descrição |
|-----------|-----------|
| 📋 [MODEL_CARD.md](MODEL_CARD.md) | **Model Card Oficial:** Arquitetura do modelo PyTorch MLP, métricas reais de avaliação, limitações e análises éticas. |
| 📝 [RELATORIO_SESSAO_FINAL.md](RELATORIO_SESSAO_FINAL.md) | **Relatório da Sessão:** Detalhamento das ações, refatorações, testes e sanitizações realizadas na sessão final. |
| 🛠️ [DOCUMENTACAO_ETAPA2.md](DOCUMENTACAO_ETAPA2.md) | **Infraestrutura e Validação:** Dependências, configurações Pydantic Settings, `.env` e funcionamento do `validate_env.py`. |
| 🧠 [IMPLEMENTACAO_MLP_PYTORCH.md](IMPLEMENTACAO_MLP_PYTORCH.md) | **Arquitetura Neural:** Detalhamento das camadas de embedding, MLP, otimizador Adam, loss MSE e Early Stopping. |
| 🌐 [FLUXO_COLETA_METADADOS.md](FLUXO_COLETA_METADADOS.md) | **Enriquecimento TMDB:** Processo de scraping e enriquecimento offline via API do TMDB para catálogo de produtos. |
| 📊 [METADATA_COVERAGE_REPORT.md](METADATA_COVERAGE_REPORT.md) | **Relatório de Cobertura:** Taxa de correspondência e integridade dos metadados extraídos. |
| 🤖 [GUIA_USABILIDADE_LLM.md](GUIA_USABILIDADE_LLM.md) | **Integração com LLM:** Guia de recomendação em linguagem natural a partir do histórico do cliente (`llm/`). |
| 📑 [RELATORIO_TESTE_LLM_ANA.pdf](RELATORIO_TESTE_LLM_ANA.pdf) | **Laudo de Testes LLM:** Relatório dos experimentos com o avaliador em linguagem natural. |
| 📌 [Tech Challenge Fase 02.pdf](Tech%20Challenge%20Fase%2002.pdf) | **Requisitos do Projeto:** Especificação dos requisitos exigidos no desafio da FIAP. |

---
*Para instruções de execução rápida e deploy, consulte o [README.md principal da raiz do projeto](../README.md).*
