#!/usr/bin/env python
"""Generate a leigh report for the llm/ demo."""

# ruff: noqa: E501, F841

from __future__ import annotations

import unicodedata
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "RELATORIO_TESTE_LLM_ANA.pdf"
FONT_NAME = "Helvetica"

# Paleta
NAVY = (15, 55, 95)
TEAL = (0, 110, 120)
GREEN = (30, 120, 70)
AMBER = (160, 100, 20)
RED = (150, 40, 40)
GRAY = (70, 70, 70)
LIGHT = (245, 248, 250)
WHITE = (255, 255, 255)


class ReportPDF(FPDF):
    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font("Calibri", "B", 9)
        self.set_text_color(*GRAY)
        self.cell(
            0, 8, "FIAP Tech Challenge 02 — Teste pasta llm/ (caso Ana)", align="L"
        )
        self.ln(4)
        self.set_draw_color(*TEAL)
        self.set_line_width(0.4)
        self.line(15, self.get_y(), 195, self.get_y())
        self.ln(6)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Calibri", "I", 8)
        self.set_text_color(*GRAY)
        self.cell(0, 10, f"Página {self.page_no()}/{{nb}}", align="C")


def _setup(pdf: ReportPDF) -> None:
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.alias_nb_pages()


def _clean(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    replacements = {
        "->": "->",
        "--": "-",
    }
    for old, new in replacements.items():
        ascii_text = ascii_text.replace(old, new)
    return ascii_text


def _banner(pdf: ReportPDF) -> None:
    pdf.set_fill_color(*NAVY)
    pdf.rect(0, 0, 210, 42, style="F")
    pdf.set_xy(15, 12)
    pdf.set_font(FONT_NAME, "B", 18)
    pdf.set_text_color(*WHITE)
    pdf.cell(0, 8, _clean("Teste de usabilidade - pasta llm/"), ln=True)
    pdf.set_x(15)
    pdf.set_font(FONT_NAME, "", 11)
    pdf.cell(
        0,
        7,
        _clean("Movie Rec Sys - FIAP Tech Challenge Fase 02 - Julho/2026"),
        ln=True,
    )
    pdf.ln(18)


def _h1(pdf: ReportPDF, text: str) -> None:
    pdf.set_font(FONT_NAME, "B", 14)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 9, _clean(text), ln=True)
    pdf.set_draw_color(*TEAL)
    pdf.set_line_width(0.6)
    y = pdf.get_y()
    pdf.line(15, y, 80, y)
    pdf.ln(5)


def _h2(pdf: ReportPDF, text: str) -> None:
    pdf.set_font(FONT_NAME, "B", 12)
    pdf.set_text_color(*TEAL)
    pdf.cell(0, 8, _clean(text), ln=True)
    pdf.ln(1)


def _p(pdf: ReportPDF, text: str) -> None:
    pdf.set_font(FONT_NAME, "", 10.5)
    pdf.set_text_color(*GRAY)
    pdf.multi_cell(0, 5.5, _clean(text))
    pdf.ln(2)


def _bullet(pdf: ReportPDF, text: str) -> None:
    pdf.set_font(FONT_NAME, "", 10.5)
    pdf.set_text_color(*GRAY)
    pdf.set_x(18)
    pdf.multi_cell(0, 5.5, _clean(f"-  {text}"))


def _callout(
    pdf: ReportPDF, title: str, body: str, color: tuple[int, int, int]
) -> None:
    pdf.set_fill_color(*LIGHT)
    pdf.set_draw_color(*color)
    pdf.set_line_width(0.8)
    x, y = 15, pdf.get_y()
    pdf.set_font(FONT_NAME, "B", 10.5)
    pdf.set_text_color(*color)
    # estimate height
    pdf.set_xy(x + 3, y + 3)
    start = pdf.get_y()
    pdf.cell(0, 5, _clean(title), ln=True)
    pdf.set_x(x + 3)
    pdf.set_font(FONT_NAME, "", 10)
    pdf.set_text_color(*GRAY)
    pdf.multi_cell(175, 5, _clean(body))
    end = pdf.get_y() + 3
    pdf.rect(x, y, 180, end - y, style="D")
    pdf.set_y(end + 3)


def _table_header(pdf: ReportPDF, cols: list[tuple[str, float]]) -> None:
    pdf.set_fill_color(*NAVY)
    pdf.set_text_color(*WHITE)
    pdf.set_font(FONT_NAME, "B", 9.5)
    for label, w in cols:
        pdf.cell(w, 7, _clean(label), border=0, align="C", fill=True)
    pdf.ln()


def _table_row(
    pdf: ReportPDF,
    cells: list[tuple[str, float, str]],
    fill: bool = False,
) -> None:
    if fill:
        pdf.set_fill_color(238, 242, 245)
    else:
        pdf.set_fill_color(*WHITE)
    pdf.set_text_color(*GRAY)
    pdf.set_font(FONT_NAME, "", 9)
    h = 6.5
    for text, w, align in cells:
        pdf.cell(w, h, _clean(text[:42]), border=0, align=align, fill=True)
    pdf.ln()


def build() -> Path:
    pdf = ReportPDF(orientation="P", unit="mm", format="A4")
    _setup(pdf)
    pdf.set_margins(15, 15, 15)
    pdf.add_page()
    _banner(pdf)

    _h1(pdf, "1. Para que serve este teste?")
    _p(
        pdf,
        "A pasta llm/ do projeto não é um ChatGPT. É uma demonstração em linguagem "
        "simples do nosso recomendador: a pessoa informa filmes que já viu e as notas "
        "que deu; o sistema sugere outros 10 títulos com nome e sinopse.",
    )
    _p(
        pdf,
        "Objetivo para o time: validar o caso de uso real (“como o usuário se beneficia?”) "
        "e gerar material claro para o vídeo STAR e a entrega do Tech Challenge.",
    )
    _callout(
        pdf,
        "Em uma frase",
        "Histórico de filmes + notas  →  perfil no modelo PyTorch  →  Top 10 com sinopse.",
        TEAL,
    )

    _h1(pdf, "2. Como funciona (bem leigo)")
    for line in [
        "Lê um arquivo JSON com o histórico (título + nota).",
        "Procura cada título no catálogo MovieLens + metadados TMDB.",
        "Monta um “gosto médio” com os embeddings dos filmes curtidos (usuário novo / cold start).",
        "Ordena os demais filmes e mostra os 10 mais alinhados, com sinopse quando existe.",
    ]:
        _bullet(pdf, line)
    pdf.ln(2)

    _h1(pdf, "3. Dados usados no teste (usuária Ana)")
    _p(pdf, "Arquivo de entrada: llm/examples/historico_exemplo.json")
    _table_header(
        pdf,
        [("O que Ana digitou", 55), ("Nota", 20), ("O que o sistema encontrou", 105)],
    )
    rows = [
        ("Toy Story", "5,0", "Toy Story (1995)"),
        ("Jumanji", "4,0", "Jumanji (1995)"),
        ("Heat", "5,0", "Heat (1972)  ← atenção: não é o Heat 1995"),
        ("GoldenEye", "4,5", "GoldenEye (1995)"),
        ("Usual Suspects", "5,0", "The Usual Suspects (1995)"),
        ("Pulp Fiction", "4,5", "Pulp Fiction (1994)"),
        ("Shawshank Redemption", "5,0", "The Shawshank Redemption (1994)"),
        ("Forrest Gump", "4,0", "Forrest Gump (1994)"),
    ]
    for i, (a, b, c) in enumerate(rows):
        _table_row(
            pdf,
            [(a, 55, "L"), (b, 20, "C"), (c, 105, "L")],
            fill=i % 2 == 0,
        )
    pdf.ln(3)
    _callout(
        pdf,
        "Ponto de atenção",
        "A busca por “Heat” casou com o filme de 1972, e não com o thriller policial de "
        "1995 (Pacino/De Niro). Isso pode ter distorcido um pouco o perfil da Ana.",
        AMBER,
    )

    _h1(pdf, "4. Resultado retornado pelo nosso modelo (Top 10)")
    _p(
        pdf,
        "Comando: python llm/recommend_from_history.py --input llm/examples/historico_exemplo.json --k 10",
    )
    _table_header(
        pdf,
        [("#", 10), ("Filme sugerido", 95), ("Score", 20), ("Sinopse?", 55)],
    )
    top = [
        ("1", "Band of Brothers (2001)", "0,970", "Indisponível"),
        ("2", "One Shot (2004)", "0,961", "Indisponível"),
        ("3", "The Silence of the Lambs (1991)", "0,958", "Sim"),
        ("4", "Pearl Jam Live in Italy (2007)", "0,956", "Sim"),
        ("5", "People on Sunday (1930)", "0,954", "Sim"),
        ("6", "Battlestar Galactica (2003)", "0,954", "Indisponível"),
        ("7", "We Stand Alone Together (2001)", "0,950", "Sim"),
        ("8", "Star Wars: A New Hope (1977)", "0,949", "Sim"),
        ("9", "The Dark Knight (2008)", "0,948", "Sim"),
        ("10", "The Dawn Patrol (1938)", "0,948", "Sim"),
    ]
    for i, row in enumerate(top):
        _table_row(
            pdf,
            [
                (row[0], 10, "C"),
                (row[1], 95, "L"),
                (row[2], 20, "C"),
                (row[3], 55, "C"),
            ],
            fill=i % 2 == 0,
        )
    pdf.ln(3)
    _p(
        pdf,
        "O “score” não é nota de estrela: é similaridade do perfil da Ana com o filme "
        "(quanto maior, mais alinhado ao embedding). Saída completa em "
        "historico_exemplo_recomendacoes.json.",
    )

    pdf.add_page()
    _h1(pdf, "5. Avaliação imparcial (GPT) sobre essas indicações")
    _p(
        pdf,
        "Usamos um prompt neutro (llm/examples/prompt_avaliacao_imparcial.md) pedindo "
        "a outro modelo para julgar se o Top 10 faz sentido. Resumo da auditoria:",
    )

    _h2(pdf, "Perfil inferido da Ana")
    for line in [
        "Forte gosto por dramas e crime clássicos (Shawshank, Usual Suspects, Pulp Fiction).",
        "Também curte ação (GoldenEye) e aventura familiar (Toy Story, Jumanji).",
        "Valoriza obras influentes; não há sinal claro de guerra, sci-fi, docs ou experimental.",
        "Heat (1972) puxou o perfil mais para “drama” do que o Heat (1995) faria.",
    ]:
        _bullet(pdf, line)
    pdf.ln(2)

    _h2(pdf, "Classificação de cada indicação")
    _table_header(
        pdf,
        [("#", 10), ("Filme", 95), ("Julgamento GPT", 75)],
    )
    judge = [
        ("1", "Band of Brothers", "Neutro"),
        ("2", "One Shot", "Neutro"),
        ("3", "Silence of the Lambs", "Faz sentido"),
        ("4", "Pearl Jam (show)", "Fora do perfil"),
        ("5", "People on Sunday", "Fora do perfil"),
        ("6", "Battlestar Galactica", "Neutro"),
        ("7", "We Stand Alone Together", "Fora do perfil"),
        ("8", "Star Wars IV", "Faz sentido"),
        ("9", "The Dark Knight", "Faz sentido"),
        ("10", "The Dawn Patrol", "Fora do perfil"),
    ]
    for i, (a, b, c) in enumerate(judge):
        _table_row(
            pdf,
            [(a, 10, "C"), (b, 95, "L"), (c, 75, "C")],
            fill=i % 2 == 0,
        )
    pdf.ln(3)
    _p(pdf, "Placar: 3 fazem sentido · 3 neutros · 4 fora do perfil.")

    _h2(pdf, "Notas do avaliador")
    _callout(
        pdf,
        "Coerência da lista: 5/10  ·  Nota geral do modelo neste caso: 6/10  ·  Veredito: PARCIALMENTE",
        "Há acertos claros (Silence of the Lambs, Dark Knight, Star Wars), mas também ruído "
        "(show do Pearl Jam, filme mudo experimental, vários títulos de guerra). Em um streaming, "
        "a reação típica seria: gostar de 3–4, indiferente a 2–3, ignorar 3–4. Aceitável para "
        "protótipo colaborativo; abaixo de sistemas maduros.",
        NAVY,
    )

    _h2(pdf, "Principais falhas apontadas")
    for line in [
        "Erro Heat 1972 vs 1995 (maior distorção do perfil).",
        "Excesso de guerra / documentário militar sem apoio no histórico.",
        "Itens de nicho pouco justificáveis (Pearl Jam, People on Sunday).",
        "Mistura filme + minissérie + documentário.",
        "Pouco aproveitamento do eixo crime/drama, o mais forte do histórico.",
    ]:
        _bullet(pdf, line)
    pdf.ln(3)

    _h1(pdf, "6. O que isso significa para o time?")
    for line in [
        "O pipeline texto → recomendação funciona de ponta a ponta (bom para o STAR).",
        "O modelo colaborativo acerta parte das vezes, mas ainda gera ruído — alinhado ao Model Card.",
        "Melhoria rápida de alto impacto: resolver melhor títulos ambíguos (ex.: Heat + ano).",
        "Melhorias seguintes: filtrar tipo de mídia; opcionalmente usar gênero/sinopse no ranking.",
        "Documentação: docs/GUIA_USABILIDADE_LLM.md e llm/examples/avaliacao_imparcial_gpt_ana.md.",
    ]:
        _bullet(pdf, line)
    pdf.ln(4)

    _h1(pdf, "7. Como reproduzir em 30 segundos")
    _p(
        pdf,
        "Na raiz do repositório (com model.pth e enriched_metadata.parquet):\n\n"
        "python llm/recommend_from_history.py --input llm/examples/historico_exemplo.json --k 10",
    )
    _callout(
        pdf,
        "Arquivos para compartilhar / versionar",
        "Este PDF · GUIA_USABILIDADE_LLM.md · historico_exemplo.json · "
        "historico_exemplo_recomendacoes.json · avaliacao_imparcial_gpt_ana.md · "
        "prompt_avaliacao_imparcial.md",
        GREEN,
    )

    pdf.set_font("Calibri", "I", 9)
    pdf.set_text_color(*GRAY)
    pdf.multi_cell(
        0,
        5,
        "Gerado automaticamente para o time do Tech Challenge 02 · "
        "MovieLens 20M + PyTorch MLP · pasta llm/",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUT))
    return OUT


if __name__ == "__main__":
    path = build()
    print(f"PDF gerado: {path}")
