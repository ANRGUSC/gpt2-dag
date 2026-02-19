from __future__ import annotations

from pathlib import Path


def _svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">',
        '<polygon points="0 0, 10 3.5, 0 7" fill="#222"/>',
        "</marker>",
        "</defs>",
    ]


def _box(x: int, y: int, w: int, h: int, label: str, fill: str = "#e8eefc") -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" ry="8" fill="{fill}" stroke="#2b2b2b" stroke-width="1"/>'
        f'<text x="{x + w/2}" y="{y + h/2 + 5}" font-size="14" text-anchor="middle" font-family="Arial">{label}</text>'
    )


def _arrow(x1: int, y1: int, x2: int, y2: int) -> str:
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#222" stroke-width="1.5" marker-end="url(#arrow)"/>'


def render_layer_schematic(path: Path) -> None:
    width, height = 1500, 430
    lines = _svg_header(width, height)
    lines.append('<text x="20" y="28" font-size="20" font-family="Arial" font-weight="bold">GPT-2 Tensor DAG (One Layer, Sh=12)</text>')

    lines.append(_box(50, 185, 120, 50, "qkv_l"))
    lines.append(_box(620, 185, 140, 50, "attn_merge_l", "#ffe9d6"))
    lines.append(_box(1310, 185, 140, 50, "mlp_merge_l", "#ffe9d6"))
    lines.append(_arrow(170, 210, 200, 210))

    # 12 attention shard tasks in two rows.
    for s in range(12):
        row = 0 if s < 6 else 1
        col = s if s < 6 else s - 6
        x = 210 + col * 65
        y = 95 + row * 145
        lines.append(_box(x, y, 58, 40, f"a{s}", "#d9f2e6"))
        lines.append(_arrow(170, 210, x, y + 20))
        lines.append(_arrow(x + 58, y + 20, 620, 210))

    lines.append(_arrow(760, 210, 810, 210))
    # 12 mlp shard tasks in two rows.
    for s in range(12):
        row = 0 if s < 6 else 1
        col = s if s < 6 else s - 6
        x = 820 + col * 75
        y = 95 + row * 145
        lines.append(_box(x, y, 68, 40, f"m{s}", "#d9f2e6"))
        lines.append(_arrow(760, 210, x, y + 20))
        lines.append(_arrow(x + 68, y + 20, 1310, 210))

    lines.append('<text x="210" y="78" font-size="13" font-family="Arial">attn_shard_l_0..11</text>')
    lines.append('<text x="820" y="78" font-size="13" font-family="Arial">mlp_shard_l_0..11</text>')
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def render_full_overview(path: Path) -> None:
    width, height = 2000, 340
    lines = _svg_header(width, height)
    lines.append('<text x="20" y="28" font-size="20" font-family="Arial" font-weight="bold">GPT-2 Tensor DAG (12 Layer Overview)</text>')
    lines.append(_box(40, 145, 130, 62, "embed", "#f0f0f0"))
    prev_x, prev_w = 40, 130
    y = 140
    h = 74
    w = 170

    shown_layers = [0, 1, 2, 3]
    x = 220
    for i in shown_layers:
        lines.append(_box(x, y, w, h, f"L{i:02d}", "#e8eefc"))
        lines.append(f'<text x="{x + w/2}" y="{y + 49}" font-size="12" text-anchor="middle" font-family="Arial">qkv-&gt;12a-&gt;am-&gt;12m-&gt;mm</text>')
        lines.append(_arrow(prev_x + prev_w, y + h / 2, x, y + h / 2))
        prev_x, prev_w = x, w
        x += 200

    lines.append(_box(x, y + 8, 140, h - 16, "L04..L09", "#fff4d6"))
    lines.append(f'<text x="{x + 70}" y="{y + 45}" font-size="26" text-anchor="middle" font-family="Arial">...</text>')
    lines.append(_arrow(prev_x + prev_w, y + h / 2, x, y + h / 2))
    prev_x, prev_w = x, 140
    x += 190

    for i in [10, 11]:
        lines.append(_box(x, y, w, h, f"L{i:02d}", "#e8eefc"))
        lines.append(f'<text x="{x + w/2}" y="{y + 49}" font-size="12" text-anchor="middle" font-family="Arial">qkv-&gt;12a-&gt;am-&gt;12m-&gt;mm</text>')
        lines.append(_arrow(prev_x + prev_w, y + h / 2, x, y + h / 2))
        prev_x, prev_w = x, w
        x += 200

    lines.append(_box(x, 145, 100, 62, "ln_f", "#f0f0f0"))
    lines.append(_arrow(prev_x + prev_w, y + h / 2, x, y + h / 2))
    lines.append(_box(x + 130, 145, 130, 62, "lm_head", "#f0f0f0"))
    lines.append(_arrow(x + 100, 176, x + 130, 176))
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    figs = root / "docs" / "figures"
    render_layer_schematic(figs / "tensor_dag_layer_sh12.svg")
    render_full_overview(figs / "tensor_dag_full_overview.svg")
    print(f"Generated {figs / 'tensor_dag_layer_sh12.svg'}")
    print(f"Generated {figs / 'tensor_dag_full_overview.svg'}")


if __name__ == "__main__":
    main()
