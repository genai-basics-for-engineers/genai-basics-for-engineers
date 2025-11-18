#!/usr/bin/env python
"""
RRF（Reciprocal Rank Fusion）の数式画像を生成するスクリプト
"""

import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

# 日本語フォントの設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 14

# LaTeX形式の数式
rrf_formula = r'$\mathrm{RRF}(d) = \sum_{r \in R} \frac{1}{k + r(d)}$'

# 図の作成
fig = plt.figure(figsize=(10, 2))
ax = fig.add_subplot(111)

# 軸を非表示
ax.axis('off')

# 数式を中央に配置
ax.text(0.5, 0.5, rrf_formula,
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax.transAxes,
        fontsize=20)

# 画像として保存
output_dir = Path("../../manuscript/images")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "rrf_formula.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.1)

print(f"RRF数式の画像を生成しました: {output_path}")

# 詳細版（説明付き）も生成
fig2 = plt.figure(figsize=(14, 5))
ax2 = fig2.add_subplot(111)
ax2.axis('off')

# 数式と説明（上下に配置）
formula_text = r'$\mathrm{RRF}(d) = \sum_{r \in R} \frac{1}{k + r(d)}$'

ax2.text(0.5, 0.7, formula_text,
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax2.transAxes,
        fontsize=22)

# 説明（LaTeX形式）
explanation = r'''
where: $d$ = document, $R$ = set of rankings (e.g., BM25, vector search),
$r(d)$ = rank of document $d$ in ranking $r$, $k$ = constant (typically 60)
'''

ax2.text(0.5, 0.25, explanation,
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax2.transAxes,
        fontsize=12,
        style='italic')

output_path2 = output_dir / "rrf_formula_detailed.png"
plt.savefig(output_path2, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.2)

print(f"詳細版のRRF数式画像も生成しました: {output_path2}")

# 具体例付きの画像も生成
fig3 = plt.figure(figsize=(14, 6))
ax3 = fig3.add_subplot(111)
ax3.axis('off')

# 上段：一般式
formula_general = r'$\mathrm{RRF}(d) = \sum_{r \in R} \frac{1}{k + r(d)}$'

ax3.text(0.5, 0.85, formula_general,
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax3.transAxes,
        fontsize=20)

# 中段：具体例（2つの検索手法の場合）
formula_example = r'$= \frac{1}{k + r_{\mathrm{BM25}}(d)} + \frac{1}{k + r_{\mathrm{vector}}(d)}$'

ax3.text(0.5, 0.55, formula_example,
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax3.transAxes,
        fontsize=18)

# 下段：数値例
numeric_example = r'''Example: Document ranked 1st in BM25 and 2nd in vector search with $k=60$:
$\mathrm{RRF} = \frac{1}{60+1} + \frac{1}{60+2} = 0.0164 + 0.0161 = 0.0325$'''

ax3.text(0.5, 0.2, numeric_example,
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax3.transAxes,
        fontsize=12,
        style='italic')

output_path3 = output_dir / "rrf_formula_example.png"
plt.savefig(output_path3, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.2)

print(f"具体例付きのRRF数式画像も生成しました: {output_path3}")

plt.close('all')