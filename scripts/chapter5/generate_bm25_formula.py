#!/usr/bin/env python
"""
BM25数式の画像を生成するスクリプト
"""

import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

# 日本語フォントの設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 14

# LaTeX形式の数式
bm25_formula = r'$\mathrm{BM25}(d,q) = \sum_{i=1}^{n} \mathrm{IDF}(q_i) \cdot \frac{f(q_i, d) \cdot (k_1 + 1)}{f(q_i, d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\mathrm{avgdl}})}$'

# 図の作成
fig = plt.figure(figsize=(12, 2))
ax = fig.add_subplot(111)

# 軸を非表示
ax.axis('off')

# 数式を中央に配置
ax.text(0.5, 0.5, bm25_formula,
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax.transAxes,
        fontsize=18)

# 画像として保存
output_dir = Path("../../manuscript/images")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "bm25_formula.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.1)

print(f"BM25数式の画像を生成しました: {output_path}")

# 各要素の説明付き画像も生成（英語版）
fig2 = plt.figure(figsize=(14, 5))
ax2 = fig2.add_subplot(111)
ax2.axis('off')

# 数式と説明（上下に配置）
formula_text = r'$\mathrm{BM25}(d,q) = \sum_{i=1}^{n} \mathrm{IDF}(q_i) \cdot \frac{f(q_i, d) \cdot (k_1 + 1)}{f(q_i, d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\mathrm{avgdl}})}$'

ax2.text(0.5, 0.7, formula_text,
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax2.transAxes,
        fontsize=20)

# 英語の説明（LaTeX形式）
explanation = r'''
where: $q$ = query, $d$ = document, $f(q_i, d)$ = term frequency of $q_i$ in $d$,
$\mathrm{IDF}(q_i)$ = inverse document frequency of $q_i$, $|d|$ = document length,
$\mathrm{avgdl}$ = average document length, $k_1 = 1.2$, $b = 0.75$ (typical values)
'''

ax2.text(0.5, 0.25, explanation,
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax2.transAxes,
        fontsize=12,
        style='italic')

output_path2 = output_dir / "bm25_formula_detailed.png"
plt.savefig(output_path2, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.2)

print(f"詳細版のBM25数式画像も生成しました: {output_path2}")

plt.close('all')