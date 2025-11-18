#!/usr/bin/env python
"""
コサイン類似度の数式画像を生成するスクリプト
"""

import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

# 日本語フォントの設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 14

# LaTeX形式の数式
cosine_formula = r'$\cos(\theta) = \frac{\vec{A} \cdot \vec{B}}{|\vec{A}| \times |\vec{B}|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \times \sqrt{\sum_{i=1}^{n} B_i^2}}$'

# 図の作成
fig = plt.figure(figsize=(12, 2))
ax = fig.add_subplot(111)

# 軸を非表示
ax.axis('off')

# 数式を中央に配置
ax.text(0.5, 0.5, cosine_formula,
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax.transAxes,
        fontsize=18)

# 画像として保存
output_dir = Path("../../manuscript/images")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "cosine_similarity_formula.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.1)

print(f"コサイン類似度の数式画像を生成しました: {output_path}")

# 詳細版（説明付き）も生成
fig2 = plt.figure(figsize=(14, 5))
ax2 = fig2.add_subplot(111)
ax2.axis('off')

# 数式と説明（上下に配置）
formula_text = r'$\mathrm{cosine\_similarity}(\vec{A}, \vec{B}) = \frac{\vec{A} \cdot \vec{B}}{|\vec{A}| \times |\vec{B}|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \times \sqrt{\sum_{i=1}^{n} B_i^2}}$'

ax2.text(0.5, 0.7, formula_text,
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax2.transAxes,
        fontsize=18)

# 説明（LaTeX形式）
explanation = r'''
where: $\vec{A}, \vec{B}$ = embedding vectors (typically 1536 dimensions),
$\vec{A} \cdot \vec{B}$ = dot product, $|\vec{A}|$ = vector magnitude,
Range: [-1, 1], with 1 = identical, 0 = orthogonal, -1 = opposite
'''

ax2.text(0.5, 0.25, explanation,
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax2.transAxes,
        fontsize=12,
        style='italic')

output_path2 = output_dir / "cosine_similarity_detailed.png"
plt.savefig(output_path2, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.2)

print(f"詳細版のコサイン類似度画像も生成しました: {output_path2}")

# Python実装との対応を示す画像も生成
fig3 = plt.figure(figsize=(14, 4))
ax3 = fig3.add_subplot(111)
ax3.axis('off')

# 上段：数式
formula_simple = r'$\mathrm{cosine\_similarity} = \frac{\vec{A} \cdot \vec{B}}{|\vec{A}| \times |\vec{B}|}$'

ax3.text(0.5, 0.75, formula_simple,
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax3.transAxes,
        fontsize=20)

# 下段：Python実装
python_code = r'\texttt{np.dot(vec\_a, vec\_b) / (np.linalg.norm(vec\_a) * np.linalg.norm(vec\_b))}'

ax3.text(0.5, 0.35, 'Python implementation:',
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax3.transAxes,
        fontsize=12,
        style='italic')

ax3.text(0.5, 0.15, python_code,
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax3.transAxes,
        fontsize=14,
        family='monospace')

output_path3 = output_dir / "cosine_with_python.png"
plt.savefig(output_path3, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.2)

print(f"Python対応版のコサイン類似度画像も生成しました: {output_path3}")

plt.close('all')