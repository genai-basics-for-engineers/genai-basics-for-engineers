#!/usr/bin/env python
"""
RRFの仕組みを詳しく説明するスクリプト
"""

def explain_rrf():
    """RRFの計算を詳細に説明"""

    print("=" * 60)
    print("RRF（Reciprocal Rank Fusion）の仕組み")
    print("=" * 60)

    # 検索結果の例（実際の実行結果に基づく）
    print("\n【元の検索結果】")
    print("\nキーワード検索（BM25スコア）:")
    print("  1位: 文書1（スコア: 0.14）")
    print("  2位: 文書3（スコア: 0.14）※同点だが順序は保持")
    print("  3位: 文書2（スコア: 0.00）")

    print("\nベクトル検索（類似度）:")
    print("  1位: 文書1（類似度: 0.90）")
    print("  2位: 文書3（類似度: 0.89）")
    print("  3位: 文書2（類似度: 0.84）")

    print("\n" + "=" * 60)
    print("【RRFの計算】")
    print("=" * 60)

    print("\nRRFの式: score = 1/(k + rank)")
    print("k = 60（標準的な値）")

    # RRFスコアの計算
    k = 60

    print("\n■文書1のRRFスコア:")
    keyword_rank = 1  # キーワード検索で1位
    vector_rank = 1   # ベクトル検索で1位
    score1_keyword = 1/(k + keyword_rank)
    score1_vector = 1/(k + vector_rank)
    score1_total = score1_keyword + score1_vector

    print(f"  キーワード検索: 1位 → 1/(60+1) = {score1_keyword:.4f}")
    print(f"  ベクトル検索:   1位 → 1/(60+1) = {score1_vector:.4f}")
    print(f"  合計: {score1_total:.4f}")

    print("\n■文書3のRRFスコア:")
    keyword_rank = 2  # キーワード検索で2位（同点でも順序は保持）
    vector_rank = 2   # ベクトル検索で2位
    score3_keyword = 1/(k + keyword_rank)
    score3_vector = 1/(k + vector_rank)
    score3_total = score3_keyword + score3_vector

    print(f"  キーワード検索: 2位 → 1/(60+2) = {score3_keyword:.4f}")
    print(f"  ベクトル検索:   2位 → 1/(60+2) = {score3_vector:.4f}")
    print(f"  合計: {score3_total:.4f}")

    print("\n■文書2のRRFスコア:")
    keyword_rank = 3  # キーワード検索で3位
    vector_rank = 3   # ベクトル検索で3位
    score2_keyword = 1/(k + keyword_rank)
    score2_vector = 1/(k + vector_rank)
    score2_total = score2_keyword + score2_vector

    print(f"  キーワード検索: 3位 → 1/(60+3) = {score2_keyword:.4f}")
    print(f"  ベクトル検索:   3位 → 1/(60+3) = {score2_vector:.4f}")
    print(f"  合計: {score2_total:.4f}")

    print("\n" + "=" * 60)
    print("【最終結果】")
    print("=" * 60)

    results = [
        (1, score1_total),
        (3, score3_total),
        (2, score2_total)
    ]
    results.sort(key=lambda x: x[1], reverse=True)

    for rank, (doc_id, score) in enumerate(results, 1):
        print(f"{rank}位: 文書{doc_id}（RRFスコア: {score:.4f}）")

    print("\n" + "=" * 60)
    print("【RRFの利点】")
    print("=" * 60)
    print()
    print("1. スコアの正規化が不要:")
    print("   - BM25スコア（0.14）とベクトル類似度（0.90）の")
    print("   - 単位が違っても問題ない！")
    print("   - 順位だけを使うから比較可能")
    print()
    print("2. 外れ値に強い:")
    print("   - 極端に高いスコアに引きずられない")
    print("   - 複数の検索手法で安定して上位 = 信頼できる")
    print()
    print("3. シンプルで効果的:")
    print("   - 複雑な重み調整が不要")
    print("   - k=60で大体うまくいく")
    print()
    print("=" * 60)

if __name__ == "__main__":
    explain_rrf()