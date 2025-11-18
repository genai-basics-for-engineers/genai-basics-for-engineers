#!/usr/bin/env python3
import re
import sys


def extract_text_from_markdown(content):
    """マークダウンから純粋なテキストを抽出（コードブロック内容を含む改良版）"""

    # HTMLコメントを除去
    content = re.sub(r'<!--[\s\S]*?-->', '', content)

    # コードブロックのマークアップのみ除去（内容は残す）
    # ```language と ``` を除去、中身は残す
    content = re.sub(r'^```[^\n]*\n', '', content, flags=re.MULTILINE)
    content = re.sub(r'^```\s*$', '', content, flags=re.MULTILINE)

    # インラインコードのマークアップのみ除去（内容は残す）
    content = re.sub(r'`([^`\n]*)`', r'\1', content)

    # ヘッダーのマークアップ記号のみ除去（内容は残す）
    content = re.sub(r'^#{1,6}\s*', '', content, flags=re.MULTILINE)

    # 箇条書きのマーカーのみ除去（内容は残す）
    content = re.sub(r'^[-*+]\s*', '', content, flags=re.MULTILINE)

    # 番号付きリストのマーカーのみ除去
    content = re.sub(r'^\d+\.\s*', '', content, flags=re.MULTILINE)

    # 太字・斜体マークアップを除去（内容は残す）
    content = re.sub(r'\*\*([^*]+?)\*\*', r'\1', content)
    content = re.sub(r'\*([^*]+?)\*', r'\1', content)
    content = re.sub(r'__([^_]+?)__', r'\1', content)
    content = re.sub(r'_([^_]+?)_', r'\1', content)

    # リンクマークアップを除去（テキスト部分は残す）
    content = re.sub(r'\[([^\]]+?)\]\([^)]+?\)', r'\1', content)

    # HTMLタグを除去
    content = re.sub(r'<[^>]+?>', '', content)

    # 画像マークアップを除去
    content = re.sub(r'!\[[^\]]*?\]\([^)]*?\)', '', content)

    # 引用符マーカーのみ除去
    content = re.sub(r'^>\s*', '', content, flags=re.MULTILINE)

    # 水平線を除去
    content = re.sub(r'^-{3,}$', '', content, flags=re.MULTILINE)
    content = re.sub(r'^={3,}$', '', content, flags=re.MULTILINE)

    # 空行を整理
    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)

    # 行頭・行末の余分な空白を除去
    lines = []
    for line in content.split('\n'):
        cleaned_line = line.strip()
        if cleaned_line:
            lines.append(cleaned_line)
        elif lines and lines[-1] != '':
            lines.append('')

    return '\n'.join(lines)


def count_japanese_chars(text):
    """日本語文字をカウント"""
    japanese_count = 0
    for char in text:
        if '\u3040' <= char <= '\u309F':  # ひらがな
            japanese_count += 1
        elif '\u30A0' <= char <= '\u30FF':  # カタカナ
            japanese_count += 1
        elif '\u4E00' <= char <= '\u9FAF':  # 漢字
            japanese_count += 1
        elif char in '。、！？：；「」『』（）':  # 日本語句読点
            japanese_count += 1
    return japanese_count


def count_practical_chars(text):
    """実用文字数（日本語＋英数字）をカウント"""
    practical_count = 0
    for char in text:
        if '\u3040' <= char <= '\u309F':  # ひらがな
            practical_count += 1
        elif '\u30A0' <= char <= '\u30FF':  # カタカナ
            practical_count += 1
        elif '\u4E00' <= char <= '\u9FAF':  # 漢字
            practical_count += 1
        elif char in '。、！？：；「」『』（）':  # 日本語句読点
            practical_count += 1
        elif char.isalnum():  # 英数字
            practical_count += 1
    return practical_count


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方法: python count_chars.py <markdownファイル>")
        sys.exit(1)

    filename = sys.argv[1]

    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # テキストを抽出
    extracted_text = extract_text_from_markdown(content)

    # 統計計算
    total_chars = len(extracted_text)
    total_no_space = len([c for c in extracted_text if not c.isspace()])
    japanese_chars = count_japanese_chars(extracted_text)
    practical_chars = count_practical_chars(extracted_text)

    print(f"📄 ファイル: {filename}")
    print(f"📏 抽出後の総文字数: {total_chars:,}文字")
    print(f"📐 総文字数（空白除く）: {total_no_space:,}文字")
    print(f"🇯🇵 日本語文字数: {japanese_chars:,}文字")
    print(f"📖 実用文字数（日本語+英数字）: {practical_chars:,}文字")
    print(f"📊 日本語の割合: {japanese_chars/total_no_space*100:.1f}%")

    # 目標との比較
    target = 40000
    remaining = target - practical_chars
    progress = practical_chars / target * 100

    print(f"\n🎯 目標: {target:,}文字")
    print(f"📈 進捗: {progress:.1f}%")
    if remaining > 0:
        print(f"⏳ 残り: {remaining:,}文字")
    else:
        print(f"✅ 目標達成！（{-remaining:,}文字超過）")

    # 抽出結果をファイルに保存
    output_file = filename.replace('.md', '_extracted.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(extracted_text)
    print(f"\n💾 抽出テキストを {output_file} に保存しました")
    print(f"📝 コードブロックの内容も文字数にカウントしています")
