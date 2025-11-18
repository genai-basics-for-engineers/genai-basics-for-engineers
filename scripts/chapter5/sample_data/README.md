# サンプルデータ

このディレクトリには第5章のRAGデモで使用するサンプルFAQデータが含まれています。

## ファイル一覧

- `faq_01_vpn.txt` - VPN接続に関するFAQ
- `faq_02_password.txt` - パスワードリセットに関するFAQ
- `faq_03_expense.txt` - 経費精算に関するFAQ
- `faq_04_vacation.txt` - 有給休暇申請に関するFAQ
- `faq_05_meeting_room.txt` - 会議室予約に関するFAQ
- `faq_06_printer.txt` - プリンター問題に関するFAQ
- `faq_07_security_card.txt` - セキュリティカード紛失に関するFAQ
- `faq_08_software.txt` - ソフトウェアインストールに関するFAQ
- `faq_09_remote_work.txt` - リモートワーク申請に関するFAQ
- `faq_10_training.txt` - 社内研修申込みに関するFAQ

## フォーマット

各ファイルは以下の形式で記述されています：

```
Q: [質問内容]
A: [回答内容]
```

## 使用方法

これらのファイルは`5-3-1-minimal-rag-chroma.py`などのスクリプトから読み込まれ、
RAGシステムのベクトル検索用データとして使用されます。