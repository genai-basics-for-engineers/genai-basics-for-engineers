#!/usr/bin/env node

const fs = require('fs')
const path = require('path')

/**
 * Chapter2の分割スクリプト
 * 巨大なchapter2.mdを管理しやすいファイルに分割
 */

const SOURCE_FILE = path.join(__dirname, '../manuscript/chapter2.md')
const OUTPUT_DIR = path.join(__dirname, '../manuscript/chapter2')

// 分割ポイントの定義（見出しパターン）
const SPLIT_POINTS = [
  {
    pattern: /^# 第 2 章 Prompt Engineering に入門してみよう$/,
    filename: 'chapter2-00-intro.md',
    description: 'はじめに・概要',
  },
  {
    pattern: /^## 2-1 LLM の基本メカニズムを理解する$/,
    filename: 'chapter2-01-basics.md',
    description: 'LLMの基本メカニズム',
  },
  {
    pattern: /^## 2-2 System Prompt：AI の役割と制約を設計する$/,
    filename: 'chapter2-02-system.md',
    description: 'System Prompt',
  },
  {
    pattern: /^## 2-3 Few-Shot：例示で学習させる$/,
    filename: 'chapter2-03-fewshot.md',
    description: 'Few-Shot学習',
  },
  {
    pattern: /^## 2-4 Chain-of-Thought：段階的思考の誘導$/,
    filename: 'chapter2-04-cot.md',
    description: 'Chain-of-Thought',
  },
  {
    pattern: /^## 2-5 実践：業務で使えるプロンプト集$/,
    filename: 'chapter2-05-usecases.md',
    description: '業務ユースケース',
  },
  {
    pattern: /^## 2-6 AI に考えさせるプロンプト設計$/,
    filename: 'chapter2-06-advanced.md',
    description: '高度なプロンプト設計',
  },
  {
    pattern: /^## 2-7 プロンプトの品質測定・改善$/,
    filename: 'chapter2-07-evaluation.md',
    description: '品質測定・改善',
  },
  {
    pattern: /^## 2-8 演習：業務プロンプト集を作ろう$/,
    filename: 'chapter2-08-exercise.md',
    description: '演習',
  },
  {
    pattern: /^## 2-9 発展：コンテキストエンジニアリング$/,
    filename: 'chapter2-09-context.md',
    description: 'コンテキストエンジニアリング',
  },
  {
    pattern:
      /^## まとめ：プロンプトエンジニアリングを身につけた今、何ができるか？$/,
    filename: 'chapter2-99-conclusion.md',
    description: 'まとめ・展望',
  },
]

/**
 * ファイル分割処理
 */
function splitChapter2() {
  console.log('✂️  Chapter2を分割中...')

  if (!fs.existsSync(SOURCE_FILE)) {
    console.error(`❌ ソースファイルが見つかりません: ${SOURCE_FILE}`)
    process.exit(1)
  }

  // 出力ディレクトリ作成
  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true })
    console.log(`📁 ディレクトリ作成: ${OUTPUT_DIR}`)
  }

  const content = fs.readFileSync(SOURCE_FILE, 'utf8')
  const lines = content.split('\n')

  let currentSection = null
  let currentContent = []
  let splitCount = 0

  // 各行を処理
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]

    // 分割ポイントをチェック
    const splitPoint = SPLIT_POINTS.find((sp) => sp.pattern.test(line))

    if (splitPoint) {
      // 前のセクションを保存
      if (currentSection && currentContent.length > 0) {
        saveSection(currentSection, currentContent)
        splitCount++
      }

      // 新しいセクション開始
      currentSection = splitPoint
      currentContent = [line]
      console.log(`📝 セクション開始: ${splitPoint.description}`)
    } else if (currentSection) {
      // 現在のセクションに行を追加
      currentContent.push(line)
    } else {
      // まだセクションが始まっていない（通常は最初のセクション）
      if (currentContent.length === 0) {
        currentSection = SPLIT_POINTS[0]
      }
      currentContent.push(line)
    }
  }

  // 最後のセクションを保存
  if (currentSection && currentContent.length > 0) {
    saveSection(currentSection, currentContent)
    splitCount++
  }

  console.log('✅ 分割完了!')
  console.log(`📊 統計:`)
  console.log(`   - 分割数: ${splitCount}`)
  console.log(`   - 元ファイル: ${lines.length}行`)
  console.log(`   - 出力先: ${OUTPUT_DIR}`)

  // 分割ファイル一覧を表示
  console.log('\n📋 分割ファイル一覧:')
  SPLIT_POINTS.forEach((sp) => {
    const filePath = path.join(OUTPUT_DIR, sp.filename)
    if (fs.existsSync(filePath)) {
      const stats = fs.statSync(filePath)
      const lines = fs.readFileSync(filePath, 'utf8').split('\n').length
      console.log(
        `   ✓ ${sp.filename} (${lines}行, ${Math.round(stats.size / 1024)}KB)`
      )
    } else {
      console.log(`   ❌ ${sp.filename} (作成されませんでした)`)
    }
  })
}

/**
 * セクションをファイルに保存
 */
function saveSection(section, content) {
  const filePath = path.join(OUTPUT_DIR, section.filename)
  const fileContent = content.join('\n')

  fs.writeFileSync(filePath, fileContent, 'utf8')
  console.log(`💾 保存: ${section.filename} (${content.length}行)`)
}

/**
 * README生成
 */
function generateReadme() {
  const readmePath = path.join(OUTPUT_DIR, 'README.md')
  const readme = `# Chapter 2: Prompt Engineering 分割ファイル

このディレクトリには、Chapter 2 の分割されたMarkdownファイルが含まれています。

## ファイル構成

${SPLIT_POINTS.map(
  (sp, i) => `${i + 1}. **${sp.filename}**: ${sp.description}`
).join('\n')}

## 使用方法

### 統合ファイル生成
\`\`\`bash
npm run merge:chapter2
# または
node scripts/merge-chapter2.js
\`\`\`

### 分割ファイル再生成
\`\`\`bash
npm run split:chapter2  
# または
node scripts/split-chapter2.js
\`\`\`

## 編集ガイドライン

1. **分割ファイルを編集**: 各セクションは対応する分割ファイルで編集
2. **統合ファイル生成**: 編集後は必ずマージスクリプトを実行
3. **TOC更新**: 統合後は \`npm run toc\` で目次を更新
4. **lintチェック**: \`npm run lint\` で品質チェック

## 注意事項

- 統合版 \`chapter2.md\` は自動生成されるため直接編集しないでください
- 分割ファイルの構造（ヘッダーレベルなど）を変更する際は注意してください
- 新しいセクションを追加する場合は、分割スクリプトの設定も更新してください
`

  fs.writeFileSync(readmePath, readme, 'utf8')
  console.log(`📖 README生成: ${readmePath}`)
}

// 実行
if (require.main === module) {
  splitChapter2()
  generateReadme()
}

module.exports = { splitChapter2, saveSection }
