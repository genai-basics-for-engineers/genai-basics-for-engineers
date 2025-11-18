#!/usr/bin/env node

const fs = require('fs')
const path = require('path')

/**
 * Chapter2のマージスクリプト
 * 分割されたファイルを統合してchapter2.mdを生成
 */

const CHAPTER2_DIR = path.join(__dirname, '../manuscript/chapter2')
const OUTPUT_FILE = path.join(__dirname, '../manuscript/chapter2.md')

// 分割ファイルの順序定義
const FILE_ORDER = [
  'chapter2-00-intro.md',
  'chapter2-01-basics.md',
  'chapter2-02-system.md',
  'chapter2-03-fewshot.md',
  'chapter2-04-cot.md',
  'chapter2-05-usecases.md',
  'chapter2-06-advanced.md',
  'chapter2-07-evaluation.md',
  'chapter2-08-exercise.md',
  'chapter2-09-context.md',
  'chapter2-99-conclusion.md',
]

/**
 * TOC生成用のヘッダー抽出
 */
function extractHeaders(content) {
  const headers = []
  const lines = content.split('\n')

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    const match = line.match(/^(#{1,6})\s+(.+)$/)
    if (match) {
      const level = match[1].length
      const title = match[2]
      const anchor = title
        .toLowerCase()
        .replace(/[^\w\s-]/g, '')
        .replace(/\s+/g, '-')

      headers.push({
        level,
        title,
        anchor,
        line: i + 1,
      })
    }
  }

  return headers
}

/**
 * TOC生成
 */
function generateTOC(headers) {
  const toc = []
  toc.push(
    '<!-- START doctoc generated TOC please keep comment here to allow auto update -->'
  )
  toc.push("<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->")
  toc.push('')

  headers.forEach((header) => {
    if (header.level >= 2) {
      // h2以下のみ
      const indent = '  '.repeat(header.level - 2)
      const link = `- [${header.title}](#${header.anchor})`
      toc.push(`${indent}${link}`)
    }
  })

  toc.push('')
  toc.push(
    '<!-- END doctoc generated TOC please keep comment here to allow auto update -->'
  )

  return toc.join('\n')
}

/**
 * メインマージ処理
 */
function mergeChapter2() {
  console.log('📚 Chapter2 分割ファイルをマージ中...')

  if (!fs.existsSync(CHAPTER2_DIR)) {
    console.error(`❌ 分割ディレクトリが見つかりません: ${CHAPTER2_DIR}`)
    process.exit(1)
  }

  let mergedContent = ''
  let allHeaders = []

  // 各ファイルを順序通りにマージ
  FILE_ORDER.forEach((filename, index) => {
    const filePath = path.join(CHAPTER2_DIR, filename)

    if (!fs.existsSync(filePath)) {
      console.warn(`⚠️  ファイルが見つかりません: ${filename}`)
      return
    }

    console.log(`📄 マージ中: ${filename}`)
    const content = fs.readFileSync(filePath, 'utf8')

    // ヘッダー情報を収集
    const headers = extractHeaders(content)
    allHeaders.push(...headers)

    // 最初のファイル以外は、重複するh1タイトルを除去
    if (index === 0) {
      mergedContent += content
    } else {
      const lines = content.split('\n')
      const filteredLines = lines.filter((line, i) => {
        // h1タイトル行をスキップ（最初の数行のみ）
        if (i < 5 && line.match(/^#\s+/)) {
          return false
        }
        return true
      })
      mergedContent += '\n\n' + filteredLines.join('\n')
    }
  })

  // TOCを生成して挿入
  const toc = generateTOC(allHeaders)
  mergedContent = mergedContent.replace(
    /<!-- START doctoc generated TOC.*?<!-- END doctoc generated TOC please keep comment here to allow auto update -->/s,
    toc
  )

  // 統合ファイルに書き出し
  fs.writeFileSync(OUTPUT_FILE, mergedContent, 'utf8')

  console.log('✅ マージ完了!')
  console.log(`📁 出力先: ${OUTPUT_FILE}`)
  console.log(`📊 統計:`)
  console.log(`   - ファイル数: ${FILE_ORDER.length}`)
  console.log(`   - 総行数: ${mergedContent.split('\n').length}`)
  console.log(`   - ヘッダー数: ${allHeaders.length}`)
}

// 実行
if (require.main === module) {
  mergeChapter2()
}

module.exports = { mergeChapter2, generateTOC, extractHeaders }
