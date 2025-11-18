#!/bin/bash

# Git worktree setup script with automatic file copying
# Usage: ./scripts/setup-worktree.sh <branch-name>

set -e

# 引数チェック
if [ $# -ne 1 ]; then
  echo "Usage: $0 <branch-name>"
  echo "Example: $0 feature/new-feature"
  exit 1
fi

BRANCH_NAME="$1"

# メインディレクトリを取得
MAIN_DIR=$(git rev-parse --show-toplevel)
if [ ! -d "$MAIN_DIR" ]; then
  echo "Error: Could not find git repository root"
  exit 1
fi

# Worktreeディレクトリ名を決定（現在のディレクトリ内に作成）
WORKTREE_DIR="${MAIN_DIR}/worktree-${BRANCH_NAME//\//-}"  # スラッシュをハイフンに置換

echo "🌲 Setting up git worktree..."
echo "Branch: $BRANCH_NAME"
echo "Worktree directory: $WORKTREE_DIR"

# ブランチが存在するかチェック
if ! git show-ref --verify --quiet "refs/heads/$BRANCH_NAME"; then
  echo "📝 Branch '$BRANCH_NAME' does not exist. Creating it..."
  git checkout -b "$BRANCH_NAME"
  git checkout -  # 元のブランチに戻る
fi

# Worktreeを作成
echo "🔧 Creating git worktree..."
git worktree add "$WORKTREE_DIR" "$BRANCH_NAME"

# 必要ファイルの存在チェック関数
check_and_copy() {
  local file_path="$1"
  local source_path="$MAIN_DIR/$file_path"
  local dest_path="$WORKTREE_DIR/$file_path"
  
  if [ -e "$source_path" ]; then
    # ディレクトリの場合は-rオプションで再帰コピー
    if [ -d "$source_path" ]; then
      cp -r "$source_path" "$dest_path"
      echo "📁 Copied directory: $file_path"
    else
      cp "$source_path" "$dest_path"
      echo "📄 Copied file: $file_path"
    fi
  else
    echo "⚠️  Warning: $file_path not found in main directory"
  fi
}

echo "📦 Copying required files..."

# 必要ファイルをコピー
check_and_copy ".envrc"
check_and_copy ".env"

# .claudeディレクトリは特別処理（ネスト問題を防ぐため）
if [ -d "$MAIN_DIR/.claude" ]; then
  mkdir -p "$WORKTREE_DIR/.claude"
  # 設定ファイルのみをコピー（ディレクトリの再帰コピーは避ける）
  for file in settings.json settings.local.json; do
    if [ -f "$MAIN_DIR/.claude/$file" ]; then
      cp "$MAIN_DIR/.claude/$file" "$WORKTREE_DIR/.claude/"
      echo "📄 Copied file: .claude/$file"
    fi
  done
else
  echo "⚠️  Warning: .claude directory not found in main directory"
fi

# direnv許可（.envrcが存在する場合）
if [ -f "$WORKTREE_DIR/.envrc" ]; then
  echo "🔐 Setting up direnv..."
  cd "$WORKTREE_DIR"
  if command -v direnv &> /dev/null; then
    direnv allow
    echo "✅ direnv allowed for worktree"
  else
    echo "⚠️  direnv not found. Please run 'direnv allow' manually in the worktree directory"
  fi
  cd - > /dev/null
fi

echo ""
echo "✅ Git worktree setup completed!"
echo "📍 Worktree location: $WORKTREE_DIR"
echo ""

# ポート情報を表示
cd "$WORKTREE_DIR"
CURRENTDIR=$(basename "$PWD")
if git rev-parse --git-dir >/dev/null 2>&1; then
  BRANCH_NAME_FOR_HASH=$(git rev-parse --abbrev-ref HEAD)
  NUM=$(( ($(echo "$BRANCH_NAME_FOR_HASH" | cksum | cut -d' ' -f1) % 1000) + 1 ))
  FRONTEND_PORT=$((5173 + NUM))
  BACKEND_PORT=$((8080 + NUM))
  POSTGRES_PORT=$((5432 + NUM))
  
  echo "🌐 Port allocation for this worktree:"
  echo "   Frontend: http://localhost:$FRONTEND_PORT"
  echo "   Backend:  http://localhost:$BACKEND_PORT"
  echo "   PostgreSQL: localhost:$POSTGRES_PORT"
fi

# worktreeディレクトリに移動して終了
echo ""
echo "📂 Moving to worktree directory..."
cd "$WORKTREE_DIR"
exec $SHELL