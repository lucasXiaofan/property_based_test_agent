#!/bin/bash

# Usage:
# ./extract_pandas_docs.sh 2.3.0 DataFrame.reindex

VERSION_RAW=$1
FUNC=$2

# Drop leading "v" if present
VERSION=${VERSION_RAW#v}
MAJOR_MINOR=$(echo "$VERSION" | cut -d. -f1,2)

BASE_DIR="/Users/xiaofanlu/Documents/github_repos/property_based_test_agent/pandas_bug_finding/analysis/docs/"
SAFE_NAME=${FUNC//./_}
OUT_FILE="$BASE_DIR/${SAFE_NAME}_${VERSION}.md"

# Potential doc URLs
URL1="https://pandas.pydata.org/pandas-docs/version/${VERSION}/reference/api/pandas.${FUNC}.html"
URL2="https://pandas.pydata.org/docs/${MAJOR_MINOR}/reference/api/pandas.${FUNC}.html"

# Clear old file
mkdir -p "$BASE_DIR"
rm -f "$OUT_FILE"

echo "🌐 Trying to fetch docs for $FUNC ($VERSION)"
echo "Output ➤ $OUT_FILE"
echo "" > "$OUT_FILE"

# Try first URL
echo "Checking $URL1..."
HTTP_STATUS=$(curl -s -o /tmp/doctmp.html -w "%{http_code}" "$URL1")

if [ "$HTTP_STATUS" -eq 200 ]; then
    DOC_URL="$URL1"
    echo "✔ Found docs at $URL1"
else
    echo "❌ No docs at $URL1 (HTTP $HTTP_STATUS), trying fallback URL..."
    echo "Checking $URL2..."
    HTTP_STATUS2=$(curl -s -o /tmp/doctmp.html -w "%{http_code}" "$URL2")
    if [ "$HTTP_STATUS2" -eq 200 ]; then
        DOC_URL="$URL2"
        echo "✔ Found docs at $URL2"
    else
        echo "❌ Could not find docs at either URL"
        exit 1
    fi
fi

# Convert HTML to Markdown
echo "📥 Fetching HTML from $DOC_URL"
curl -s "$DOC_URL" -o /tmp/doctmp.html

echo "📄 Converting to Markdown"
pandoc /tmp/doctmp.html -f html -t markdown -o "$OUT_FILE"

# Prepend header
sed -i '' "1i\\
# Documentation for $FUNC ($VERSION)\\
Source URL: $DOC_URL\\
\\
" "$OUT_FILE"

echo "✅ Done! Open $OUT_FILE to view."