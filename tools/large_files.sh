#!/usr/bin/env bash

# Usage:
#   ./find_large_files.sh [directory]
# If no directory is supplied, defaults to current directory.

set -euo pipefail

TARGET_DIR="${1:-.}"
LINE_LIMIT=1000

echo "Scanning directory: $TARGET_DIR"
echo "Reporting files over $LINE_LIMIT lines"
echo

# Use find with null separation for safety
find "$TARGET_DIR" -type f \( \
	-name "*.py" -o \
	-name "*.js" -o \
	-name "*.html" -o \
	-name "*.css" \
\) -print0 |
while IFS= read -r -d '' file; do
	line_count=$(wc -l < "$file")
	if [ "$line_count" -gt "$LINE_LIMIT" ]; then
		printf "%8d  %s\n" "$line_count" "$file"
	fi
done | sort -nr
