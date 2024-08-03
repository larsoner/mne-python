#!/bin/bash

set -eo pipefail
THIS_DIR=$(dirname "$0")
FILE_STATS_FILE="$THIS_DIR/file_stats.txt"
CO_AUTHORS_FILE="$THIS_DIR/co_author.txt"
# One idea is to use the local git log:
# https://git-scm.com/docs/pretty-formats
# first line has email then hash
git log --no-merges --numstat --pretty="%aE	%aN	%H	x" | awk -F "	" '{ if (NF == 4){ email = $1; name = $2; hash = $3 }; if(NF == 3) {file=$3; print hash"	"file"	"name"	"email"	"$1"	"$2}}' > "$FILE_STATS_FILE"
git log --no-merges --format="%H %n%(trailers:key=Co-authored-by)" | awk '{ if (NF == 1){ commit = $1; }; if(NF > 1) { email=$(NF); print commit"	"email }}' > "$CO_AUTHORS_FILE"

# But this becomes very tricky with squash+merge vs old merge style. So instead,
# let's query GH directly for the PR numbers and merge commits, then use
# "git show" on those.
#ARGS="-R mne-tools/mne-python"
#PR_NUMBERS_FILE="$THIS_DIR/pr_numbers.json"
#if [ ! -f "$PR_NUMBERS_FILE" ]; then
#    echo "Listing PRs (this can take a while) ..."
#    gh pr list $ARGS --limit 10000 --state merged --json number,title,mergeCommit > "$PR_NUMBERS_FILE"
#fi
#rm -f $FILE_STATS_FILE
#for COMMIT in $(cat "$PR_NUMBERS_FILE" | jq -r '.[].mergeCommit.[]'); do
#    echo "$COMMIT"
#    # This will be way too slow:
#    # gh pr diff $ARGS $PR_NUM
#    # So instead use local information:
#    git show $COMMIT --numstat --pretty="%aE	%aN	%H	x" | awk -F "	" '{ if (NF == 4){ email = $1; name = $2; hash = $3 }; if(NF == 3) {file=$3; print hash"	"file"	"name"	"email"	"$1"	"$2}}' >> $FILE_STATS_FILE
#    git show $COMMIT --format="%H %n%(trailers:key=Co-authored-by)" | awk '{ if (NF == 1){ commit = $1; }; if(NF > 1) { email=$(NF); print commit"	"email }}' >> $CO_AUTHORS_FILE
#done
