#!/bin/bash

set -eo pipefail
# https://git-scm.com/docs/pretty-formats
# first line has email then hash
git log --no-merges --numstat --pretty="%aE	%aN	%H	x" | awk -F "	" '{ if (NF == 4){ email = $1; name = $2; hash = $3 }; if(NF == 3) {file=$3; print hash"	"file"	"name"	"email"	"$1"	"$2}}' > file_stats.txt
git log --no-merges --format="%H %n%(trailers:key=Co-authored-by)" | awk '{ if (NF == 1){ commit = $1; }; if(NF > 1) { email=$(NF); print commit"	"email }}' > co_author.txt
