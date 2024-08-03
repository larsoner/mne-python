"""TODO: Maybe remove. Not currently used because it's too slow."""

import os
import re
from tqdm import tqdm
from pathlib import Path
from github import Github
from github import Auth

ignores = [
    1420,  # Split huge viz.py to a proper module
    5862,  # MAINT: splitting utils.py to submodules
    6767,  # MRG, DOC: Integrate manual parts
    11667,  # MAINT: Run black on codebase
    11868,  # MAINT: Fix YAML spacing
    12097,  # Sort imports
    12261,  # MRG: Use ruff-format instead of Black
    12533,  # MAINT: Clean up PyVista contexts
    12588,  # [pre-commit.ci] pre-commit autoupdate
    12603,  # STY: Apply ruff/pyupgrade rule UP028
]

auth = Auth.Token(os.getenv('GITHUB_TOKEN'))
g = Github(auth=auth, per_page=100)
fname = Path(__file__).parent / "file_stats.txt"

# gh pr list --repo mne-tools/mne-python --state merged
r = g.get_repo('mne-tools/mne-python')
co_re = re.compile("Co-authored-by: ([^<>]+) <([^()]+)>")
progress = total = None
for pull in r.get_pulls('closed'):
    if progress is None:
        total = pull.number
        progress = tqdm(total, desc="PR")
    progress.update(total - pull.number)
    # TODO: Should add:
    # pull.get_comments()
    # pull.get_review_comments()

    # PR diff credit
    if pull.number in ignores:
        continue
    if not pull.merged:
        continue
    # One option is to do a git diff between pull.base and pull.head,
    # but let's see if we can stay pythonic
    merge_commit = r.get_commit(pull.merge_commit_sha)
    authors_emails = [(pull.user.name, pull.user.email)]
    msg = merge_commit.commit.message.replace("\r", "")
    authors_emails.extend(co_re.findall(msg))
    # In theory we could go through the files in Python, but it take forever to load
    # so make use of the fact that our local history has the same information.
    cmd = [
        "git",
        "show",
        merge_commit.commit.sha, --numstat --pretty="%aE%aN%Hx" | awk -F "" '{ if (NF == 4){ email = $1; name = $2; hash = $3 }; if(NF == 3) {file=$3; print hash""file""name""email""$1""$2}}'
    ]
    print(merge_commit.commit.sha)
    raise RuntimeError
    for file in pull.get_files():
        for author, email in authors_emails:
            print(f"{pull.merge_commit_sha}\t{file.filename}\t{author}\t{email}\t{file.additions}\t{file.deletions}")
tqdm.update(total)

# g.close()
