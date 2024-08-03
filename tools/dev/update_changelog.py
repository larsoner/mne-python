"""Compute code credit.

Run ./tools/dev/credit.sh first.

TODO:
- Remove orphan by linking somewhere
- Populate examples/ and tutorials/
"""
from pathlib import Path
from collections import defaultdict
import glob
import pathlib
import pprint
from textwrap import indent

import mne
import re

import numpy as np
import mne

repo_root = Path(__file__).parents[2]
doc_root = repo_root / "doc"
this_dir = Path(__file__).parent

ignore_removals = False

# make stats dict with filename keys
stats = defaultdict(lambda: defaultdict(lambda: np.zeros(2, int)))
ignores = {
    line.split("#")[0].strip(): list()
    for line in (repo_root / ".git-blame-ignore-revs").read_text().strip().splitlines()
    if not line.startswith("#")
}
# Get parent commit for black (use with --first-parent)
# ignores["16fe5b5dbb1622ddfb7269fab889bab44ef43212"] = ignores.pop(
#    "e81ec528a42ac687f3d961ed5cf8e25f236925b0"
# )
# MAINT: splitting utils.py to submodules (#5862)
ignores["203a96cbba2732d2e349a8f96065e74bbfd2a53b"] = list()
# API : refactor huge viz.py to have a proper module (still need to update tests)
ignores["5c1d64b1045f96c7c3a7e45bc4068e19fce695e7"] = list()
# MRG, DOC: Integrate manual parts #6767
ignores["ff349f356edb04e1b5f0db13deda8d1a20aca351"] = list()
# DOC : adding first version of converted mne manual WIP
ignores["5e744ee2cc38629f7708a35a27358eec2525c95d"] = list()

# Get "co-authored by" commits and start translating emails to actual names
mailmap = dict()
usernames = dict()
for line in (repo_root / ".mailmap").read_text("utf-8").splitlines():
    name = re.match("^([^<]+) <([^<>]+)>", line.strip()).group(1)
    assert name is not None
    emails = list(re.findall("<([^<>]+)>", line.strip()))
    assert len(emails) > 0
    new = emails[0]
    if new in usernames:
        assert usernames[new] == name
    else:
        usernames[new] = name
    if len(emails) == 1:
        continue
    for old in emails[1:]:
        if old in mailmap:
            assert new == mailmap[old]  # can be different names
        else:
            mailmap[old] = new
co_authorship = defaultdict(lambda: list())
for line in (this_dir / "co_author.txt").read_text("utf-8").splitlines():
    commit, email = line.strip().split("\t", maxsplit=1)
    email = email[1:-1]  # remove <>
    co_authorship[commit].append(mailmap.get(email, email))

commits = defaultdict(lambda: 0)
file_stats = (this_dir / "file_stats.txt").read_text("utf-8")
for line in file_stats.splitlines():
    commit, file, name, email, p, m = line.strip().split("\t")
    if email not in usernames:
        usernames[email] = name  # email.split("@")[0]

unknown_emails = set()
for line in file_stats.splitlines():
    commit, file, name, email, p, m = line.strip().split("\t")
    # TODO: Actually resolve these using gh
    if email not in usernames:
        usernames[email] = name  # email.split("@")[0]
    names = [name]
    if commit in co_authorship:
        emails = co_authorship.pop(commit)
        for email in emails:
            try:
                names.append(usernames[email])
            except KeyError:
                unknown_emails.add(email)
    if commit in ignores:
        ignores[commit].append([file, email, name, commit, p, m])
        continue
    if " => " in file:  # ignore moves
        continue
    # not in mne/
    if not file.startswith("mne/"):  # only mne/
        if file.startswith(("examples/", "tutorials/", "doc/")):
            if file.endswith(
                ("whats_new.rst", "latest.inc", "devel.rst", "changelog.rst")
            ):
                continue
            elif not file.endswith((".py", ".rst")):
                continue
        elif "/" not in file:  # root-level
            pass
        elif file.startswith(("tools/", ".circleci/", ".github/")):
            pass
        else:
            continue
    # in mne/
    elif file.split(".")[-1] != "py":
        if "html_templates" in file:
            pass  # allowed
        else:
            continue
    if p == "-" or m == "-":  # binary
        raise RuntimeError(f"Unexcluded binary file: {file}")
    p, m = int(p), int(m)
    if ignore_removals:
        m = 0
    for name in names:
        commits[(name, commit)] += p + m
        stats[file][name] += [p, m]
assert len(co_authorship) == 0
unknown_emails = set(
    email for email in unknown_emails
    if "autofix-ci[bot]" not in email
)
assert len(unknown_emails) == 0, "\n".join(sorted(unknown_emails))

print("Biggest included commits:")
commits = dict(
    (k, commits[k]) for k in sorted(commits, key=lambda k_: commits[k_], reverse=True)
)
for ni, name in enumerate(commits, 1):
    if ni > 10:
        break
    print(f"{name[1]} @ {commits[name]:5d} by {name[0]}")

print("\nIgnored commits:")
# Report the ignores
for commit in ignores:  # should have found one of each
    print(f"ignored {len(ignores[commit]):3d} files for hash {commit}")
    assert len(ignores[commit]) >= 1, (ignores[commit], commit)
globs = dict()
root_path = pathlib.Path(mne.__file__).parent
mod_file_map = dict()
for file in root_path.iterdir():
    rel = file.relative_to(root_path).with_suffix("")
    mod = f"mne.{rel}"
    if file.is_dir():
        globs[f"mne/{rel}/*.*"] = mod
        globs[f"mne/{rel}.*"] = mod
    elif file.is_file() and file.suffix == ".py":
        key = f"mne/{rel}.py"
        globs[key] = mod
        mod_file_map[mod] = key
# aliases for old stuff
globs["mne/artifacts/*.py"] = "mne.preprocessing"
for key in (
    "mne/info.py",
    "mne/fiff/*.*",
    "mne/_fiff/*.*",
    "mne/raw.py",
    "mne/testing.py",
    "mne/_hdf5.py",
    "mne/compensator.py",
):
    globs[key] = "mne.io"
for key in ("mne/transforms/*.py", "mne/_freesurfer.py"):
    globs[key] = "mne.transforms"
globs["mne/mixed_norm/*.py"] = "mne.inverse_sparse"
globs["mne/__main__.py"] = "mne.commands"
globs["mne/morph_map.py"] = "mne.surface"
globs["mne/baseline.py"] = "mne.epochs"
for key in (
    "mne/parallel.py",
    "mne/rank.py",
    "mne/misc.py",
    "mne/data/*.*",
    "mne/defaults.py",
    "mne/fixes.py",
    "mne/icons/*.*",
    "mne/icons.*",
):
    globs[key] = "mne.utils"
for key in ("mne/_ola.py", "mne/cuda.py"):
    globs[key] = "mne.filter"
for key in (
    "mne/*digitization/*.py",
    "mne/layouts/*.py",
    "mne/montages/*.py",
    "mne/selection.py",
):
    globs[key] = "mne.channels"
globs["mne/sparse_learning/*.py"] = "mne.inverse_sparse"
globs["mne/csp.py"] = "mne.preprocessing"
globs["mne/bem_surfaces.py"] = "mne.bem"
globs["mne/coreg/__init__.py"] = "mne.coreg"
globs["mne/inverse.py"] = "mne.minimum_norm"
globs["mne/stc.py"] = "mne.source_estimate"
globs["mne/surfer.py"] = "mne.viz"
globs["mne/tfr.py"] = "mne.time_frequency"
globs["mne/connectivity/*.py"] = "mne-connectivity (moved)"
globs["mne/realtime/*.py"] = "mne-realtime (moved)"
globs["mne/html_templates/*.*"] = "mne.report"
globs[".circleci/*"] = "maintenance"
globs["tools/*"] = "maintenance"
globs["doc/*"] = "doc"
globs["examples/*"] = "examples"
globs["tutorials/*"] = "tutorials"
for key in ("*.txt", "*.yml", ".*", "*.md", "setup.*", "MANIFEST.in", "Makefile"):
    globs[key] = "maintenance"
for key in ("README.rst", "flow_diagram.py", "*.toml"):
    globs[key] = "maintenance"
for key in ("mne/_version.py", "mne/externals/*.py", "*/__init__.py", "*/resources.py"):
    globs[key] = "null"
for key in ("mne/__init__.py", "AUTHORS.rst", "CITATION.cff", "CONTRIBUTING.rst"):
    globs[key] = "null"
for key in ("codemeta.json", "*conftest.py", "mne/tests/*.*", "jr-tools", "mne.qrc"):
    globs[key] = "null"
mod_stats = defaultdict(lambda: defaultdict(lambda: np.zeros(2, int)))
other_files = set()
total_lines = np.zeros(2, int)
for fname, counts in stats.items():
    for pattern, mod in globs.items():
        if glob.fnmatch.fnmatch(fname, pattern):
            break
    else:
        other_files.add(fname)
        mod = "other"
    for e, pm in counts.items():
        if mod == "mne._fiff":
            raise RuntimeError
        mod_stats[mod][e] += pm
        mod_stats["mne"][e] += pm
        total_lines += pm
mod_stats.pop("null")  # stuff we shouldn't give credit for
mod_stats = dict(
    (k, mod_stats[k])
    for k in sorted(
        mod_stats,
        key=lambda x: (
            not x.startswith("mne"),
            x == "maintenance",
            x.replace("-", "."),
        ),
    )
)  # sort modules alphabetically
other_files = sorted(other_files)
if len(other_files):
    pprint.pprint(other_files)
    raise RuntimeError(f"{len(other_files)} misc file(s) found")
print(f"\nTotal line change count: {list(total_lines)}")

BDG_KINDS = ["bdg-primary-line", "bdg-info-line"]
with open(doc_root / "credit.rst", "w", encoding="utf-8") as fid:
    fid.write(
        f"""
:orphan:

.. _contributors:

============
Contributors
============

There are many different ways to contribute to MNE-Python! So far we only list
code contributions below, but plan to add other metrics in the future.

.. _code_credit:

Code credit
===========

Below are lists of code contributors to MNE-Python. The numbers in parentheses are the
number of lines changed in our code history.

- :{BDG_KINDS[0]}:`This Color` is used for the top 10% of contributors.
- :{BDG_KINDS[1]}:`This Color` is used for the remaining 90% of contributors.

.. raw:: html

   <style>
   /* Make it occupy more page width */
   .bd-main .bd-content .bd-article-container {{
       max-width: 90vw;
   }}
   /* Limit max card height */
   div.sd-card-body {{
     max-height: 15em;
   }}
   </style>

Entire codebase
---------------

""")
    for mi, (mod, counts) in enumerate(mod_stats.items()):
        if mi == 0:
            assert mod == "mne", mod
        elif mi == 1:
            fid.write("""

By submodule
------------

Contributors often have domain-specific expertise, so we've broken down the
contributions by submodule as well below.

.. grid:: 1 2 3 3
   :gutter: 1

""")
        # if there are 10 this is 100, if there are 100 this is 100
        these_stats = dict((k, v.sum()) for k, v in counts.items())
        these_stats = dict(
            (k, these_stats[k])
            for k in sorted(these_stats, key=lambda x: these_stats[x], reverse=True)
        )
        kind = "blame" if mod in mod_file_map else "tree"
        link_mod = mod_file_map.get(mod, mod.replace(".", "/"))
        link = f"https://github.com/mne-tools/mne-python/{kind}/main/{link_mod}"
        # Use badges because they flow nicely, inside a grid to make it more compact
        stat_lines = []
        for ki, (k, v) in enumerate(these_stats.items()):
            idx = 0 if ki < (len(these_stats) - 1) // 10 + 1 else 1
            if "[bot]" in k:
                continue
            stat_lines.append(
                f":{BDG_KINDS[idx]}:`{k} ({format(v, ",")})`"
            )
        stat_lines = " ".join(stat_lines)
        if mi == 0:
            fid.write(
            f"""

.. card:: {mod}
   :class-card: overflow-auto
   :link: https://github.com/mne-tools/mne-python/graphs/contributors

   {stat_lines}

""")
        else:
            fid.write(
            f"""

   .. grid-item-card:: {mod}
      :class-card: overflow-auto
      :link: {link}

      {stat_lines}

""")
