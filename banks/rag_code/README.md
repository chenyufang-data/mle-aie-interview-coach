# rag_code - coding problems with rubrics and hint ladders

The sixth bank (roadmap step 7, phase 2): the problems for coding practice
and the mock coding round. It is not a question-practice corpus - it is not
in `coach/config.py` `CORPUS_PATHS`, so question retrieval never sees it.

Seeded from the author's practice notebooks: the weekly DSA homework and
the additional DSA notebooks (LeetCode problems), and their own ML-coding
and PyTorch exercises.

What a record holds, by LeetCode's terms (no copying or republishing its
content, no scraping - docs/plan.md step 7):
- **LeetCode problems**: number, title and the link
  `https://leetcode.com/problems/<slug>/` - nothing else from LeetCode. The
  user opens the problem there in a separate window. Some notebook cells
  hold LeetCode's statement pasted in; from those the builder reads only
  the number, the title line and the link, and the pasted text is never
  stored and never sent to a model (it is held in memory only to check that
  nothing generated overlaps it).
- **The author's own exercises** (`ML NN`, `PT NN`): the full statement and
  any starter code, since the text is theirs; answer keys are left out.
- **Written by us, per problem**: approach labels (e.g. "hash map"), role
  (`shared` for general DSA, `mle` for classic ML and data work, `aie` for
  LLM and retrieval work), difficulty, an estimated complexity, a rubric
  (key points, edge cases, common mistakes, code quality, what to say out
  loud), follow-up questions, and a four-level hint ladder: a clarifying
  question, a concept nudge, the approach in words, then pseudocode for one
  step - never a full solution.
- Not included yet: the SQL drills, which paraphrase LeetCode's SQL
  problems with the same tables and columns.

Records keep the bank chunk shape (`id`, `interview`, `metadata`) so
`tools/review_bank.py rag_code` reviews them; the coding fields sit under
`code`.

Build / update it locally (the notes folder is the author's, outside the
repository):

    .venv\Scripts\python ingest\ingest_code.py --source "<notes folder>" --list
    .venv\Scripts\python ingest\ingest_code.py --source "<notes folder>" --generate            # dry run
    .venv\Scripts\python ingest\ingest_code.py --source "<notes folder>" --generate --confirm  # writes missing records
    .venv\Scripts\python ingest\ingest_code.py --source "<notes folder>" --check               # leak and hint checks

Generation runs on the signed-in Claude Code or Codex subscription
(`coach/cli_engine.py`, the local default), with the model and effort from
`.env`. Re-runs are idempotent - existing ids are skipped. Review the
records with `tools/review_bank.py rag_code`, then `--apply` the decisions.
The bank file is gitignored and backed up to the private repository by
`tools/backup_private.py`.
