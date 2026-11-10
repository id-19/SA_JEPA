# Human-owned repo (with boilerplate exception)

Core learning and model code is written by the human. Do not write or
modify algorithm/model/training code, even framed as "just wiring" or
"boilerplate," unless the human explicitly asks for that exact change.

Mechanical scaffolding may be written only when the human explicitly
authorizes it (e.g., eval adapters, IO wrappers, config plumbing).
Before doing so, confirm it's something the human could type and do
themselves — you are saving them time, not removing their learning.
Confirm they understand the effect and approve it.

The same rule applies to any worktree, subrepo, or scratch work derived
from this project unless the human explicitly relaxes it for that instance.

Helping, without writing: advising, proposing snippets, reading code,
running commands, and analysing results. Anything the human could type
themselves, the human types.

# Existing guardrails
- Never `git add -A` / `git commit -a`; stage by explicit path.
- Local commits fine; push/PR/ticket each need explicit approval.
- Never force-push or rewrite shared history without confirmation.