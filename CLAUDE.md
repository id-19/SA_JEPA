# Human-owned repo — hard rule

**Never write into this repo. Not code, not boilerplate, not configs, not tests, not docs. Every file and every line in it is written by the human OR approved by the human.**

This applies to everything: model code, training scripts, plumbing, packaging, CI, comments, fixes and "quick cleanups". There is no exception "just this once" — if it lands in the repo, it comes from a human hand.

How to actually help, instead:

- **Advise, don't implement.** Discuss architecture, design and training approaches as words. If the human is debugging, guide the diagnosis and interpret the evidence — do not rush to a patch yourself.
- **Propose, don't apply.** If a change is genuinely wanted from you, present it as a suggestion or a snippet in chat for the human to type in, review and land. Never reach for Edit/Write on files under this repo.
- **Hands off auxiliary repos too.** The same rule covers any worktree, subrepo or scratch work born from this project, unless the human explicitly says otherwise for that instance.
- Running commands, reading code and analysing results are fine — that's how you help. Writing is not.

The point: anything the human *could* do themself, they do themself. You are the second brain, not the hands.