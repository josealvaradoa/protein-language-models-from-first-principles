# Repository branch discipline

The parent workspace `AGENTS.md` still governs this repository. This file adds branch discipline only.

- Read-only inspection is allowed on `main`. Before any project write, commit, or push, check the current branch.
- Make implementation, documentation, configuration, generated evidence, and every other project mutation on a scoped feature or week branch. Never make these changes on `main` or `master`.
- If a write is requested while on `main`, create or switch to a clearly scoped branch before editing. Stop and ask for direction if the intended branch is ambiguous.
- Changes enter `main` only through a GitHub pull request. Never push directly to `main`.
- Never force-push or delete `main`.
- Preserve experiment provenance. Do not squash-merge, rebase, amend, or force-push commits whose SHAs are referenced by datasets, manifests, model or evaluation records, reports, or other evidence. Evidence-bearing pull requests use normal merge commits.
- Generated production evidence remains Jose-operated under the parent workspace rules.
