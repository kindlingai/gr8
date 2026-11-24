## What and why


Pull request titles follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `perf:`,
`docs:`, `ci:`, `chore:`, etc., with optional scope such as `feat(qw38): ...`), checked by `pr-title.yml` for
automated `release-please` releases.

CI runs on every push, only the jobs this PR's files need, and pr-labels.yml labels the PR by area. A maintainer's
`/ready` marks it ready for the merge queue (`/unready` to undo); a new push clears it.
