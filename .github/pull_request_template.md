## What and why


CI runs on every push, only the jobs this PR's files need, and pr-labels.yml labels the PR by area. A maintainer's
`/ready` marks it ready for the merge queue (`/unready` to undo); a new push clears it.
