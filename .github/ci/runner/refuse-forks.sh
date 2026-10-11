#!/bin/bash
# The self-hosted runner's job-started hook: it refuses a job for a pull request whose head is not in this repository.
# A fork's pull request runs the fork's own copy of the workflow, which can drop any `if:` or name `runs-on: self-hosted`,
# so only the runner can keep fork code off the box. Install it on the runner, outside anything a job can change:
#   ACTIONS_RUNNER_HOOK_JOB_STARTED=/opt/runner/refuse-forks.sh     (in the runner's .env)
# A nonzero exit fails the job before its first step. .github/ci/runner/test_refuse_forks.sh tests it.
exec python3 - <<'PY'
import json, os, sys
e = json.load(open(os.environ['GITHUB_EVENT_PATH']))
pr = e.get('pull_request')
if pr is not None:
    head = (pr.get('head') or {}).get('repo') or {}
    if head.get('full_name') != os.environ.get('GITHUB_REPOSITORY'):
        sys.exit(f"refused: a pull request from {head.get('full_name') or 'a deleted fork'} does not run on this runner")
PY
