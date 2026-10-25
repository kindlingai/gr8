// Which files a front-end change may touch. pr-category.yml refuses "Front end only" on a PR that touches anything else,
// and ci.yml skips the engine builds and the engine CPU tests for a PR that touches nothing else.
// The front end, the launcher, the smoketest, ./gr8 and Markdown. Not frontend/g6front-rs/Cargo.lock: it pins the
// XGrammar sources the engine compiles too (scripts/build.sh, the Dockerfile's xgrammar-src stage).
const FRONTEND = [/^frontend\//, /^tools\/launch\//, /^smoketest\//, /^gr8$/, /\.md$/];
const ENGINE = [/^frontend\/g6front-rs\/Cargo\.lock$/];

/** The paths of `files` (pulls.listFiles entries) outside the front end: a rename counts by both of its names. */
function outsideFrontend(files) {
  const paths = files.flatMap(f => [f.filename, f.previous_filename].filter(Boolean));
  return [...new Set(paths)].filter(p => ENGINE.some(r => r.test(p)) || !FRONTEND.some(r => r.test(p)));
}

/** Every file of the PR, or null when GitHub lists only part of them (3,000 at most). */
async function prFiles(github, context, pr) {
  const files = await github.paginate(github.rest.pulls.listFiles,
    { ...context.repo, pull_number: pr.number, per_page: 100 });
  return files.length < pr.changed_files ? null : files;
}

module.exports = { outsideFrontend, prFiles };
