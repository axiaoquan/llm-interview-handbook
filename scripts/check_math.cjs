// Optional TeX syntax validation. Requires katex on the Node module search path.
// KaTeX parsing is NOT proof of GitHub/MathJax visual rendering.
const path = require('node:path');
const { spawnSync } = require('node:child_process');
let katex;
try {
  katex = require('katex');
} catch {
  console.error('Install katex in an isolated directory, then set NODE_PATH to its node_modules.');
  process.exit(1);
}
const root = path.resolve(__dirname, '..');
const result = spawnSync(process.env.PYTHON || 'python3', ['scripts/check_docs.py', '--math-json'], {
  cwd: root, encoding: 'utf8', maxBuffer: 16 * 1024 * 1024,
});
if (result.error || result.status !== 0) {
  console.error(result.error || result.stderr);
  process.exit(1);
}
const formulas = JSON.parse(result.stdout);
let errors = 0;
for (const item of formulas) {
  try {
    katex.renderToString(item.formula, { throwOnError: true, strict: 'ignore', trust: false });
  } catch (error) {
    console.error(`${item.file}:${item.line}: ${error.message}`);
    errors++;
  }
}
console.log(`KaTeX ${katex.version}: ${formulas.length} formulas, ${errors} syntax errors.`);
process.exitCode = errors ? 1 : 0;
