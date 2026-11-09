// Parse TeX snippets with MathJax 3 (what GitHub uses in the browser) and report errors.
// Input (stdin): JSON [{id, tex, display}]. Output (stdout): JSON [{id, error}] for failures.
import {mathjax} from 'mathjax-full/js/mathjax.js';
import {TeX} from 'mathjax-full/js/input/tex.js';
import {SVG} from 'mathjax-full/js/output/svg.js';
import {liteAdaptor} from 'mathjax-full/js/adaptors/liteAdaptor.js';
import {RegisterHTMLHandler} from 'mathjax-full/js/handlers/html.js';
import {AllPackages} from 'mathjax-full/js/input/tex/AllPackages.js';

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const tex = new TeX({
  // drop the packages that hide mistakes (unknown macros drawn in red, errors swallowed)
  packages: AllPackages.filter((p) => p !== 'noundefined' && p !== 'noerrors'),
  formatError: (_jax, err) => { throw err; },
});
const doc = mathjax.document('', {InputJax: tex, OutputJax: new SVG({fontCache: 'none'})});

let input = '';
for await (const chunk of process.stdin) input += chunk;

const failures = [];
for (const {id, tex: src, display} of JSON.parse(input)) {
  try {
    doc.convert(src, {display});
  } catch (e) {
    failures.push({id, error: String(e.message || e)});
  }
}
process.stdout.write(JSON.stringify(failures));
