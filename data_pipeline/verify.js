import {
  createTerm, createFraction,
  differentiateFraction, integrateFraction,
  definiteIntegrateFraction, evaluateFraction
} from '../../SLaNg/slang-basic.js';
import {
  productRuleDifferentiate, quotientRuleDifferentiate,
  gradient
} from '../../SLaNg/slang-advanced.js';

const ORACLE = {
  diff: inp => differentiateFraction(inp.expr, inp.var),
  integrate: inp => integrateFraction(inp.expr, inp.var),
  def_integrate: inp => definiteIntegrateFraction(inp.expr, inp.lower, inp.upper, inp.var),
  gradient: inp => gradient(inp.expr, inp.vars),
  product_rule: inp => productRuleDifferentiate([inp.u, inp.v], inp.var),
  quotient_rule: inp => quotientRuleDifferentiate(inp.u, inp.v, inp.var),
};

function testPoints(variables, n = 50) {
  return Array.from({ length: n }, () => {
    const pt = {};
    for (const v of variables) pt[v] = (Math.random() * 9.6 - 4.8) + 0.01;
    return pt;
  });
}

function compare(a, b, variables, tol = 1e-9) {
  const pts = testPoints(variables);
  let agreed = 0, tested = 0;
  for (const pt of pts) {
    try {
      const va = evaluateFraction(a, pt);
      const vb = evaluateFraction(b, pt);
      if (!isFinite(va) || !isFinite(vb)) continue;
      tested++;
      if (Math.abs(va - vb) <= tol) agreed++;
    } catch { continue; }
  }
  return { match: agreed === tested && tested > 0, confidence: tested > 0 ? agreed / tested : 0 };
}

function verify(inputEnv, modelOutput) {
  const fn = ORACLE[inputEnv.op];
  if (!fn) return { verified: false, confidence: 0 };
  let oracle;
  try { oracle = fn(inputEnv); }
  catch (e) { return { verified: false, confidence: 0, error: e.message }; }
  
  const variables = inputEnv.vars || (inputEnv.var ? [inputEnv.var] : ['x']);
  
  if (inputEnv.op === 'gradient') {
    for (const v of variables) {
      const r = compare(oracle[v], modelOutput[v], variables);
      if (!r.match) return { verified: false, confidence: r.confidence };
    }
    return { verified: true, confidence: 1.0 };
  }
  
  const r = compare(oracle, modelOutput, variables);
  return { verified: r.match, confidence: parseFloat(r.confidence.toFixed(4)) };
}

const iIdx = process.argv.indexOf('--input');
const oIdx = process.argv.indexOf('--output');

if (iIdx !== -1 && oIdx !== -1) {
  try {
    const result = verify(
      JSON.parse(process.argv[iIdx + 1]),
      JSON.parse(process.argv[oIdx + 1])
    );
    console.log(JSON.stringify(result));
    process.exit(result.verified ? 0 : 1);
  } catch (e) {
    console.log(JSON.stringify({ verified: false, confidence: 0, error: e.message }));
    process.exit(2);
  }
}

export { verify };