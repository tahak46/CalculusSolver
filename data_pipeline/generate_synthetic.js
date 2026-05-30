import fs from "fs";
import path from "path";
import {
  createTerm,
  createFraction,
  differentiateFraction,
  integrateFraction,
  evaluateFraction,
} from "../../SLaNg/slang-basic.js";
import {
  productRuleDifferentiate,
  quotientRuleDifferentiate,
  gradient,
} from "../../SLaNg/slang-advanced.js";

const DEFAULT_OPS = [
  "diff",
  "integrate",
  "product_rule",
  "quotient_rule",
  "gradient",
];
const COEFF_CHOICES = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5];

function randInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function randChoice(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

function parseCsv(arg, defaultValue = []) {
  if (!arg || arg.trim() === "") return defaultValue;
  return arg
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function parseFlags() {
  const args = process.argv.slice(2);
  const getArg = (name, fallback) => {
    const idx = args.indexOf(name);
    return idx !== -1 ? args[idx + 1] : fallback;
  };

  const count = parseInt(getArg("--count", "5000000"), 10);
  const opsArg = getArg("--ops", "all");
  const maxDepth = parseInt(getArg("--max-depth", "5"), 10);
  const vars = parseCsv(getArg("--vars", "x,y"));
  const outDir = getArg("--out", "data/synthetic/");

  const ops =
    opsArg === "all" || opsArg.trim() === "" ? DEFAULT_OPS : parseCsv(opsArg);

  const invalidOps = ops.filter((op) => !DEFAULT_OPS.includes(op));
  if (invalidOps.length > 0) {
    throw new Error(
      `Unsupported op types: ${invalidOps.join(", ")}. Supported ops: ${DEFAULT_OPS.join(", ")}`,
    );
  }

  return {
    count: Number.isFinite(count) && count > 0 ? count : 5000000,
    ops,
    maxDepth: Number.isFinite(maxDepth) && maxDepth > 0 ? maxDepth : 5,
    vars: vars.length > 0 ? vars : ["x", "y"],
    outDir: path.resolve(outDir),
  };
}

function randomTerm(vars, maxExp = 4) {
  const coeff = randChoice(COEFF_CHOICES);
  const term = { coeff };

  if (Math.random() < 0.5 || vars.length === 0) {
    return term;
  }

  const varCount = randInt(1, Math.min(vars.length, 2));
  const chosen = [...vars].sort(() => Math.random() - 0.5).slice(0, varCount);
  term.var = {};
  for (const v of chosen) {
    term.var[v] = randInt(1, maxExp);
  }
  return term;
}

function randomPolynomial(vars, termCount, maxExp) {
  const terms = [];
  const seen = new Set();

  for (let i = 0; i < termCount; i += 1) {
    let term = randomTerm(vars, maxExp);
    const key = JSON.stringify(term.var || {});
    if (seen.has(key) && term.coeff !== 0) {
      term = { ...term, coeff: randChoice(COEFF_CHOICES) };
    }
    seen.add(key);
    terms.push(term);
  }

  if (terms.length === 0) {
    terms.push({ coeff: randChoice(COEFF_CHOICES) });
  }

  return { terms };
}

function randomFraction(vars, depth) {
  const maxExp = Math.min(5, Math.max(1, depth + 1));
  const numerator = randomPolynomial(
    vars,
    randInt(1, Math.min(4, depth + 1)),
    maxExp,
  );

  if (depth <= 1 || Math.random() < 0.6) {
    return createFraction(numerator.terms, [createTerm(1)]);
  }

  const denominator = randomPolynomial(
    vars,
    Math.max(1, randInt(1, Math.min(3, depth))),
    maxExp,
  );
  return createFraction(numerator.terms, denominator.terms);
}

function randomMultivarFraction(vars, depth) {
  if (vars.length === 0) vars = ["x", "y"];
  return randomFraction(vars, depth);
}

function numericallyEqual(exprA, exprB, variables, tol = 1e-9, points = 50) {
  for (let i = 0; i < points; i += 1) {
    const pt = {};
    for (const v of variables) {
      pt[v] = Math.random() * 9.6 - 4.8 + 0.01;
    }
    try {
      const a = evaluateFraction(exprA, pt);
      const b = evaluateFraction(exprB, pt);
      if (!isFinite(a) || !isFinite(b)) continue;
      if (Math.abs(a - b) > tol) return false;
    } catch {
      continue;
    }
  }
  return true;
}

function genDiff(vars, maxDepth) {
  const variable = vars[0] || "x";
  const expr = randomFraction([variable], maxDepth);
  const output = differentiateFraction(expr, variable);
  return {
    input: { op: "diff", var: variable, expr },
    output: { expr: output, status: "solved", steps: [] },
    verify: () =>
      numericallyEqual(output, differentiateFraction(expr, variable), [
        variable,
      ]),
  };
}

function genIntegrate(vars, maxDepth) {
  const variable = vars[0] || "x";
  const expr = randomFraction([variable], maxDepth);
  const output = integrateFraction(expr, variable);
  return {
    input: { op: "integrate", var: variable, expr },
    output: { expr: output, status: "solved", steps: [] },
    verify: () =>
      numericallyEqual(output, integrateFraction(expr, variable), [variable]),
  };
}

function genProductRule(vars, maxDepth) {
  const variable = vars[0] || "x";
  const u = randomFraction([variable], Math.max(1, Math.floor(maxDepth / 2)));
  const v = randomFraction([variable], Math.max(1, Math.floor(maxDepth / 2)));
  const output = productRuleDifferentiate([u, v], variable);
  return {
    input: { op: "product_rule", var: variable, u, v },
    output: { expr: output, status: "solved", steps: [] },
    verify: () =>
      numericallyEqual(output, productRuleDifferentiate([u, v], variable), [
        variable,
      ]),
  };
}

function genQuotientRule(vars, maxDepth) {
  const variable = vars[0] || "x";
  const u = randomFraction([variable], Math.max(1, Math.floor(maxDepth / 2)));
  const v = randomFraction([variable], Math.max(1, Math.floor(maxDepth / 2)));
  const output = quotientRuleDifferentiate(u, v, variable);
  return {
    input: { op: "quotient_rule", var: variable, u, v },
    output: { expr: output, status: "solved", steps: [] },
    verify: () =>
      numericallyEqual(output, quotientRuleDifferentiate(u, v, variable), [
        variable,
      ]),
  };
}

function genGradient(vars, maxDepth) {
  const expressionVars = vars.length > 0 ? vars : ["x", "y"];
  const expr = randomMultivarFraction(expressionVars, maxDepth);
  const output = gradient(expr, expressionVars);
  return {
    input: { op: "gradient", vars: expressionVars, expr },
    output: { expr: output, status: "solved", steps: [] },
    verify: () => {
      const oracle = gradient(expr, expressionVars);
      return Object.keys(oracle).every((v) =>
        numericallyEqual(oracle[v], output[v], expressionVars),
      );
    },
  };
}

function buildGenerators(ops, vars, maxDepth) {
  return ops.map((op) => {
    switch (op) {
      case "diff":
        return () => genDiff(vars, maxDepth);
      case "integrate":
        return () => genIntegrate(vars, maxDepth);
      case "product_rule":
        return () => genProductRule(vars, maxDepth);
      case "quotient_rule":
        return () => genQuotientRule(vars, maxDepth);
      case "gradient":
        return () => genGradient(vars, maxDepth);
      default:
        throw new Error(`Unsupported generator op: ${op}`);
    }
  });
}

async function run() {
  const { count, ops, maxDepth, vars, outDir } = parseFlags();
  const generators = buildGenerators(ops, vars, maxDepth);

  fs.mkdirSync(outDir, { recursive: true });
  const outPath = path.join(outDir, "synthetic.jsonl");
  const stream = fs.createWriteStream(outPath, { flags: "w" });

  let attempts = 0;
  let saved = 0;
  let skipped = 0;

  while (attempts < count) {
    attempts += 1;
    let pair;
    try {
      const generator =
        generators[Math.floor(Math.random() * generators.length)];
      pair = generator();
    } catch (error) {
      skipped += 1;
      continue;
    }

    let ok = false;
    try {
      ok = pair.verify();
    } catch {
      ok = false;
    }

    if (!ok) {
      skipped += 1;
      continue;
    }

    stream.write(
      JSON.stringify({ input: pair.input, output: pair.output }) + "\n",
    );
    saved += 1;

    if (attempts % 100000 === 0) {
      process.stdout.write(
        `Attempted ${attempts}/${count}, saved ${saved}, rejected ${skipped}\r`,
      );
    }
  }

  stream.end();
  console.log(
    `\nDone. Attempts: ${attempts}, saved: ${saved}, rejected: ${skipped}`,
  );
  console.log(`Output: ${outPath}`);
}

if (process.argv[1].endsWith("generate_synthetic.js")) {
  run().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}
