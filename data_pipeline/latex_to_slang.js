import fs from "fs";
import path from "path";
import { latexToSlang } from "../slang/src/convertor.js";
import { verify } from "./verify.js";

const DEFAULT_FIELDS = {
  input: ["input", "question", "problem", "expr", "latex", "prompt"],
  output: ["output", "answer", "solution", "target", "result"],
};

function parseArgs() {
  const args = process.argv.slice(2);
  const getArg = (name, fallback) => {
    const idx = args.indexOf(name);
    return idx !== -1 ? args[idx + 1] : fallback;
  };

  return {
    inputFile: getArg("--input"),
    outDir: path.resolve(getArg("--out", "data/slang_pairs/")),
    verify: getArg("--verify", "false").toLowerCase() === "true",
  };
}

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

function getField(obj, keys) {
  for (const key of keys) {
    if (obj[key] !== undefined && obj[key] !== null) {
      return obj[key];
    }
  }
  return null;
}

function normalizeLatex(value) {
  if (typeof value !== "string") return null;
  return value.trim().replace(/\s+/g, " ");
}

function parseProblemLatex(latex) {
  latex = normalizeLatex(latex);
  if (!latex) return null;

  const diffMatch = latex.match(
    /^(?:\\frac\s*\{\s*d\s*\}\s*\{\s*d([a-zA-Z])\s*\}|d\/d([a-zA-Z]))\s*\(?\s*([^\)]+?)\s*\)?$/,
  );
  if (diffMatch) {
    const variable = diffMatch[1] || diffMatch[2];
    const exprLatex = diffMatch[3];
    return {
      op: "diff",
      var: variable,
      expr: latexToSlang(exprLatex, { strictMode: true }),
    };
  }

  const integralMatch = latex.match(/\\int\s*([^dx]+?)\s*d([a-zA-Z])/);
  if (integralMatch) {
    const exprLatex = integralMatch[1];
    const variable = integralMatch[2];
    return {
      op: "integrate",
      var: variable,
      expr: latexToSlang(exprLatex, { strictMode: true }),
    };
  }

  const gradientMatch = latex.match(/\\nabla\s*([a-zA-Z]+)/);
  if (gradientMatch) {
    const variable = gradientMatch[1];
    return {
      op: "gradient",
      vars: [variable],
      expr: latexToSlang(latex.replace(gradientMatch[0], "").trim(), {
        strictMode: true,
      }),
    };
  }

  return null;
}

function formatRejection(entry, reason) {
  return JSON.stringify({ ...entry, reason }, null, 0);
}

function makeOutputPath(outDir, inputFile) {
  const basename = path.basename(inputFile).replace(/\.jsonl$/i, "");
  return path.join(outDir, `${basename}.jsonl`);
}

function loadJsonl(filePath) {
  return fs
    .readFileSync(filePath, "utf8")
    .split(/\r?\n/)
    .filter((line) => line.trim());
}

function writeJsonlLine(stream, object) {
  stream.write(JSON.stringify(object) + "\n");
}

function run() {
  const { inputFile, outDir, verify: verifyFlag } = parseArgs();
  if (!inputFile) {
    throw new Error("Missing --input argument.");
  }

  ensureDir(outDir);
  const outPath = makeOutputPath(outDir, inputFile);
  const rejectLogPath = path.join(outDir, "rejection_log.jsonl");

  const lines = loadJsonl(inputFile);
  const outStream = fs.createWriteStream(outPath, { flags: "w" });
  const rejectStream = fs.createWriteStream(rejectLogPath, { flags: "w" });

  let kept = 0;
  let rejected = 0;

  for (const line of lines) {
    let item;
    try {
      item = JSON.parse(line);
    } catch (error) {
      rejectStream.write(formatRejection({ raw: line }, "invalid_json") + "\n");
      rejected += 1;
      continue;
    }

    const inputLatex = normalizeLatex(getField(item, DEFAULT_FIELDS.input));
    const outputLatex = normalizeLatex(getField(item, DEFAULT_FIELDS.output));

    if (!inputLatex || !outputLatex) {
      rejectStream.write(
        formatRejection(item, "missing_input_or_output") + "\n",
      );
      rejected += 1;
      continue;
    }

    let parsedInput;
    try {
      parsedInput = parseProblemLatex(inputLatex);
    } catch (error) {
      rejectStream.write(
        formatRejection({ item, error: error.message }, "parse_input_failed") +
          "\n",
      );
      rejected += 1;
      continue;
    }

    if (!parsedInput) {
      rejectStream.write(
        formatRejection({ item, reason: "unsupported_problem_structure" }) +
          "\n",
      );
      rejected += 1;
      continue;
    }

    let parsedOutput;
    try {
      parsedOutput = latexToSlang(outputLatex, { strictMode: true });
    } catch (error) {
      rejectStream.write(
        formatRejection({ item, error: error.message }, "parse_output_failed") +
          "\n",
      );
      rejected += 1;
      continue;
    }

    const pair = {
      input: parsedInput,
      output: { status: "solved", expr: parsedOutput, steps: [] },
    };

    if (verifyFlag) {
      try {
        const result = verify(pair.input, pair.output.expr);
        if (!result.verified) {
          rejectStream.write(
            formatRejection({ item, verify: result }, "verification_failed") +
              "\n",
          );
          rejected += 1;
          continue;
        }
      } catch (error) {
        rejectStream.write(
          formatRejection(
            { item, error: error.message },
            "verification_error",
          ) + "\n",
        );
        rejected += 1;
        continue;
      }
    }

    writeJsonlLine(outStream, pair);
    kept += 1;
  }

  outStream.end();
  rejectStream.end();

  console.log(`Wrote ${kept} verified pairs to ${outPath}`);
  console.log(`Logged ${rejected} rejected records to ${rejectLogPath}`);
}

if (process.argv[1].endsWith("latex_to_slang.js")) {
  try {
    run();
  } catch (error) {
    console.error(error.message);
    process.exit(1);
  }
}
