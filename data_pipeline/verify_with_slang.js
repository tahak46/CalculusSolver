import { verify } from "./verify.js";

const args = process.argv.slice(2);
const inputIndex = args.indexOf("--input");
const outputIndex = args.indexOf("--output");

if (inputIndex === -1 || outputIndex === -1) {
  console.error(
    "Usage: node data_pipeline/verify_with_slang.js --input <json-string> --output <json-string>",
  );
  process.exit(2);
}

try {
  const input = JSON.parse(args[inputIndex + 1]);
  const output = JSON.parse(args[outputIndex + 1]);
  const result = verify(input, output);
  console.log(JSON.stringify(result));
  process.exit(result.verified ? 0 : 1);
} catch (error) {
  console.error(
    JSON.stringify({ verified: false, confidence: 0, error: error.message }),
  );
  process.exit(2);
}
