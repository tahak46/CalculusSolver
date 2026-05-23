import fs from 'fs';

const vocab = JSON.parse(fs.readFileSync('tokenizer/vocab.json', 'utf8'));
const flat = {};

for (const [category, entries] of Object.entries(vocab)) {
  if (category.startsWith('_')) continue;
  for (const [token, id] of Object.entries(entries)) {
    flat[token] = id;
  }
}

fs.writeFileSync('tokenizer/vocab_flat.json', JSON.stringify(flat, null, 2));
console.log(`Written ${Object.keys(flat).length} tokens to vocab_flat.json`);