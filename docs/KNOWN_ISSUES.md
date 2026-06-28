# Known Issues and Tracked Gaps

This file records bugs and discrepancies that were discovered, their root cause,
and their resolution. Each entry includes the fix reference so the full history
is preserved even after the issue is closed.

---

## [RESOLVED] STRUCT:OPEN missing from tokenizer/vocab.json

**Discovered:** During unit test implementation (Task 1, PR adding test suite)  
**Fixed:** Fix 3 (vocab.json v1.1)  
**Severity:** High — silent data corruption in neural training and inference  
**Affected path:** Neural solver only (FallbackSolver and GroqSolver unaffected)

### What was wrong

`tokenizer/slang_serializer.py` emits `"STRUCT:OPEN"` as the opening bracket
token for every fraction node and op-node it serializes. This token is defined
as a module-level constant:

```python
OPEN = "STRUCT:OPEN"
```

`tokenizer/vocab.json` defined `STRUCT:CLOSE` (ID 7) but had no entry for
`STRUCT:OPEN`. The `structure_tokens` section contained six tokens (IDs 4–9)
with no gap available for insertion without renumbering.

### Why it mattered

In `inference/solve.py`, `CalculusSolverInference._serialize_input()` converts
token strings to integer IDs using:

```python
self.vocab_map["token_to_id"].get(token, self.pad_id)
```

Because `"STRUCT:OPEN"` was absent from the vocab map, every occurrence of it
silently mapped to `self.pad_id` (ID 0, `[PAD]`). This corrupted the entire
input token sequence before it reached the transformer encoder — every
structural opening bracket was encoded as padding.

### Why it was not caught earlier

The FallbackSolver and GroqSolver do not use the vocab at all — they operate
on raw SLaNg dicts. The discrepancy only affects the neural path
(`CalculusSolverInference`), which requires a trained checkpoint to exercise.
In CI and local development without a checkpoint, the neural path is never
reached, so the corruption was invisible.

The unit test for `test_fraction_contains_struct_open` correctly asserted that
`"STRUCT:OPEN"` appears in the serialized token list, but the test had no
assertion that the token also exists in the vocab — it only tested the
serializer output, not the vocab lookup.

### The fix

`STRUCT:OPEN` was assigned ID **23** in `vocab.json` v1.1. ID 23 was chosen
because:

- IDs 4–9 (structure tokens) were fully occupied; inserting there would
  require renumbering existing tokens and invalidating all trained model weights
- ID 23 is the first unused ID after operation tokens end at 22
- Assigning it here shifts nothing and breaks no existing weights

### Files changed

| File | Change |
|---|---|
| `tokenizer/vocab.json` | Added `"STRUCT:OPEN": 23` to `structure_tokens`; version bumped to `1.1` |
| `tests/unit/test_slang_serializer.py` | Updated two comments that described this as a known discrepancy |
| `docs/KNOWN_ISSUES.md` | This file created |

### Verification

After the fix, the following confirms `STRUCT:OPEN` is correctly round-trippable
through the vocab:

```python
import json

with open("tokenizer/vocab.json") as f:
    vocab = json.load(f)

token_to_id = {}
for category in vocab.values():
    if isinstance(category, dict):
        token_to_id.update(category)

assert "STRUCT:OPEN" in token_to_id, "STRUCT:OPEN must be in vocab"
assert token_to_id["STRUCT:OPEN"] == 23, "STRUCT:OPEN must have ID 23"

id_to_token = {v: k for k, v in token_to_id.items()}
assert id_to_token[23] == "STRUCT:OPEN", "ID 23 must map back to STRUCT:OPEN"

print("STRUCT:OPEN correctly registered at ID 23")
```

---

## Filing new issues

To add a new entry, copy the template below and fill it in:

```markdown
## [STATUS] Short description

**Discovered:** When/how found  
**Fixed:** Fix reference or "Pending"  
**Severity:** Low / Medium / High  
**Affected path:** Which solver modes / components are affected

### What was wrong
...

### Why it mattered
...

### The fix
...

### Files changed
...
```
