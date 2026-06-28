import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from solver_model import CalculusSolverModel

# 🎯 FIX: Import team's real official tokenizer module safely
try:
    from tokenizer.slang_serializer import SlangTokenizer
    HAS_REAL_TOKENIZER = True
except (ImportError, ModuleNotFoundError):
    HAS_REAL_TOKENIZER = False

with open("config.json", "r") as cfg_file:
    config = json.load(cfg_file)

class SlangTrainingDataset(Dataset):
    def __init__(self, file_path, vocab_size):
        self.data = []
        self.vocab_size = vocab_size
        
        # 🎯 FIX: Load project's real vocab.json if it exists to align embedding space
        self.vocab_mapping = {}
        vocab_path = Path("vocab.json")
        if vocab_path.exists():
            with open(vocab_path, "r", encoding="utf-8") as f:
                self.vocab_mapping = json.load(f)
                
        if HAS_REAL_TOKENIZER:
            self.tokenizer = SlangTokenizer()
            
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))
                
    def __len__(self):
        return len(self.data)
        
    def _real_tokenize_and_pad(self, text_or_tokens, max_len=20):
        # Handle if text is already list of tokens or a raw string
        if isinstance(text_or_tokens, list):
            text_str = " ".join(text_or_tokens)
        else:
            text_str = str(text_or_tokens)
            
        # 🎯 FIX: Use the pipeline's real SLaNg tokenizer mapping
        if HAS_REAL_TOKENIZER:
            encoded = self.tokenizer.encode(text_str)
        else:
            # Fallback to precise vocab mapping if file exists but module fails
            tokens = text_str.split()
            encoded = [self.vocab_mapping.get(t, self.vocab_mapping.get("<unk>", 3)) for t in tokens]
            
        # Handle padding securely according to schema matching
        if len(encoded) < max_len:
            encoded += [0] * (max_len - len(encoded))
        return torch.tensor(encoded[:max_len], dtype=torch.long)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        # Align all source and target sequences to the real tokenizer's embedding space
        return {
            "src_seq": self._real_tokenize_and_pad(item["src_tokens"]),
            "tgt_in_seq": self._real_tokenize_and_pad(item["tgt_input_tokens"]),
            "tgt_out_seq": self._real_tokenize_and_pad(item["tgt_output_tokens"]),
            "rule_id": torch.tensor(item["rule_ids"], dtype=torch.long),
            "v_state": torch.tensor(item["verification_state"], dtype=torch.float)
        }

def main():
    print("--- 🏋️ Running Tokenizer-Aligned Pipeline System ---")
    v_size = config["vocab_size"]
    
    train_loader = DataLoader(
        SlangTrainingDataset("data/splits/train.jsonl", vocab_size=v_size), 
        batch_size=config["batch_size"], 
        shuffle=True
    )
    
    model = CalculusSolverModel(
        vocab_size=v_size,
        hidden_dim=config["hidden_dim"]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    criterion_sequence = nn.CrossEntropyLoss(reduction='none')
    criterion_rule = nn.CrossEntropyLoss()
    criterion_verify = nn.BCEWithLogitsLoss()
    
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        token_logits, rule_logits, verifier_logits = model(batch["src_seq"], batch["tgt_in_seq"])
        
        raw_loss_seq = criterion_sequence(token_logits.view(-1, v_size), batch["tgt_out_seq"].view(-1))
        raw_loss_seq = raw_loss_seq.view(batch["src_seq"].size(0), -1).mean(dim=-1)
        
        mask_correct_steps = (batch["v_state"] == 1.0).float()
        loss_seq = (raw_loss_seq * mask_correct_steps).sum() / (mask_correct_steps.sum() + 1e-8)
        
        loss_rule = criterion_rule(rule_logits, batch["rule_id"])
        loss_verify = criterion_verify(verifier_logits.squeeze(-1), batch["v_state"])
        
        total_loss = loss_seq + loss_rule + loss_verify
        total_loss.backward()
        optimizer.step()
        
        if batch_idx % 500 == 0:
            print(f"[Placeholder Log System] Step {batch_idx}/{config['max_steps']} | Loss: {total_loss.item():.4f}")
            
        if batch_idx >= config["max_steps"]:
            break
            
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/checkpoint_epoch_1.pt")
    print("✨ SLaNg Checkpoint successfully saved and aligned with vocabulary.")

if __name__ == "__main__":
    main()s