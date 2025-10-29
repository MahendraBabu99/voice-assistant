import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import json
import os
import math
from pathlib import Path
import csv

# 1. Load and prepare dataset correctly
data_path = r"C:\Users\chapa mahindra\Downloads\archive\Conversation.csv"

def load_conversation_data(data_path):
    dialogs = []
    with open(data_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Extract question and answer
            q = row.get('question') or row.get('Question') or list(row.values())[0]
            a = row.get('answer') or row.get('Answer') or list(row.values())[1]
            
            if q and a:
                # Create conversation pairs
                dialogs.append([q.strip(), a.strip()])
    return dialogs

# Load the data
dialogs = load_conversation_data(data_path)


# 2. Build tokenizer
def build_tokenizer(sentences, path="chat_tokenizer.json"):
    if os.path.exists(path):
        tokenizer = Tokenizer.from_file(path)
    else:
        tokenizer = Tokenizer(models.BPE(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"])
        tokenizer.train_from_iterator(sentences, trainer)
        tokenizer.save(path)
    return tokenizer

# Prepare sentences for tokenizer training
sentences = []
for dialog in dialogs:
    sentences.extend(dialog)

tokenizer = build_tokenizer(sentences)
vocab_size = tokenizer.get_vocab_size()
# print(f"Tokenizer vocab size: {vocab_size}")

# 3. Improved Dataset class
class ChatDataset(Dataset):
    def __init__(self, conversations, tokenizer, max_len=50):
        self.data = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        for dialog in conversations:
            if len(dialog) >= 2:  # Ensure we have at least a question and answer
                # Encode source (question) and target (answer)
                src_encoding = tokenizer.encode(dialog[0])
                tgt_encoding = tokenizer.encode(dialog[1])
                
                # Truncate if necessary and add special tokens
                src_ids = src_encoding.ids[:max_len-2]  # Reserve space for [SOS] and [EOS]
                tgt_ids = tgt_encoding.ids[:max_len-2]
                
                self.data.append({
                    "src": [1] + src_ids + [2],  # [SOS] = 1, [EOS] = 2
                    "tgt": [1] + tgt_ids + [2]
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx]["src"], dtype=torch.long),
            torch.tensor(self.data[idx]["tgt"], dtype=torch.long)
        )

# Create dataset
chat_dataset = ChatDataset(dialogs, tokenizer)
# print(f"Dataset size: {len(chat_dataset)}")

# 4. Collate function
def collate_fn(batch):
    src_batch = [x[0] for x in batch]
    tgt_batch = [x[1] for x in batch]
    
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    
    return src_padded, tgt_padded

# Create data loader
train_loader = DataLoader(chat_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

class Embeddings(nn.Module):
        def __init__(self, d_model, vocab_size):
            super().__init__()
            self.d_model = d_model
            self.embeddings = nn.Embedding(vocab_size, d_model)

        def forward(self, x):
            return self.embeddings(x) * math.sqrt(self.d_model)
class Positionalencoding(nn.Module):
        def __init__(self, d_model, seq_len, dropout):
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            pe = torch.zeros(seq_len, d_model)
            positions = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
            div_terms = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(positions * div_terms)
            pe[:, 1::2] = torch.cos(positions * div_terms)
            pe = pe.unsqueeze(0)
            self.register_buffer("pe", pe)

        def forward(self, x):
            x = x + self.pe[:, :x.size(1), :].detach()
            return self.dropout(x)

class Layernormalization(nn.Module):
        def __init__(self, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.alpha = nn.Parameter(torch.ones(1))
            self.bias = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True)
            return self.alpha * (x - mean) / (std + self.eps) + self.bias

class Feedforward(nn.Module):
        def __init__(self, d_model, d_ff, dropout):
            super().__init__()
            self.linear_1 = nn.Linear(d_model, d_ff)
            self.linear_2 = nn.Linear(d_ff, d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class Multiheadattentionblock(nn.Module):
        def __init__(self, d_model, h, dropout):
            super().__init__()
            assert d_model % h == 0
            self.d_model = d_model
            self.h = h
            self.d_k = d_model // h
            self.w_q = nn.Linear(d_model, d_model)
            self.w_k = nn.Linear(d_model, d_model)
            self.w_v = nn.Linear(d_model, d_model)
            self.w_o = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)

        @staticmethod
        def attention(query, key, value, mask, dropout):
            d_k = query.size(-1)
            scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            scores = scores.softmax(dim=-1)
            if dropout is not None:
                scores = dropout(scores)
            return (scores @ value), scores

        def forward(self, q, k, v, mask):
            query = self.w_q(q)
            key = self.w_k(k)
            value = self.w_v(v)

            batch_size = query.size(0)

            query = query.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            key = key.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            value = value.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

            x, _ = self.attention(query, key, value, mask, self.dropout)

            x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
            return self.w_o(x)

class Residualconnection(nn.Module):
        def __init__(self, dropout):
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = Layernormalization()

        def forward(self, x, sublayer):
            return x + self.dropout(sublayer(self.norm(x)))

class Encoderblock(nn.Module):
        def __init__(self, self_attention_block, feed_forward_block, dropout):
            super().__init__()
            self.self_attention_block = self_attention_block
            self.feed_forward_block = feed_forward_block
            self.residual_connection = nn.ModuleList([Residualconnection(dropout) for _ in range(2)])

        def forward(self, x, src_mask):
            x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
            x = self.residual_connection[1](x, self.feed_forward_block)
            return x

class Encoder(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.layers = layers
            self.norm = Layernormalization()

        def forward(self, x, mask):
            for layer in self.layers:
                x = layer(x, mask)
            return self.norm(x)

class Decoderblock(nn.Module):
        def __init__(self, self_attention_block, cross_attention_block, feed_forward_block, dropout):
            super().__init__()
            self.self_attention_block = self_attention_block
            self.cross_attention_block = cross_attention_block
            self.feed_forward_block = feed_forward_block
            self.residual_connections = nn.ModuleList([Residualconnection(dropout) for _ in range(3)])

        def forward(self, x, encoder_output, src_mask, tgt_mask):
            x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
            x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
            x = self.residual_connections[2](x, self.feed_forward_block)
            return x

class Decoder(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.layers = layers
            self.norm = Layernormalization()

        def forward(self, x, encoder_output, src_mask, tgt_mask):
            for layer in self.layers:
                x = layer(x, encoder_output, src_mask, tgt_mask)
            return self.norm(x)

class Projectionlayer(nn.Module):
        def __init__(self, d_model, vocab_size):
            super().__init__()
            self.proj = nn.Linear(d_model, vocab_size)

        def forward(self, x):
            return torch.log_softmax(self.proj(x), dim=-1)

class Transformer(nn.Module):
        def __init__(self, encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.src_embed = src_embed
            self.tgt_embed = tgt_embed
            self.src_pos = src_pos
            self.tgt_pos = tgt_pos
            self.projection_layer = projection_layer

        def encode(self, src, src_mask):
            src = self.src_embed(src)
            src = self.src_pos(src)
            return self.encoder(src, src_mask)

        def decode(self, encoder_output, src_mask, tgt, tgt_mask):
            tgt = self.tgt_embed(tgt)
            tgt = self.tgt_pos(tgt)
            return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

        def project(self, x):
            return self.projection_layer(x)

        @staticmethod
        def build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len,
                            d_model=256, N=2, h=4, dropout=0.1, d_ff=1024):
            src_embedding = Embeddings(d_model, src_vocab_size)
            tgt_embedding = Embeddings(d_model, tgt_vocab_size)

            src_pos = Positionalencoding(d_model, src_seq_len, dropout)
            tgt_pos = Positionalencoding(d_model, tgt_seq_len, dropout)

            encoder_blocks = [
                Encoderblock(Multiheadattentionblock(d_model, h, dropout),
                            Feedforward(d_model, d_ff, dropout), dropout)
                for _ in range(N)
            ]

            decoder_blocks = [
                Decoderblock(Multiheadattentionblock(d_model, h, dropout),
                            Multiheadattentionblock(d_model, h, dropout),
                            Feedforward(d_model, d_ff, dropout), dropout)
                for _ in range(N)
            ]

            encoder = Encoder(nn.ModuleList(encoder_blocks))
            decoder = Decoder(nn.ModuleList(decoder_blocks))
            projection_layer = Projectionlayer(d_model, tgt_vocab_size)

            model = Transformer(encoder, decoder, src_embedding, tgt_embedding,
                                src_pos, tgt_pos, projection_layer)

            for p in model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

            return model


# 6. Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

model = Transformer.build_transformer(
    src_vocab_size=vocab_size,
    tgt_vocab_size=vocab_size,
    src_seq_len=50,
    tgt_seq_len=50,
    d_model=256,
    N=2,
    h=4,
    d_ff=1024
).to(device)

# 7. Improved training setup
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
# Use CrossEntropyLoss instead of NLLLoss for better stability
loss_fn = nn.CrossEntropyLoss(ignore_index=0)

# print("Training started...")
if __name__ == "__main__":
# 8. Training loop with improvements
    for epoch in range(20):
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch_idx, (src, tgt) in enumerate(train_loader):
            src = src.to(device)
            tgt = tgt.to(device)
            
            # Prepare input and target sequences
            tgt_input = tgt[:, :-1]   # Remove last token for input
            tgt_output = tgt[:, 1:]   # Remove first token for target
            
            # Create masks
            src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(device)
            
            # Create causal mask for decoder
            seq_len = tgt_input.size(1)
            tgt_mask = torch.tril(torch.ones(seq_len, seq_len)).bool().unsqueeze(0).to(device)
            
            # Forward pass
            enc_out = model.encode(src, src_mask)
            dec_out = model.decode(enc_out, src_mask, tgt_input, tgt_mask)
            logits = model.project(dec_out)
            
            # Calculate loss
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent explosions
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / batch_count
        print(f"📘 Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")

    # 9. Save model and tokenizer
    model_path = "chatbot_transformer.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer_path': 'chat_tokenizer.json',
        'vocab_size': vocab_size,
        'config': {
            'd_model': 256,
            'n_layers': 2,
            'n_heads': 4,
            'd_ff': 1024
        }
    }, model_path)

    print("✅ Model saved to:", os.path.abspath(model_path))
    print("Training completed!")
