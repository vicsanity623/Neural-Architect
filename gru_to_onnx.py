import torch
import torch.nn as nn
import json
import numpy as np
import os

class NeuralArchitectGRU(nn.Module):
    def __init__(self, vocab_size, hidden_size, embed_size):
        super(NeuralArchitectGRU, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        
        # 1. Embedding Layer
        # JS stores as [EmbedSize, VocabSize], we will transpose on load
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # 2. GRU Weights
        # Input to gates is size: EmbedSize + HiddenSize
        input_dim = embed_size + hidden_size
        
        # Update Gate (z)
        self.Wz = nn.Parameter(torch.zeros(hidden_size, input_dim))
        self.bz = nn.Parameter(torch.zeros(hidden_size))
        
        # Reset Gate (r)
        self.Wr = nn.Parameter(torch.zeros(hidden_size, input_dim))
        self.br = nn.Parameter(torch.zeros(hidden_size))
        
        # Candidate Hidden (h~)
        self.Wh = nn.Parameter(torch.zeros(hidden_size, input_dim))
        self.bh = nn.Parameter(torch.zeros(hidden_size))
        
        # 3. Output Decoder
        self.Why = nn.Parameter(torch.zeros(vocab_size, hidden_size))
        self.by = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, token_idx, h_prev):
        """
        token_idx: (batch) - Integer Token ID
        h_prev: (batch, hidden_size) - Previous hidden state
        """
        # 1. Get Embedding
        # x shape: (batch, embed_size)
        x = self.embedding(token_idx)
        
        # 2. Concatenate [x, h_prev]
        # xh shape: (batch, embed_size + hidden_size)
        xh = torch.cat([x, h_prev], dim=1)
        
        # 3. Update Gate z = sigmoid(Wz * xh + bz)
        z = torch.sigmoid(torch.matmul(xh, self.Wz.t()) + self.bz)
        
        # 4. Reset Gate r = sigmoid(Wr * xh + br)
        r = torch.sigmoid(torch.matmul(xh, self.Wr.t()) + self.br)
        
        # 5. Candidate Hidden h~
        # In JS: rh = r * h; concat_cand = [x, rh]
        rh = r * h_prev
        concat_cand = torch.cat([x, rh], dim=1)
        h_cand = torch.tanh(torch.matmul(concat_cand, self.Wh.t()) + self.bh)
        
        # 6. New Hidden State
        # h = (1-z)*h_prev + z*h_cand
        h_next = (1 - z) * h_prev + z * h_cand
        
        # 7. Output Logits
        # y = Why * h + by
        logits = torch.matmul(h_next, self.Why.t()) + self.by
        
        return logits, h_next

def export_to_onnx(json_path, onnx_path):
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    print(f"Loading {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 1. Extract and Process Weights
    rnn_data = data['rnn']
    
    # Embeddings: JS is [Embed, Vocab], Torch wants [Vocab, Embed]. We transpose.
    Embed_js = np.array(rnn_data['Embed'])
    Embed_w = Embed_js.T 
    
    # Helper to load and flatten
    def load_w(name): return np.array(rnn_data[name])
    def load_b(name): return np.array(rnn_data[name]).flatten()

    Wz, bz = load_w('Wz'), load_b('bz')
    Wr, br = load_w('Wr'), load_b('br')
    Wh, bh = load_w('Wh'), load_b('bh')
    Why, by = load_w('Why'), load_b('by')

    # 2. Determine Dimensions
    vocab_size, embed_size = Embed_w.shape
    hidden_size, _ = Why.shape.T if len(Why.shape) > 1 else (0,0) # Safety check
    # Check hidden size via Wz [Hidden, Input]
    hidden_size = Wz.shape[0]

    print(f"Model Specs: Vocab={vocab_size}, Embed={embed_size}, Hidden={hidden_size}")

    # 3. Initialize Model
    model = NeuralArchitectGRU(vocab_size, hidden_size, embed_size)
    
    # 4. Load Weights into PyTorch Model
    with torch.no_grad():
        model.embedding.weight.copy_(torch.from_numpy(Embed_w).float())
        
        model.Wz.copy_(torch.from_numpy(Wz).float())
        model.bz.copy_(torch.from_numpy(bz).float())
        
        model.Wr.copy_(torch.from_numpy(Wr).float())
        model.br.copy_(torch.from_numpy(br).float())
        
        model.Wh.copy_(torch.from_numpy(Wh).float())
        model.bh.copy_(torch.from_numpy(bh).float())
        
        model.Why.copy_(torch.from_numpy(Why).float())
        model.by.copy_(torch.from_numpy(by).float())

    model.eval()

    # 5. Export to ONNX
    # Dummy Input: Batch size 1. Input is Int (Long), Hidden is Float.
    dummy_input = torch.tensor([0], dtype=torch.long) # Token Index
    dummy_hidden = torch.zeros(1, hidden_size)

    print(f"Exporting to {onnx_path}...")
    torch.onnx.export(
        model,
        (dummy_input, dummy_hidden),
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['token_id', 'hidden_in'],
        output_names=['logits', 'hidden_out'],
        dynamic_axes={
            'token_id': {0: 'batch_size'},
            'hidden_in': {0: 'batch_size'},
            'logits': {0: 'batch_size'},
            'hidden_out': {0: 'batch_size'}
        }
    )
    print("Export successful! Compatible with v2.3 Deep/Turbo")

if __name__ == "__main__":
    import sys
    # Default filename matching your HTML save function
    json_input = "gru_brain_v2.json" 
    
    if len(sys.argv) > 1:
        json_input = sys.argv[1]
    
    onn_output = json_input.replace(".json", ".onnx")
    # Safety if extension wasn't there
    if onn_output == json_input:
        onn_output += ".onnx"
        
    export_to_onnx(json_input, onn_output)