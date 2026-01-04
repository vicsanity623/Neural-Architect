import torch
import torch.nn as nn
import json
import numpy as np
import os

class NeuralArchitectRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralArchitectRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Weights
        self.Wxh = nn.Parameter(torch.zeros(hidden_size, input_size))
        self.Whh = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.Why = nn.Parameter(torch.zeros(output_size, hidden_size))
        
        # Biases (JS uses [rows, 1] matrices, we use 1D tensors)
        self.bh = nn.Parameter(torch.zeros(hidden_size))
        self.by = nn.Parameter(torch.zeros(output_size))

    def forward(self, x, h):
        """
        x: (batch, input_size) - One-hot input
        h: (batch, hidden_size) - Previous hidden state
        """
        # h = tanh(Wxh * x + Whh * h + bh)
        # Note: torch.matmul(x, self.Wxh.t()) is equivalent to Wxh @ x if x is (input_size, 1)
        h_next = torch.tanh(
            torch.matmul(x, self.Wxh.t()) + 
            torch.matmul(h, self.Whh.t()) + 
            self.bh
        )
        # y = Why * h + by
        y = torch.matmul(h_next, self.Why.t()) + self.by
        return y, h_next

def export_to_onnx(json_path, onnx_path):
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    print(f"Loading {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract RNN weights
    rnn_data = data['rnn']
    Wxh = np.array(rnn_data['Wxh'])
    Whh = np.array(rnn_data['Whh'])
    Why = np.array(rnn_data['Why'])
    # Biases are [rows, 1] in the JSON
    bh = np.array(rnn_data['bh']).flatten()
    by = np.array(rnn_data['by']).flatten()

    hidden_size, input_size = Wxh.shape
    output_size, _ = Why.shape

    print(f"Model Specs: Input={input_size}, Hidden={hidden_size}, Output={output_size}")

    model = NeuralArchitectRNN(input_size, hidden_size, output_size)
    
    # Load weights into the model
    with torch.no_grad():
        model.Wxh.copy_(torch.from_numpy(Wxh).float())
        model.Whh.copy_(torch.from_numpy(Whh).float())
        model.Why.copy_(torch.from_numpy(Why).float())
        model.bh.copy_(torch.from_numpy(bh).float())
        model.by.copy_(torch.from_numpy(by).float())

    model.eval()

    # Prepare dummy inputs for export
    # Batch size 1
    dummy_x = torch.zeros(1, input_size)
    dummy_h = torch.zeros(1, hidden_size)

    print(f"Exporting to {onnx_path}...")
    torch.onnx.export(
        model,
        (dummy_x, dummy_h),
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input', 'hidden_in'],
        output_names=['output', 'hidden_out'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'hidden_in': {0: 'batch_size'},
            'output': {0: 'batch_size'},
            'hidden_out': {0: 'batch_size'}
        }
    )
    print("Export successful!")

if __name__ == "__main__":
    import sys
    json_input = "neural_brain.json"
    if len(sys.argv) > 1:
        json_input = sys.argv[1]
    
    onn_output = json_input.replace(".json", ".onnx")
    if onn_output == json_input:
        onn_output += ".onnx"
        
    export_to_onnx(json_input, onn_output)
