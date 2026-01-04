import onnxruntime as ort
import numpy as np
import sys

# Vocab must match index.html exactly
CHARS_STRING = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,?!:;\n-'
CHARS = list(CHARS_STRING)
VOCAB_SIZE = len(CHARS)
CHAR_TO_INDEX = {c: i for i, c in enumerate(CHARS)}
INDEX_TO_CHAR = CHARS

def one_hot(idx, size):
    arr = np.zeros((1, size), dtype=np.float32)
    if 0 <= idx < size:
        arr[0, idx] = 1.0
    return arr

def sample(probs, temperature=0.7):
    # Apply temperature
    probs = np.log(probs + 1e-10) / temperature
    exp_probs = np.exp(probs - np.max(probs))
    probs = exp_probs / np.sum(exp_probs)
    
    # Sample from the distribution
    return np.random.choice(len(probs), p=probs)

class LocalInference:
    def __init__(self, onnx_model_path):
        print(f"Loading model: {onnx_model_path}")
        self.session = ort.InferenceSession(onnx_model_path)
        
        # Get hidden size from the model inputs
        # input_names: ['input', 'hidden_in']
        # input_shapes: [[batch, vocab], [batch, hidden]]
        hidden_in = self.session.get_inputs()[1]
        self.hidden_size = hidden_in.shape[1]
        self.input_size = self.session.get_inputs()[0].shape[1]
        
        print(f"Model initialized. Input Size: {self.input_size}, Hidden Size: {self.hidden_size}")

    def generate(self, prefix, max_len=200, temperature=0.7):
        # Initialize hidden state
        h = np.zeros((1, self.hidden_size), dtype=np.float32)
        
        # Process prefix
        for char in prefix:
            idx = CHAR_TO_INDEX.get(char, -1)
            x = one_hot(idx, self.input_size)
            
            # Run inference: y, h_next = session.run(None, {'input': x, 'hidden_in': h})
            _, h = self.session.run(None, {'input': x, 'hidden_in': h})

        # Generate response
        response = ""
        for _ in range(max_len):
            # We need a dummy input for the current char, but wait...
            # The JS code uses the last char of the prefix to get the first hidden state for generation.
            # In our forward pass, we already updated 'h' with the last char of the prefix.
            # Now we need to predict the NEXT char.
            
            # Actually, the sessions's output[0] is the prediction for the NEXT char given 'input' and 'hidden_in'.
            # But the hidden state update happens INSIDE for the NEXT step.
            
            # So if we just processed the last char of prefix, 'h' is the hidden state AFTER that char.
            # The output 'y' from that last step is the logits for the next char.
            # But we didn't capture 'y' in the loop above. Let's fix that.
            pass

        # Corrected Loop
        h = np.zeros((1, self.hidden_size), dtype=np.float32)
        y = None
        for char in prefix:
            idx = CHAR_TO_INDEX.get(char, -1)
            x = one_hot(idx, self.input_size)
            y, h = self.session.run(None, {'input': x, 'hidden_in': h})

        response = ""
        for _ in range(max_len):
            # y contains the logits for the next char
            probs = np.squeeze(y)
            # Softmax is not in the model (based on my export script), or is it?
            # My export script: y = Why @ h_next + by. No softmax.
            # So y are logits.
            
            # Manual softmax with temperature
            exp_y = np.exp((probs - np.max(probs)) / temperature)
            probs = exp_y / np.sum(exp_y)
            
            next_idx = np.random.choice(len(probs), p=probs)
            next_char = INDEX_TO_CHAR[next_idx]
            
            if next_char == '\n':
                break
                
            response += next_char
            
            # Prepare for next step
            x = one_hot(next_idx, self.input_size)
            y, h = self.session.run(None, {'input': x, 'hidden_in': h})
            
        return response.strip()

def main():
    if len(sys.argv) < 2:
        print("Usage: python local_inference.py <model.onnx>")
        return

    model_path = sys.argv[1]
    infer = LocalInference(model_path)
    
    print("\n--- Neural Architect Local Inference ---")
    print("Type 'exit' to quit.\n")
    
    history = ""
    
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ['exit', 'quit']:
                break
                
            history += f"User: {user_input}\nAssistant: "
            
            # We use the whole history as prefix to maintain context, matching JS
            response = infer.generate(history, temperature=0.7)
            print(f"Assistant: {response}")
            
            history += f"{response}\n"
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
