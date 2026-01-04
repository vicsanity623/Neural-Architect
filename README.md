# 🧠 Neural Architect v2.0 (GRU Edition)

Neural Architect is a high-performance, browser-based Neural Network Assistant and Research Sandbox. It features a custom-built **Gated Recurrent Unit (GRU)** engine implemented entirely from scratch in pure JavaScript. This project demonstrates the power of modern web technologies by handling complex matrix mathematics, vector embeddings, and backpropagation through time (BPTT) directly on the client side without any server-side dependencies.

---

## 🚀 Vision

The goal of Neural Architect is to provide a transparent, accessible, and interactive platform for studying neural dynamics. By moving the "brain" into the browser, we eliminate the black-box nature of remote AI. Version 2.0 represents a massive leap forward, moving from character-level spelling to word-level conceptual understanding.

## ✨ Core Features

### 🧩 Word-Level GRU Engine
- **Architecture**: A Gated Recurrent Unit (GRU) with specialized Update and Reset gates. This solves the **Vanishing Gradient Problem**, allowing the AI to remember context from much earlier in the conversation compared to standard RNNs.
- **Concept Embeddings**: Includes a learnable **Embedding Layer** (32-dim). The AI converts words into dense vectors, allowing it to learn semantic relationships (e.g., understanding that "Hi" and "Hello" are mathematically similar).
- **No More Spelling**: Unlike character-level models, v2.0 sees whole words. It grasps concepts immediately without needing to learn how to spell "h-e-l-l-o" first.

### ⚡ Zero Dependencies
- **Pure JavaScript**: Built with vanilla ES6+. No TensorFlow, PyTorch, NumPy, or Python required.
- **Dynamic Vocabulary**: The model builds its dictionary on the fly. As you type new words, the matrix dynamically expands to accommodate them.

### 🤖 Advanced Training Systems
- **Progressive Curriculum Learning**: Automates the "step-up" training method. The AI masters one pair at a time, with cumulative restarts to ensure long-term memory retention before expanding the dataset.
- **Rotational Multi-Target Mastery**: Support for multiple valid responses (synonyms). The AI rotates its focus across variants until every response meets the mastery threshold.
- **Manual Feedback Loop**: Real-time "Reward" and "Penalize" system allows for surgical correction of the model's weights during live conversation.

### 📊 Live Analytics Dashboard
- **Loss Value**: Real-time tracking of the model's error rate.
- **Mastery Tracking**: Visual representation of the model's accuracy and consistency.
- **Vocab & Params**: Live counters for the size of the neural brain.

---

## 🛠 Technical Details

| Component | Detail |
| :--- | :--- |
| **Model Type** | Word-Level Gated Recurrent Unit (GRU) |
| **Gating** | Update (z) & Reset (r) Gates |
| **Activation** | Sigmoid (Gates) & Tanh (Candidate) |
| **Hidden State** | 128 Units |
| **Embedding Size** | 32 Dimensions |
| **Input/Output** | Tokenized Word Vectors |
| **Persistence** | JSON-based Weight & Vocab Export |

## 📚 Training Methodologies

### 1. Manual Tutoring
Engage in direct conversation. When the AI fails, click **Penalize**. The system will mathematically adjust the vector embedding of the incorrect word away from the context and pull the correct word closer.

### 2. Auto-Teach (Bulk Learning)
Input massive datasets in the `Trigger|Response` format.
*   **Example**: `Who are you?|I am Neural Architect,I am an AI`
*   The system uses the GRU's short-term memory to learn the sequence of words required to generate these specific phrases.

### 3. Progressive Curriculum (Recommended)
Enable **Curriculum Mode** during Auto-Teach. The system will:
1. Drill the first pair to mastery.
2. Automatically expand the set.
3. **Restart from the beginning** to ensure the AI utilizes the GRU's memory capacity to retain old pairs while learning new ones.

---

## 📦 Installation & Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/vicsanity623/Neural-Architect.git
   ```
2. **Launch**:
   Open `index.html` in any modern browser. 
3. **PWA Support**: 
   Install as an app on iOS/Android/Desktop via the browser's "Add to Home Screen" option for offline access.

---

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

---

*Neural Architect - Building Intelligence, One Neuron at a Time.*