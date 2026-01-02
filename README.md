# 🧠 Neural Architect v1.1.3

Neural Architect is a high-performance, browser-based Neural Network Assistant and Research Sandbox. It features a custom-built **Recurrent Neural Network (RNN)** engine implemented entirely from scratch in pure JavaScript. This project demonstrates the power of modern web technologies by handling complex matrix mathematics, backpropagation through time (BPTT), and real-time training directly on the client side without any server-side dependencies.

---

## 🚀 Vision

The goal of Neural Architect is to provide a transparent, accessible, and interactive platform for studying neural dynamics. By moving the "brain" into the browser, we eliminate the black-box nature of remote AI, allowing users to shape, monitor, and master their own local intelligence agents.

## ✨ Core Features

### 🧮 Custom 512-Unit RNN Engine
- **Architecture**: A deep character-level RNN with 512 hidden units.
- **Initialization**: Optimized with **Xavier/Glorot Initialization** to ensure stable signal propagation and prevent `tanh` saturation in larger layers.
- **Zero Dependencies**: Built with vanilla ES6+ JavaScript. No TensorFlow, PyTorch, or NumPy.
- **Precision Control**: Adjustable **Creativity (Temperature)** sampling allows for deterministic (Low Temp) vs. expressive (High Temp) output generations.

### 🤖 Advanced Training Systems
- **Progressive Curriculum Learning**: Automates the "step-up" training method. The AI masters one pair at a time, cumulative restarts ensure long-term memory retention before expanding the dataset.
- **Rotational Multi-Target Mastery**: Support for multiple valid responses (synonyms). The AI rotates its focus across variants until every response meets the 6-streak mastery threshold.
- **Manual Feedback Loop**: Real-time "Reward" and "Penalize" system allows for surgical correction of the model's weights during live conversation.

### � Live Analytics Dashboard
- **Loss Value (Confusion Assessment)**: Real-time tracking of the model's objective function performance.
- **Mastery Tracking**: Visual representation of the model's accuracy and consistency across the training set.
- **Training Loops**: Automated loop counting to track dataset exposure over time.

---

## 🛠 Technical Details

| Component | Detail |
| :--- | :--- |
| **Model Type** | Character-level Recurrent Neural Network (RNN) |
| **Activation** | Hyperbolic Tangent (tanh) |
| **Optimization** | RMSprop-inspired Gradient Descent |
| **Weight Init** | Xavier/Glorot Normalization |
| **Hidden Layers** | 512 Units |
| **Input/Output** | One-Hot Encoded Characters |
| **Persistence** | JSON-based Weight Export/Import |

## 📚 Training Methodologies

### 1. Manual Tutoring
Engage in direct conversation. When the AI fails, provide a correction. The **⚡ Train (5 Epochs)** button performs deep reinforced learning on that specific correction history to cement the new knowledge.

### 2. Auto-Teach (Bulk Learning)
Input massive datasets in the `Trigger|Response` format. Support for synonyms like `Hello|Hi,Hey,Greetings` allows for more human-like response variability.

### 3. Progressive Curriculum (Recommended)
Enable **Curriculum Mode** during Auto-Teach. The system will:
1. Drill the first pair to mastery.
2. Automatically expand the set.
3. **Restart from the beginning** to ensure the AI hasn't forgotten earlier pairs.
4. Auto-save the model at every milestone.

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

## � License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

---

*Neural Architect - Building Intelligence, One Neuron at a Time.*
