# 🧠 Neural Architect

Neural Architect is a sophisticated, browser-based Neural Network Assistant and Builder. It features a custom-built Recurrent Neural Network (RNN) engine implemented from scratch in pure JavaScript, allowing users to train and interact with an AI model directly in their web browser.

## ✨ Key Features

- **From-Scratch RNN Engine**: No external libraries (NumPy, TensorFlow, etc.) – everything from matrix multiplication to backpropagation is built with Vanilla JS.
- **Real-Time Interactive Training**: Watch the model learn in real-time as you chat. Use the "Reward" and "Penalize" system to shape the assistant's behavior.
- **🤖 Auto-Train Mode**: Bulk-train the model using custom trigger-response pairs with an automated mastery-based curriculum.
- **Live Metrics Dashboard**: Track "Loss Value" (Confusion Assessment) and "Mastery Percentage" through a dynamic visual interface.
- **State Persistence**: Save your trained "brain" as a JSON file and load it back later to continue its education.
- **Professional Dark UI**: A sleek, glassmorphic dashboard optimized for both desktop and mobile (PWA ready).

## 🚀 Getting Started

1. **Clone the repo**:
   ```bash
   git clone https://github.com/vicsanity623/Neural-Architect.git
   ```
2. **Launch**:
   Simply open `index.html` in any modern web browser. No server or compilation required.

## 🛠 Tech Stack

- **Core**: Vanilla HTML5, CSS3, ES6 JavaScript.
- **Neural Engine**: Custom Recurrent Neural Network (RNN) with Tanh activation and Cross-Entropy loss.
- **PWA**: Service Worker integration for offline accessibility.

## 📚 How to Teach

1. **Chat**: Send a message to the AI.
2. **Evaluate**: If the response is good, hit **Reward**. If it's poor, hit **Penalize** and provide a correction.
3. **Train**: Use the **Train 10 Epochs** button to perform deeper weight adjustments based on the current conversation history.
4. **Automate**: Use **Auto Train** to feed the AI specific knowledge sets (e.g., `Hello|Hi!`).

## 🧠 The Math Behind the Brain

The engine utilizes:
- **Matrix Operations**: Custom methods for multiplication, transposition, and element-wise scaling.
- **Gradient Clipping**: Prevents the "exploding gradient" problem common in RNNs.
- **Optimization**: RMSprop-like weight updates for stable training convergence.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Built with passion for Neural Networks and Web Technology.*
