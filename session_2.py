# =====================================================================
# 🌟 Kawaii-Style MLP – Gradient Descent Adventure ♡
# =====================================================================
#
#  A cheerful, hand-written Multi-Layer Perceptron using only NumPy!
#  Perfect for learning how gradient descent and backpropagation really work.
#
#  You will see:
#  • Forward pass with cozy tanh activation
#  • MSE loss
#  • Every gradient calculation step-by-step
#  • Gentle parameter updates via gradient descent
#
#  Goal of this demo:
#  Watch the loss happily go down while the network learns to approximate
#  y ≈ sin(x₁ + x₂) + a little noise   ✨
#
#  Just run this file and enjoy the journey! 🍰
#
#  Dependencies: numpy, matplotlib (for the pretty loss curve)
#
# =====================================================================

import numpy as np
import matplotlib.pyplot as plt


class KawaiiMLP:
    """✨ Friendly 3-layer MLP ready for gradient descent fun! ♡"""

    def __init__(self, in_dim=2, hidden=32, out_dim=1, lr=0.015):
        """
        Hello friend! Let's create a cute little network together~ ♡
        
        Args:
            in_dim    → number of input features
            hidden    → neurons in each hidden layer (the thinking buddies)
            out_dim   → number of outputs
            lr        → learning rate (how big each happy step is)
        """
        self.lr = lr

        # Small random weights so we start gently
        self.w1 = (np.random.randn(in_dim, hidden) * 0.12).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)

        self.w2 = (np.random.randn(hidden, hidden) * 0.12).astype(np.float32)
        self.b2 = np.zeros(hidden, dtype=np.float32)

        self.w3 = (np.random.randn(hidden, out_dim) * 0.08).astype(np.float32)
        self.b3 = np.zeros(out_dim, dtype=np.float32)

    def forward(self, x):
        """Forward pass — thinking really hard! ♡"""
        h1 = np.tanh(np.dot(x, self.w1) + self.b1)     # first hug layer
        h2 = np.tanh(np.dot(h1, self.w2) + self.b2)    # second hug layer
        y  = np.tanh(np.dot(h2, self.w3) + self.b3)    # final answer

        return y, (x, h1, h2, y)

    def train_batch(self, x_batch, y_batch):
        """
        One gradient descent step — learning and growing stronger! ✨
        Returns current MSE loss so we can cheer when it drops ♡
        """
        y_pred, (x, h1, h2, y_out) = self.forward(x_batch)

        # How far off were we?
        diff = y_pred - y_batch
        loss = np.mean(diff ** 2)

        n = x.shape[0]
        dy = (2.0 / n) * diff

        # Backpropagation = sending love notes backward to fix mistakes ♡
        dz3 = dy * (1.0 - y_out ** 2)
        gw3 = np.dot(h2.T, dz3)
        gb3 = np.sum(dz3, axis=0)

        dh2 = np.dot(dz3, self.w3.T)
        dz2 = dh2 * (1.0 - h2 ** 2)
        gw2 = np.dot(h1.T, dz2)
        gb2 = np.sum(dz2, axis=0)

        dh1 = np.dot(dz2, self.w2.T)
        dz1 = dh1 * (1.0 - h1 ** 2)
        gw1 = np.dot(x.T, dz1)
        gb1 = np.sum(dz1, axis=0)

        # Happy updates!
        self.w3 -= self.lr * gw3
        self.b3 -= self.lr * gb3
        self.w2 -= self.lr * gw2
        self.b2 -= self.lr * gb2
        self.w1 -= self.lr * gw1
        self.b1 -= self.lr * gb1

        return loss

    def __repr__(self):
        return (f"✨ KawaiiMLP ✨  (in={self.w1.shape[0]}, hidden={self.w1.shape[1]}, "
                f"out={self.b3.shape[0]}, lr={self.lr:.4f}) ♡")


# ────────────────────────────────────────────────────────────────
#   ✨ Fun Demo: Learning sin(x₁ + x₂) with gradient descent magic
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("✨ Welcome to the Gradient Descent Party! ♡\n")
    print("Target function:   y ≈ sin(x₁ + x₂) + tiny sparkle noise\n")
    print("Goal: watch the loss smile and get smaller over time~ 🌸\n")

    # ── Meet our friendly network ────────────────────────────────
    model = KawaiiMLP(in_dim=2, hidden=32, out_dim=1, lr=0.015)
    print(model, "\n")

    # ── Prepare playful training data ─────────────────────────────
    np.random.seed(42)
    N = 4000
    x_train = np.random.uniform(-3.0, 3.0, (N, 2))
    y_true = np.sin(x_train[:, 0] + x_train[:, 1])[:, np.newaxis]
    y_train = y_true + np.random.randn(N, 1) * 0.07
    y_train = np.tanh(y_train)   # keep outputs cozy in [-1, 1]

    # ── Training time! Let's cheer every few steps ────────────────
    epochs = 400
    batch_size = 64
    losses = []

    print(f"Starting training for {epochs} epochs (batch={batch_size}) …\n")

    for epoch in range(epochs):
        idx = np.random.permutation(N)
        x_shuf = x_train[idx]
        y_shuf = y_train[idx]

        total_loss = 0.0
        num_batches = 0

        for i in range(0, N, batch_size):
            xb = x_shuf[i:i+batch_size]
            yb = y_shuf[i:i+batch_size]
            loss = model.train_batch(xb, yb)
            total_loss += loss
            num_batches += 1

        avg_loss = total_loss / num_batches
        losses.append(avg_loss)

        if epoch % 40 == 0 or epoch == epochs - 1:
            stars = "✨" * (min(epoch // 40 + 1, 5))
            print(f"Epoch {epoch:3d}   |   loss = {avg_loss:.6f}   {stars}")

    print("\nYayyyy! Training complete~ 🎉♡\n")

    # ── Draw a happy learning curve ───────────────────────────────
    plt.figure(figsize=(10, 5))
    plt.plot(losses, color='#ff85a2', linewidth=2.4, marker='o', markersize=3, alpha=0.9)
    plt.title("Our Network's Happy Learning Journey ♡", fontsize=15, fontweight='bold')
    plt.xlabel("Epochs (learning steps)", fontsize=12)
    plt.ylabel("Mean Squared Error (how confused we were)", fontsize=12)
    plt.grid(True, alpha=0.25, linestyle='--')
    plt.tight_layout()
    plt.show()

    # ── Quick peek at how well we did ─────────────────────────────
    print("Some example predictions after training:\n")
    test_x = np.array([
        [0.0,  0.0],
        [1.5,  1.0],
        [-2.0, 1.5],
        [2.5, -1.0]
    ])
    pred, _ = model.forward(test_x)
    true_tanh = np.tanh(np.sin(test_x[:,0] + test_x[:,1]))

    for i in range(len(test_x)):
        print(f"  x = {test_x[i]}   →   pred = {pred[i,0]:+6.4f}    "
              f"(should be ≈ {true_tanh[i,0]:+6.4f})")

    print("\nYou're all set! Feel free to:")
    print("  • change lr (try 0.005, 0.03, …)")
    print("  • make hidden bigger or smaller")
    print("  • add more epochs")
    print("  • invent your own target function")
    print("\nHappy learning & coding~ 🌸✨")
