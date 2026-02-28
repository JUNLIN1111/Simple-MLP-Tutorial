

import numpy as np
import matplotlib.pyplot as plt


class KawaiiMLP:
    """✨ Our tiny cute baby neural network puppy ♡"""

    def __init__(self, in_dim=6, hidden=32, out_dim=4, lr=0.005):
        """
        Hello baby! Let's make a cute little brain together~ ♡
        
        Parameters:
            in_dim    → how many things baby can see (default 6)
            hidden    → how many thinking neurons in each hidden hug layer
            out_dim   → how many wiggly actions baby can do
            lr        → learning speed (small = gentle learning)
        """
        self.lr = lr
        
        # Baby's connections (weights) and little biases
        self.w1 = (np.random.randn(in_dim, hidden) * 0.12).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        
        self.w2 = (np.random.randn(hidden, hidden) * 0.12).astype(np.float32)
        self.b2 = np.zeros(hidden, dtype=np.float32)
        
        self.w3 = (np.random.randn(hidden, out_dim) * 0.08).astype(np.float32)
        self.b3 = np.zeros(out_dim, dtype=np.float32)

    def forward(self, x):
        """Baby thinks really hard and gives wiggly answer! ♡"""
        h1 = np.tanh(x @ self.w1 + self.b1)          # first cozy hug layer
        h2 = np.tanh(h1 @ self.w2 + self.b2)         # second cozy hug layer
        y  = np.tanh(h2 @ self.w3 + self.b3)         # final cute action
        
        return y, (x, h1, h2, y)

    def train_batch(self, x_batch, y_batch_norm):
        """
        Baby practices and gets smarter with every try~!
        Returns how confused baby was (loss) 💦
        """
        y_pred, (x, h1, h2, y_out) = self.forward(x_batch)
        
        diff = y_pred - y_batch_norm
        loss = float(np.mean(diff * diff))           # how much baby missed
        
        n = x.shape[0]
        dy = (2.0 / n) * diff
        
        # Backpropagation = giving gentle correction hugs ♡
        dz3 = dy * (1.0 - y_out * y_out)
        gw3 = h2.T @ dz3
        gb3 = np.sum(dz3, axis=0)
        
        dh2 = dz3 @ self.w3.T
        dz2 = dh2 * (1.0 - h2 * h2)
        gw2 = h1.T @ dz2
        gb2 = np.sum(dz2, axis=0)
        
        dh1 = dz2 @ self.w2.T
        dz1 = dh1 * (1.0 - h1 * h1)
        gw1 = x.T @ dz1
        gb1 = np.sum(dz1, axis=0)
        
        # Update baby's brain with tiny happy steps
        self.w3 -= self.lr * gw3
        self.b3 -= self.lr * gb3
        self.w2 -= self.lr * gw2
        self.b2 -= self.lr * gb2
        self.w1 -= self.lr * gw1
        self.b1 -= self.lr * gb1
        
        return loss

    def __repr__(self):
        return f"✨ KawaiiMLP ✨ (input={self.w1.shape[0]}, hidden={self.w1.shape[1]}, output={self.b3.shape[0]}) ♡ lr={self.lr}"


# ────────────────────────────────────────────────────────────────
#  Cute Demo: Baby Robot Learning to Reach the Shiny Apple 🍎🐾
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("✨ Starting Baby Robot's Apple Adventure! 🍎💕✨\n")

    # Create our cute little robot brain
    baby_robot = KawaiiMLP(in_dim=6, hidden=32, out_dim=4, lr=0.005)
    print(baby_robot, "\n")

    # Make pretend playground data
    N = 3000
    x_train = np.random.uniform(-1, 1, (N, 6))              # [curr_x,y, targ_x,y, plan_dx,dy]
    dist = x_train[:, 2:4] - x_train[:, 0:2]                # apple - me!
    y_train = np.tanh(np.hstack([dist, dist * 0.6]))        # good wiggles

    losses = []
    epochs = 250
    batch_size = 64

    print("Baby is practicing really hard... ฅ^•ﻌ•^ฅ\n")

    for epoch in range(epochs):
        # Shuffle toys so baby doesn't memorize order
        idx = np.random.permutation(N)
        x_shuf = x_train[idx]
        y_shuf = y_train[idx]

        total_loss = 0
        num_batches = N // batch_size

        for i in range(0, N, batch_size):
            loss = baby_robot.train_batch(
                x_shuf[i:i+batch_size],
                y_shuf[i:i+batch_size]
            )
            total_loss += loss

        avg_loss = total_loss / num_batches
        losses.append(avg_loss)

        if epoch % 25 == 0:
            hearts = "💖" * (epoch // 25 + 1)
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.5f}  {hearts}  (getting closer to apple!)")

    print("\nYayyyy! Baby finished training! 🎉🍎\n")

    # Show happy learning curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses, color='#ff99cc', linewidth=3, marker='o', markersize=4)
    plt.title("Baby Robot's Learning Adventure 🍎💕", fontsize=16, fontweight='bold')
    plt.xlabel("Hugs & Epochs", fontsize=12)
    plt.ylabel("How Confused I Was ♡ (Loss)", fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()

    # Final cute test!
    print("Let's see if baby can reach this apple~!")
    test_input = np.array([[0.0, 0.0,  0.6, 0.8,  0.12, 0.15]])
    actions, _ = baby_robot.forward(test_input)

    print("\nBaby robot wiggles:")
    print("   Joint 1 :", f"{actions[0,0]:+.3f}")
    print("   Joint 2 :", f"{actions[0,1]:+.3f}")
    print("   Joint 3 :", f"{actions[0,2]:+.3f}")
    print("   Joint 4 :", f"{actions[0,3]:+.3f}")
    print("\nGood job baby! ฅ^•ﻌ•^ฅ 🍎✨")

    print("\nYou can now play with baby_robot.forward() or train with your own data!")
    print("Have fun~ ♡")
