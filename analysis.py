import pandas as pd
import matplotlib.pyplot as plt

# Load both datasets
rope = pd.read_csv("rope.csv")
rpb = pd.read_csv("rpb.csv")

# Plot independent: RoPE
plt.figure(figsize=(8,5))
plt.plot(rope["Epoch"], rope["Train_Loss"], label="RoPE Train", marker="o")
plt.plot(rope["Epoch"], rope["Val_Loss"], label="RoPE Val", marker="s")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("RoPE Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("rope_loss.png", dpi=300, bbox_inches="tight")
plt.close()

# Plot independent: RPB
plt.figure(figsize=(8,5))
plt.plot(rpb["Epoch"], rpb["Train_Loss"], label="RPB Train", marker="o")
plt.plot(rpb["Epoch"], rpb["Val_Loss"], label="RPB Val", marker="s")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("RPB Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("rpb_loss.png", dpi=300, bbox_inches="tight")
plt.close()

# Overlay comparison (Validation loss only)
plt.figure(figsize=(8,5))
plt.plot(rope["Epoch"], rope["Val_Loss"], label="RoPE Val", marker="s")
plt.plot(rpb["Epoch"], rpb["Val_Loss"], label="RPB Val", marker="d")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("RoPE vs RPB (Validation Loss)")
plt.legend()
plt.grid(True)
plt.savefig("rope_vs_rpb_val.png", dpi=300, bbox_inches="tight")
plt.close()
