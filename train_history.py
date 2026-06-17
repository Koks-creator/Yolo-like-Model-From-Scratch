import pandas as pd
from matplotlib import pyplot as plt

h = pd.read_csv(r"C:\Users\table\PycharmProjects\MojeCos2\objekt_detekszyn\models\model_mosaic2\training_tuned_log.csv")

plt.figure(figsize=(14, 5))
plt.suptitle("3 Heads Model with CIoU tuned", fontsize=16)
plt.subplot(1, 2, 1)
plt.plot(h["loss"], label="train", linewidth=2)
plt.plot(h["val_loss"], label="val", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Total loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.title("Total loss")

plt.subplot(1, 2, 2)
for key in h:
    if key.startswith("output_") and key.endswith("_loss") and "val" not in key:
        plt.plot(h[key], label=key, linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Per-scale loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.title("Loss per scale")

plt.tight_layout()
plt.show()