# 🏄 Subway Surfers AI — CNN + Imitation Learning

An AI that plays Subway Surfers in real time using a custom CNN trained on my own gameplay. It watches the screen, predicts an action, and presses the keys — all at ~15-20 FPS.

> **Status:** CNN (supervised learning) ✅ &nbsp;|&nbsp; Reinforcement Learning 🔄 in progress

---

## How It Works

```
Screen capture → Crop game region → Grayscale + Resize (96×96) → Horizontally Flip it to expand dataset → CNN → Keypress
```

The model was trained using **imitation learning**: I played the game myself, saved labeled screenshots, and trained the CNN to mimic my actions.

---

## Project Structure

```
subway-surfers-ai/
├── dataset/
│   ├── jump/
│   ├── roll/
│   ├── left/
│   ├── right/
│   └── noop/
├── models/
│   └── subway_cnn.pth
├── train_model.py      # Training loop
├── play.py             # Screen capture + real-time inference
└── README.md
```

---

## Model Architecture

A lightweight CNN designed for low-resolution grayscale gameplay frames.

```
Input: [1 × 96 × 96]
  → Conv2d(1→32, k=8, s=4)   + ReLU      # spatial feature extraction
  → Conv2d(32→64, k=4, s=2)  + ReLU
  → Conv2d(64→64, k=3, s=1)  + ReLU
  → Flatten → Linear(4096→512) + ReLU
  → Linear(512→5)                          # 5 actions
```

**Actions:** `jump (↑)` · `roll (↓)` · `left (←)` · `right (→)` · `noop`

---

## Dataset Pipeline

| Step | Detail |
|------|--------|
| Collection | Played manually for ~30 min, captured screen + label per frame |
| Region crop | Fixed 400×500 px window centered on screen |
| Preprocessing | Grayscale → resize to 96×96 |
| Augmentation | Horizontal flip of every image (label swapped: left ↔ right) doubles dataset size |
| Split | 85% train / 15% validation |

---

## Training

```bash
python -m train_model
```

| Hyperparameter | Value |
|----------------|-------|
| Batch size | 32 |
| Epochs | 20 |
| Optimizer | Adam (lr=1e-3) |
| Loss | CrossEntropyLoss |
| Device | CPU |

Best model checkpoint saved to `models/subway_cnn.pth` based on validation accuracy.

---

## Running the Bot

```bash
python -m play
```

**Controls while running:**

| Key | Action |
|-----|--------|
| `P` | Pause / Resume |
| `Q` | Quit |

The bot only acts when model confidence is above **90%** — otherwise it does nothing (`noop`). This prevents jittery low-confidence inputs.

---

## Requirements

**Install uv and python +3.12**

```bash
uv sync
```

## Important Notes

- **Version of Subway Surfers I used**
[Subway Surfers Classic](https://g2.igroutka.ru/games/164/ZXLa594fek6p7nVR/10/subway_surfers_classic/)
- **My Screen Resolution:** 1920 × 1080  

- Make sure to enable the **browser favorites/bookmarks bar** while using or testing the model.  
  This is important because it slightly reduces the effective visible game window size.

- The model was trained specifically on this adjusted screen layout.  
  If the game is run in a different layout, resolution, or UI configuration, performance may degrade or the model may not generalize correctly.
---

## What I Learned

Building this from scratch gave me hands-on intuition for:

- **Kernels & filters** — how conv layers detect edges, shapes, and patterns
- **Activation functions** — why ReLU works and where it's placed
- **Pooling & stride** — downsampling spatial resolution efficiently
- **Dense layers** — mapping abstract features to action probabilities
- **Data augmentation** — flipping + label remapping as a free dataset doubler

---

## Roadmap

- [x] Dataset collection pipeline
- [x] CNN training with imitation learning
- [x] Real-time play loop with confidence gating
- [ ] Frame stacking (3×96×96) for motion awareness
- [ ] Reinforcement Learning (Policy Gradient) on top of pre-trained weights
- [ ] Score-based reward signal from screen OCR

---

## Demo

> Recorded on phone — running the model and screen capture simultaneously was too heavy for my CPU 😅




https://github.com/user-attachments/assets/fc41a001-06df-4953-b9a6-a2091082910f




---

## License

MIT
