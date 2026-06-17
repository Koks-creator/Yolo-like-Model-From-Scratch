# YOLO-like model from Scratch
This project is aboyt making YOLO-like model from scratch in order to understand it's architecture on lower level and because it's a cool project sar.

[![video](https://img.youtube.com/vi/dBpNzq9-vfI/0.jpg)](https://www.youtube.com/watch?v=dBpNzq9-vfI)

## What's YOLO
YOLO stands for You Only Look Once is object detection model with unique approach of predicting bounding boxes and class probabilities directly from full images in one evaluation which is much faster then previous two-stage detectors like RCNN or Fast-RCNN.

<a href="https://arxiv.org/abs/1506.02640">https://arxiv.org/abs/1506.02640</a>
<br>
<a href="https://docs.ultralytics.com/#faq">https://docs.ultralytics.com/#faq</a>

## First approach: 3 detections heads
In the previous experiment there was an issue with detecting person that are close or far away, it seemed like model was good at being average and had problems with edge-cases. This weakness has been improved by adding 3 detection heads:
 - 52x52 head for small object
 - 26x26 head for medium objects
 - 13x13 head for large images

### Model flow
| Stage | What does it do? | Detection? | Classification? |
|---|---|---|---|
| Backbone | Visual features | NO | NO |
| FPN | Combine scales | NO | NO |
| Head conv blocks | Refine for prediction | NOT YET | NOT YET |
| **Final 1×1 Conv** | **Output channels** | **YES** (ch 0-4) | **YES** (ch 5+) |

#### Backbone
Backbone (Darknet-lite, 5 residual stages) is task-agnostic. It does not recognise concepts such as 'here is a person' or 'here is a box'. It learns to recognise generic visual patterns at increasingly higher levels of abstraction:

- First layers: edges, gradients, textures
- Central: parts of objects (body parts, faces, clothes)
- Deep: whole objects + context (recognises ‘this is a person’ in a semantic sense)

#### FPN
Nor does it predict, semantic transfer.

This solves a specific problem: feat_52 has excellent spatial resolution (52x52 = many cells = good for small objects), BUT shallow features — it can detect textures, but doesn't recognise what a 'person' is. In contrast, feat_13 has deep semantic features (“this is a person”), but poor resolution (13x13 = few cells = coarse grain).

Top-down FPN: transfers semantic understanding from deep layers to shallow ones. feat_13 → 2x upsampling → concatenation with feat_26 → now feat_26 has its own spatial detail + “knows what a person is” from the depth. Repeats for feat_52.

Result: detection of small objects (head 52) benefits from the deep semantic understanding transferred from deep features. Without FPN, small objects would be difficult — cells would only see textures without context.

Still — FPN does not predict, it merely blends features.

#### Detection Heads
This is where detection and classification take place.

Each head consists of a two-stage mechanism:
 1. Conv blocks (refinement) — several conv_blocks with a mix of 1x1 and 3x3 kernels. They transform features in the direction required for prediction. It's a bit like “an MLP dressed up as a convolutional network” — the model learns: “OK, these FPN-enriched features are general; tweak them so that they represent boxes and classes”.

 2. Final Conv2D(out_channels, kernel=1) — this single 1x1 convolution produces all the predictions. For v3 single-class:
      ```python
      out_channels = num_boxes * 5 + num_classes = 1 * 5 + 1 = 6
      ```
Key point: 1x1 conv = 'fully-connected per-cell'. Each cell makes a decision independently based on its own features (channels). 1x1 does not mix information from neighbouring cells — it operates point-by-point, cross-channel. This is why we can say 'cell (gy, gx) is responsible' — because the decision is literally per-cell.

```
channels [0:4]  →  xc, yc, w, h      →  DETECTION (where the box is)
channel  [4]    →  confidence        →  OBJECTNESS (whether it exists at all)
channels [5:]   →  class probabilities       →  CLASSIFICATION (what it is)
```

### Why three detection heads — the multi-scale rationale

A single 13x13 grid has fundamental limitations with size variation:

```
416 / 13 = 32 pixels per cell
```

This means each cell "owns" a 32x32 px region. Consider what happens with different object sizes:
- **Small object (20x20 px in input)**: fits entirely inside one cell. The cell's receptive field is way larger than the object → most of the activations are background → weak signal.
- **Medium object (100x100 px)**: spans ~3x3 cells. Workable — one center cell can take responsibility.
- **Large object (300x300 px, foreground person)**: spans ~10x10 cells. Which cell is "responsible"? The one with the object's center. But that cell has only seen a small patch of the object — it doesn't have visibility into the whole thing.

The main thing is the cells share the same conv weights, so we want it to be good at basically everything (small, medium or big objects), so in the result we get model which is mediocre at everything, terrible at the extremes.

<br>

### How three heads fix the problems

Each scale has its own receptive field per cell:

| Scale | Stride | Cell size | Specializes in |
|-------|--------|-----------|----------------|
| 13x13 | 32 | 32x32 px | Large objects (>40% of image) |
| 26x26 | 16 | 16x16 px | Medium objects (15-40%) |
| 52x52 | 8 | 8x8 px | Small objects (<15%) |

The encoder routes each ground-truth object to **ONE scale** based on its `max(w, h)`:
- `max(w, h) < 0.15` → 52x52 head
- `0.15 ≤ max(w, h) < 0.40` → 26x26 head
- `max(w, h) ≥ 0.40` → 13x13 head

This means each head **only trains on objects in its size range**. No more weight-sharing conflicts:
- The 13x13 head's conv weights specialize in "what does a big object look like, and how to box it"
- The 52x52 head's conv weights specialize in "what does a small object look like"
- No compromise, no conflict

During inference, all 3 heads predict in parallel. Each typically lights up for objects in its target size range. The final list goes through unified NMS to dedupe (in case an object near the size threshold gets flagged on two adjacent scales).

### Why FPN (top-down feature fusion)

The heads aren't independent — they connect via Feature Pyramid Network top-down. The deepest features (13x13, after the full backbone) are upsampled and concatenated with shallower features before feeding the 26x26 head. Same trick from 26 to 52.

**Why this matters**: deep features have rich **semantic info** ("this looks like a person") but coarse spatial resolution. Shallow features have fine **spatial detail** but weaker semantics. The small-object head needs BOTH — it needs to know "is this a person" (semantics from deep) AND "where exactly are its edges" (spatial from shallow).

Without FPN, the 52x52 head would only have shallow features and would mostly learn texture-level patterns (not "person-shaped" patterns). FPN gives it semantic grounding while preserving spatial resolution.

### What is cell?
![alt text](assets/image.png)

Each "cell" is **a single point in the model's input grid**. If the model outputs a tensor of size `(13, 13, ...)`, you have 169 cells (13x13). Each cell **"owns" a portion of the image** — with a 13x13 grid on a 416x416 input, this is exactly 32x32 pixels (416/13=32).

"Owns" specifically means: **if the centre of the object falls within the area of that cell (RF not included, only 32px area in case of 13x13 head), THEN it is responsible for its prediction.** And only that cell — other cells for that object have `obj_mask=0` and learn nothing about it. Only the responsible cell has `obj_mask=1` and it is this cell that receives the gradient from the coord loss, class loss, etc.

**An important distinction — "owns" ≠ "sees":** a cell *owns* its 32x32 px patch under its responsibility, but through the convolutional network it *sees* much more. This is called the **receptive field** — the area of the input that actually influences the value of that cell. A 13x13 cell, after passing through the entire backbone, has a receptive field of several hundred pixels — practically half the image. This is why deep cells perform well with large objects: they can see the entire object even though they *own* only a small fragment of it.

### What is receptive field?
![alt text](assets/image2.png)

The receptive field is the portion of the input image that influences the value of a single cell in the activation map. Each convolutional layer aggregates information from a local window, so the deeper the layer, the larger the RF becomes. It is calculated recursively using two formulas:

```
RF_after  = RF_before + (kernel_size - 1) * jump_before
jump_after = jump_before * stride
```
Where jump is the distance in pixels of the input image between the centres of the RFs of two adjacent cells (i.e. 'at what speed' the RF moves across the image). We start with RF=1, jump=1 (one pixel affects only itself).

**Quick rules:**

- `Conv2D(kernel=3, stride=1)` → `RF += 2 * jump`, jump unchanged
- `Conv2D(kernel=3, stride=2)` → `RF += 2 * jump`, then `jump *= 2`
- `Conv2D(kernel=1, ...)` → no RF change (kernel - 1 = 0)
- Residual block (`1x1 → 3x3 stride 1`) counts effectively as one 3x3 conv (the 1x1 contributes nothing)

### Stage-by-stage calculation

| Layer | kernel | stride | jump after | RF after |
|---|---|---|---|---|
| (start) | — | — | 1 | 1 |
| Stem · conv 3x3, s=1 | 3 | 1 | 1 | **3** |
| Stage 1 · conv 3x3, s=2 | 3 | 2 | 2 | 5 |
| Stage 1 · 1 res block | 3 | 1 | 2 | **9** |
| Stage 2 · conv 3x3, s=2 | 3 | 2 | 4 | 13 |
| Stage 2 · 2 res blocks | 3 | 1 | 4 | 21 → **29** |
| Stage 3 · conv 3x3, s=2 | 3 | 2 | 8 | 37 |
| Stage 3 · 4 res blocks | 3 | 1 | 8 | 53 → 69 → 85 → **101** ← `feat_52` |
| Stage 4 · conv 3x3, s=2 | 3 | 2 | 16 | 117 |
| Stage 4 · 4 res blocks | 3 | 1 | 16 | 149 → 181 → 213 → **245** ← `feat_26` |
| Stage 5 · conv 3x3, s=2 | 3 | 2 | 32 | 277 |
| Stage 5 · 2 res blocks | 3 | 1 | 32 | 341 → **405** ← `feat_13` |

### Final results per head

| Head | Stride | Theoretical RF | Effective RF (~½ of theoretical) |
|---|---|---|---|
| `feat_52` | 8 | ~100 px | ~50 px |
| `feat_26` | 16 | ~245 px | ~125 px |
| `feat_13` | 32 | ~405 px (≈ entire 416 px image) | ~200 px |

### Why RF grows so fast in deeper stages

Every stride-2 conv **doubles the jump**, so each subsequent residual block contributes more pixels to RF than the last:

- Stage 3 res blocks: **+16 px** each (jump = 8)
- Stage 4 res blocks: **+32 px** each (jump = 16)
- Stage 5 res blocks: **+64 px** each (jump = 32)

RF grows roughly exponentially with depth, not linearly. That's why just 2 res blocks in Stage 5 take RF from 277 → 405, while 4 res blocks in Stage 3 took it from 37 → 101.

### How do we get to 52, 26, 13 from input 416x416?

![alt text](assets/image3.png)

```python
x = conv_block(inputs, w(32))           # stem — no stride, stays 416
x = conv_block(x, w(64), strides=2)     # Stage 1 input: 416 → 208 ← halving
# residual blocks (stride=1) — don't change spatial dim
x = conv_block(x, w(128), strides=2)    # Stage 2 input: 208 → 104 ← halving
x = conv_block(x, w(256), strides=2)    # Stage 3 input: 104 → 52  ← halving
feat_52 = x                              # Head for smol objects (52x52)
x = conv_block(x, w(512), strides=2)    # Stage 4 input: 52 → 26   ← halving
feat_26 = x                              # Head for medium (26x26)
x = conv_block(x, w(1024), strides=2)   # Stage 5 input: 26 → 13   ← halving
feat_13 = x                              # Head for big boys (13x13)
```

Each stage starts with conv_block with strides=2 which divides dim in half.

```
416 → 208 → 104 → 52 → 26 → 13
       ↑     ↑    ↑     ↑    ↑
       /2    /2   /2    /2   /2
```

5 divided by 2 five times = `2^5 = 32` times smaller (416/32 = 13). That is why the stride of the deepest scale is 32.

**The deeper the stage, the:**
- **Fewer cells** (more downsampling). 13x13 = 169 cells is ~16x fewer than 52x52 = 2704
- **A larger receptive field per cell.** A cell in a 13x13 grid sees a wider section of the image because many layers of convolution have accumulated
- **Richer semantics.** Deep features "understand" what an object is (a person? a car?), whilst shallow ones only see textures

That is why routing based on object size makes sense:
- **Small object** → needs a dense grid (52x52) to fit into a single cell with meaningful ownership, but its receptive field does not need to be large because the object is small
- **Large object** → its centre will easily fit within a 13x13 grid (a 32x32 px "own" cell, the centre will fit there), and a deep cell has a receptive field that sees the entire large figure

zrodla na koniec tej sekcji - czyli takie co maja headery, fpn

https://arxiv.org/abs/1612.03144
https://arxiv.org/abs/1804.02767
cos jeszcze
