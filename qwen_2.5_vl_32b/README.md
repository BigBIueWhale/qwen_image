# Qwen 2.5 VL 32B

Run on `Ollama 0.11.7`

```sh
ollama pull qwen2.5vl:32b-q4_K_M
```

> Note: It occurs to me that https://github.com/zai-org/GLM-V is **much better** and more accurate at creating bounding boxes.
> In other words- `qwen2.5vl:32b` is actually unusable, relatively.

## Build Prompt

- Image to attach: Place image at [./input_image.png](./input_image.png), and run `python3 image_normalize.py`.

- New chat in OpenWebUI and attach the created [./output_image.png](./output_image.png), and use the following prompt:

```txt
Describe bounding boxes around every shwarma wheel
Output JSON only:
[
  { "label": "...", "bbox_2d": [x1, y1, x2, y2] },
  ...
]
```

- Take the resulting json and place it in [./result.json](./result.json)

- Run `python3 draw_boxes.py`. The script will take [./result.json](./result.json) and apply them to [./output_image.png](./output_image.png) to create a new image: [bounding_boxes.png](./bounding_boxes.png).


## Metaparameters for OpenWebUI

here’s a set of **evidence-based presets for `qwen2.5vl:32b` on Ollama**, pulled from Unsloth write-ups, official Qwen guidance, Reddit field reports, and a few YouTube demos.

### TL;DR presets (drop into Ollama `options`)

**General chat / vision reasoning**

**NOTE: This is what I used in my config.**

```json
{ "temperature": 0.6, "top_p": 0.9, "top_k": 20, "repeat_penalty": 1.05, "num_ctx": 18432, "num_predict": -1 }
```

Why: Qwen’s own “official” guidance across recent releases clusters around **temp ≈ 0.6–0.7, top\_p 0.8–0.95, top\_k \~20**, with **repetition\_penalty \~1.05**. Unsloth’s Qwen pages mirror the same advice. ([qwen.readthedocs.io][1])

**Coding / precise reasoning**

```json
{ "temperature": 0.6, "top_p": 0.8, "top_k": 20, "repeat_penalty": 1.05, "num_ctx": 18432, "num_predict": -1 }
```

Why: Qwen’s coder “Best Practices” explicitly recommend **0.7 / 0.8 / 20 / 1.05**; a slightly cooler `temp` (0.6) is a common tweak for fewer tangents. ([Hugging Face][2])

**OCR / structured extraction (JSON, tables)**

```json
{ "temperature": 0, "top_k": 1, "top_p": 1.0, "repeat_penalty": 1.05, "seed": 7, "num_ctx": 18432, "num_predict": 800 }
```

Why: For OCR you want **determinism**. Community tests report better consistency with **greedy-ish decoding** (temp=0; `top_k`=1). Keep a small repeat penalty and set a seed. ([Reddit][3])

**Creative writing / open-ended**

```json
{ "temperature": 0.8, "top_p": 0.95, "top_k": 40, "repeat_penalty": 1.05, "num_ctx": 18432, "num_predict": 800 }
```

Why: Qwen’s guidance allows **top\_p up to \~0.95** and **top\_k 20–40** when you want more variety. ([qwen.readthedocs.io][1])

---

### Why these (and what to ignore)

* The **model’s baked config** sets an almost-zero temperature (1e-6) to behave *nearly greedy* out-of-the-box; that’s great for stability but lifeless for chat. Raise temp for anything creative. ([Hugging Face][4])
* Qwen’s **official recs** (Qwen3/QwQ docs & cards) consistently land on **temp \~0.6–0.7, top\_p 0.8–0.95, top\_k ≈20**, and **rep\_penalty ≈1.0–1.05**. Those numbers transfer well to Qwen2.5-VL too. ([qwen.readthedocs.io][1])
* For **vision**, the Qwen team’s docs emphasize *image resizing budgets* (`min_pixels`, `max_pixels`) rather than sampler fiddling. Ollama doesn’t expose those knobs, so if you hit memory/latency walls with huge frames, **downscale images before sending** (e.g., long edge ≈ 1280px) to stay within Qwen’s typical max-pixels window. ([GitHub][5])

---

### Anti-loop & de-hallucination add-ons (optional)

If you see repetition or “spirals,” add one of these:

* **Presence penalty**: `presence_penalty: 0.5–1.5` (supported via API options) – Qwen’s docs mention using **0–2** to reduce repetition, with trade-offs. ([qwen.readthedocs.io][1])
* **Lower temp + lower top\_p** (e.g., `temperature: 0.4, top_p: 0.85`) for stricter outputs—several community posts report this helps for coding and long tasks. ([Reddit][6])

---

