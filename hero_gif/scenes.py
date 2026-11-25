# Copyright (C) 2026 * Ltd. All rights reserved.
# author: Sanghyun Jo <shjo.april@gmail.com>

"""Scene data of the hero GIF: every label and number that build_gif.py draws, with its source in the paper."""

SCENES = {
 "title": "TTD",
 "badge": "ECCV 2024",
 "subtitle": "Text-Tag Self-Distillation Enhancing Image-Text Alignment in CLIP to Alleviate Single Tag Bias",
 "tabs": ["1 · Single Tag Bias", "2 · Tag Selection", "3 · Self-Distillation", "4 · Results"],
 "logos": {
  "left": [["ogq.png", 32], ["kaist_ai.png", 40]],
  "right": [["aitrics.png", 26], ["mgh.png", 42], ["hms.png", 42]]
 },
 "bias": {
  "text": "image of a green chameleon on a branch",
  "panels": [
   ["panels/chameleon_image.jpg", "Image"],
   ["panels/chameleon_without_ttd.jpg", "CLIP Variants w/o TTD"],
   ["panels/chameleon_with_ttd.jpg", "w/ TTD (Ours)"]
  ],
  "box": [0.230, 0.026, 0.988, 0.752],
  "charts": ["panels/chameleon_scores_without_ttd.png", "panels/chameleon_scores_with_ttd.png"],
  "note": "Paper Fig. 1 · top: similarity map between the pixels of the image and the text · bottom: similarity between the image and each tag · TCL is used for the analysis"
 },
 "selection": {
  "image": "panels/pillow_image.jpg",
  "text": "animal demands separate pillow for paw",
  "words": ["animal", "demands", "separate", "pillow", "for", "paw"],
  "sorted": [["pillow", 1.00], ["animal", 0.71], ["paw", 0.63], ["separate", 0.15], ["demands", 0.13], ["for", 0.12]],
  "gaps": [0.29, 0.08, 0.48, 0.02, 0.01],
  "selected": 3,
  "note": "Paper Fig. 4(a) and Eq. (2) · heatmaps of one image-text pair, sorted by the maximum similarity score per heatmap"
 },
 "distillation": {
  "image": "panels/jacuzzi_image.jpg",
  "text": "woman spends time in a jacuzzi",
  "clip": "panels/jacuzzi_clip_heatmap.jpg",
  "union": "panels/jacuzzi_union.jpg",
  "tags": [["jacuzzi", 1.00], ["woman", 0.82], ["in", 0.22], ["a", 0.18], ["spends", 0.03], ["time", 0.00]],
  "selected": 2,
  "steps": [
   ["1 · Image-text map", "It should reflect every relevant tag of the text, but it often highlights only one."],
   ["2 · Pseudo-tags", "Pixel-tag scoring keeps the tags that are in the image: jacuzzi and woman."],
   ["3 · Ideal image-text map", "The union of the pseudo-tag heatmaps: the pixel-wise maximum of the min-max normalized maps."],
   ["4 · Self-distillation", "The distillation loss aligns the image-text map with the union. The tag loss drives the maps of non-selected tags toward zero."]
  ],
  "note": "Paper Fig. 2 and Eqs. (4) to (6) · tag heatmaps and pixel-tag scores from Fig. 2(a), CLIP heatmap and union of heatmaps from Fig. 2(b)"
 },
 "results": {
  "columns": [["image", "Image"], ["gt", "Ground Truth"], ["maskclip", "MaskCLIP"], ["maskclip_ours", "MaskCLIP+Ours"], ["tcl", "TCL"], ["tcl_ours", "TCL+Ours"]],
  "text": {
   "head": "Text-level segmentation: which region of the image does the text describe?",
   "rows": [
    ["stop_sign", "red stop sign on a stone road", 132],
    ["sheepdog", "sheepdog gripping the wool on a sheep", 142],
    ["cargo_ship", "dry cargo ship in the sea", 131]
   ],
   "cards": [
    ["82.8", "F1 of multi-tag selection", "pixel-tag scoring with TCL + TTD · NLTK: 69.8"],
    ["41.0 → 50.2", "CaptionIoU (+9.2)", "MaskCLIP + TTD"],
    ["60.4 → 65.5", "CaptionIoU (+5.1)", "TCL + TTD"]
   ],
   "line": "With TTD, the masks cover the regions of all image-related tags in the text",
   "note": "Paper Fig. 6, Table 2(a) and Table 3 · text-level semantic segmentation: the text encoder processes the whole text"
  },
  "tags": {
   "head": "Open-vocabulary segmentation with tag inputs on six benchmarks",
   "rows": [
    ["voc", "PASCAL VOC", 148],
    ["context", "PASCAL Context", 148],
    ["stuff", "COCO-Stuff", 148]
   ],
   "cards": [
    ["19.0 → 27.5", "Average mIoU (+8.5)", "MaskCLIP + TTD · six benchmarks"],
    ["30.4 → 33.9", "Average mIoU (+3.5)", "TCL + TTD · six benchmarks"],
    ["61.1 · 37.4 · 37.4", "mIoU of TCL + TTD", "VOC · Context · COCO-Object"]
   ],
   "line": "Trained with image-text pairs only · no additional annotations or external models",
   "note": "Paper Fig. 7 and Table 4 · open-vocabulary semantic segmentation, ViT-B/16 backbone"
  }
 }
}
