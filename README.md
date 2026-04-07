# CivicLens


### Pitch

Turning dashcam footage into real-time, geo-tagged road maintenance reports with AI.


### Problem statement
Urban road infrastructure issues often go unreported or are detected too late, leading to accidents, vehicle damage, and inefficient maintenance workflows.

RoadScan AI addresses this by using AI-powered computer vision to automatically detect road hazards from dashcam or smartphone footage, geo-tag incidents, and generate actionable maintenance tickets in real time — enabling faster response and smarter city management.

## Project Summary
End-to-end civic infrastructure reporting system. Citizen uploads a photo →
EfficientNetV2-S classifies the road defect → confidence gate decides routing →
OpenAI enriches confirmed damage into a structured complaint →
auto-filed to dashboard with department routing.

### Final classes
```
1. potholes
2. cracked_pavement  
3. debris_obstruction
4. faded_lane_markings
5. broken_road_signs
6. normal_road
```
---

## Dataset
- **7,200 images**, 6 classes, balanced to ~1,200/class
- **Classes:** potholes, cracked_pavement, road_debris_obstruction,
  broken_road_signs, faded_lane_markings, normal_road
- **Split:** 80/10/10 (train/val/test) using stratified sklearn split, SEED=42
- **Source:** Roboflow Universe (multiple merged datasets), <100 synthetic
  images per class from Bing search + manual annotation
- **Cleaning:** corrupt headers, undersized images (<64x64) removed
- **Balancing:** undersample if n > 1200, oversample with replacement if n < 1200
- ⚠️ Oversampled classes (e.g. normal_road had only 130 raw images) —
  duplicate source images may appear across splits (minor caveat)

---

## Model
- **Architecture:** EfficientNetV2-S (fine-tuned, frozen backbone initially,
  then unfrozen from block 6+)
- **Trainable params:** ~329K (head only in phase 1)
- **Training:** CPU only (slow), 20 epochs planned, best model saved on val_acc
- **Val accuracy:** ~88.6% (epoch 1-2)
- **Test accuracy:** 96.8% overall on ~800 test images
- **Loss curves:** val loss slightly higher than train loss after epoch 2 —
  mild overfitting but not severe

---

## Per-Class Test Accuracy
| Class | Accuracy |
|---|---|
| potholes | 92% |
| cracked_pavement | 96% |
| road_debris_obstruction | 98% |
| broken_road_signs | 100% |
| faded_lane_markings | 96% |
| normal_road | 99% |

---

## Inference Pipeline
- **conf < 0.60** → logged as Clear Road, OpenAI skipped
- **0.60–0.70** → flagged for manual review
- **conf ≥ 0.70** → image + class sent to OpenAI for enrichment

## OpenAI Enrichment Output
Title, Description, Severity, Likely Cause, Est. Fix Time, Department

---

## Resume Bullet
"Built an end-to-end civic reporting pipeline — fine-tuned EfficientNetV2-S
on a custom 7,200-image dataset (96.8% test accuracy, 6 road defect classes)
with confidence-gated OpenAI enrichment for automated complaint generation
and department routing." — MadData 2026 Hackathon

---

## Limitations
- Oversampling used for minority classes — possible train/test overlap for
  those classes (normal_road, road_debris_obstruction)
- Trained on CPU — limited epochs
- Val loss slightly unstable after epoch 2 — would benefit from LR scheduling
  and more epochs on GPU


