# Data and Model Assets

## Sample PII Documents

The `markdown_data/` folder contains 5 synthetic Markdown documents used to evaluate the PII redaction module. They contain injected PII across 8 categories (names, emails, phone numbers, SSNs, credit cards, IP addresses, physical addresses, dates of birth).

## Video Dataset

The video evaluation dataset (76 clips, 1,540 frames) was sourced from YouTube egocentric footage and is **not included** in this repository due to size. To reproduce the evaluation:

1. Download egocentric/street-level clips from YouTube using `yt-dlp`:
   ```bash
   pip install yt-dlp
   yt-dlp -f "best[height<=480]" -o "images/%(id)s_clip%(autonumber)04d.%(ext)s" <URL>
   ```
2. Extract frames at 1 fps into `images/<clip_id>/` directories (JPEG format).

Any outdoor egocentric footage with pedestrians will produce comparable results.

## Model Weights

### EgoBlur Gen2 (face + license plate detection)

Download from Meta's official release:
- `ego_blur_face_gen2.jit` — face detection TorchScript model
- `ego_blur_lp_gen2.jit` — license plate detection TorchScript model

```bash
# Follow instructions at: https://ai.meta.com/research/publications/egoblur-responsible-innovation-in-aria/
```

Place both `.jit` files in the **project root** (same directory as `src/`).

### OpenCV Re-ID Models (included in `src/agent/tools/`)

These small models are bundled with the source code:
- `face_detection_yunet_2023mar.onnx` (227 KB) — YuNet face detector
- `face_recognition_sface_2021dec.onnx` (37 MB) — SFace re-identification model
- `deploy.prototxt` + `res10_300x300_ssd_iter_140000.caffemodel` — SSD face detector

The SFace ONNX model (37 MB) can be downloaded from the OpenCV Zoo:
```bash
wget https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx \
     -O src/agent/tools/face_recognition_sface_2021dec.onnx
```
