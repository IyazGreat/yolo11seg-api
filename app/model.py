import io
import base64
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO

# MUST match your data.yaml order:
CLASS_NAMES = ["fruit borer", "fruit fly", "rot", "sclerotium"]

MODEL_PATH = "weights/best.pt"
model = YOLO(MODEL_PATH)

def _png_base64(img_bgr):
    # img_bgr -> PNG bytes -> base64 string
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("utf-8")

def predict_image(image_bytes: bytes):
    # Read image
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = np.array(pil)  # RGB
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Run inference (segmentation)
    results = model.predict(source=img, verbose=False)
    r = results[0]

    # If nothing detected
    if r.boxes is None or len(r.boxes) == 0:
        return {
            "detected": False,
            "class_id": None,
            "class_name": None,
            "confidence": None,
            "mask_png_b64": None,
            "overlay_png_b64": None,
        }

    # Choose top detection by confidence
    confs = r.boxes.conf.cpu().numpy()
    best_i = int(np.argmax(confs))

    cls_id = int(r.boxes.cls[best_i].cpu().numpy())
    conf = float(r.boxes.conf[best_i].cpu().numpy())

    class_name = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else str(cls_id)

    # --- Make a mask image for ONLY that best detection ---
    mask_png_b64 = None
    overlay_png_b64 = None

    if r.masks is not None and r.masks.data is not None:
        # masks.data: [N, H, W] in 0/1 (float)
        masks = r.masks.data.cpu().numpy()
        best_mask = masks[best_i]

        # Create binary mask (white foreground)
        mask_u8 = (best_mask * 255).astype(np.uint8)  # HxW
        mask_bgr = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)

        mask_png_b64 = _png_base64(mask_bgr)

        # Optional overlay (input + mask)
        overlay = img_bgr.copy()
        overlay[mask_u8 > 0] = (0.5 * overlay[mask_u8 > 0] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
        overlay_png_b64 = _png_base64(overlay)

    return {
        "detected": True,
        "class_id": cls_id,
        "class_name": class_name,
        "confidence": conf,
        "mask_png_b64": mask_png_b64,
        "overlay_png_b64": overlay_png_b64,
    }