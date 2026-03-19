import os
import cv2
import uuid
from pathlib import Path
from flask import Blueprint, request, jsonify, send_from_directory

from .abnv2_single import ABNCropper
from .classifier import Classifier

# ================= CONFIG =================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
IMAGES_DIR = BASE_DIR.parent / "images"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

ABN_MODEL_PATH = MODEL_DIR / "imxv11-2300.pt"
BINARY_MODEL_PATH = MODEL_DIR / "Binary.pt"
# CLS_MODEL_PATH = MODEL_DIR / "den.pth"

BINARY_CLASS_NAMES = ["no-error", "error"]
# CLASS_NAMES = [
#     "bridging", "layer-shift", "MISCC", "no-error",
#     "over extrusion", "spaggeti", "stringing",
#     "under-extrusion", "warping"
# ]
CLS_MODEL_PATH = MODEL_DIR / "best_efficientnet_b0.pth"
# CLS_MODEL_PATH = MODEL_DIR / "augmentbest.pt"
CLASS_MAPPING_PATH = MODEL_DIR / "class_mapping.json"
# =========================================

pipeline_api = Blueprint("pipeline_api", __name__)

# Load models ONCE
abn = ABNCropper(str(ABN_MODEL_PATH))
# binary_classifier = Classifier(str(BINARY_MODEL_PATH), BINARY_CLASS_NAMES)
# classifier = Classifier(str(CLS_MODEL_PATH), CLASS_NAMES)
classifier = Classifier(
    model_path=str(CLS_MODEL_PATH),
    class_mapping_path=str(CLASS_MAPPING_PATH)
)

@pipeline_api.route("/process-image", methods=["POST"])
def process_image_api():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save uploaded image
    image_name = f"{uuid.uuid4().hex}.jpg"
    image_path = UPLOAD_DIR / image_name
    file.save(image_path)

    img = cv2.imread(str(image_path))
    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    # -------- ABN CROP --------
    crop = abn.crop(str(image_path))

    if crop is None:
        return jsonify({
            "image": file.filename,
            "abn_detected": False,
            # "binary_prediction": None,
            "error_type": None,
            "crop_url": None
        })

    # Save crop
    crop_name = f"crop_{uuid.uuid4().hex}.jpg"
    crop_path = UPLOAD_DIR / crop_name
    cv2.imwrite(str(crop_path), crop)

    # -------- Binary Classification --------
    # binary_label = binary_classifier.predict_from_array(crop)
    # binary_norm = binary_label.lower().strip()

    # error_type = None
    # if binary_norm in ("error", "anomaly", "anomal"):
    # error_type = classifier.predict_from_array(crop)
    pred = classifier.predict_from_array(crop)

    error_type = pred["label"]
    confidence = pred["confidence"]
    all_probs = pred["all_probabilities"]


    return jsonify({
        "image": file.filename,
        "abn_detected": True,
        # "binary_prediction": binary_label,
        "error_type": error_type,
        "confidence": confidence,
        "all_probabilities": all_probs,
        "crop_url": f"/static/uploads/{crop_name}"
    })


@pipeline_api.route("/clear-uploads", methods=["POST"])
def clear_uploads():
    try:
        if UPLOAD_DIR.exists():
            for file in os.listdir(UPLOAD_DIR):
                file_path = UPLOAD_DIR / file
                if file_path.is_file() and file_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    file_path.unlink()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@pipeline_api.route("/images-list", methods=["GET"])
def get_images_list():
    if not IMAGES_DIR.exists():
        return jsonify([])
    valid_exts = {".jpg", ".jpeg", ".png"}
    files = [f for f in os.listdir(IMAGES_DIR) if Path(f).suffix.lower() in valid_exts]
    return jsonify(sorted(files))

@pipeline_api.route("/get-image/<filename>", methods=["GET"])
def get_image(filename):
    return send_from_directory(IMAGES_DIR, filename)

@pipeline_api.route("/process-server-image", methods=["POST"])
def process_server_image():
    data = request.json
    if not data or "filename" not in data:
        return jsonify({"error": "No filename provided"}), 400
        
    filename = data["filename"]
    image_path = IMAGES_DIR / filename
    
    if not image_path.exists():
        return jsonify({"error": "File not found"}), 404

    img = cv2.imread(str(image_path))
    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    # -------- ABN CROP --------
    crop = abn.crop(str(image_path))

    if crop is None:
        return jsonify({
            "image": filename,
            "abn_detected": False,
            "error_type": None,
            "crop_url": None
        })

    # Save crop
    crop_name = f"crop_{uuid.uuid4().hex}.jpg"
    crop_path = UPLOAD_DIR / crop_name
    cv2.imwrite(str(crop_path), crop)

    # -------- Classification --------
    pred = classifier.predict_from_array(crop)

    error_type = pred["label"]
    confidence = pred["confidence"]
    all_probs = pred["all_probabilities"]

    return jsonify({
        "image": filename,
        "abn_detected": True,
        "error_type": error_type,
        "confidence": confidence,
        "all_probabilities": all_probs,
        "crop_url": f"/static/uploads/{crop_name}"
    })
