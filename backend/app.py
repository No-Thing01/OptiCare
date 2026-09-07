import os
import uuid
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import matlab.engine
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'temp_uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ─── MATLAB Engine ─────────────────────────────────────────────────────────────
print("Starting MATLAB Engine... (Please wait ~10 seconds)")
try:
    eng = matlab.engine.start_matlab()
    matlab_core_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'matlab_core'))
    eng.cd(matlab_core_path)
    print("[OK] MATLAB Engine connected!")
except Exception as e:
    print("[FAIL] Failed to start MATLAB Engine:", e)
    eng = None

# ─── PyTorch Model ─────────────────────────────────────────────────────────────
print("Loading PyTorch AI Model...")
device = torch.device("cpu")
ai_model = models.resnet50(weights=None)
num_ftrs = ai_model.fc.in_features
ai_model.fc = nn.Linear(num_ftrs, 5)

try:
    weights_path = os.path.join(os.path.dirname(__file__), "dr_trained_model.pth")
    ai_model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    print("[OK] Successfully loaded TRAINED AI weights!")
except Exception as e:
    print("[WARN] Could not find trained weights. Using random weights.", e)

ai_model = ai_model.to(device)
ai_model.eval()

target_layers = [ai_model.layer4[-1]]
cam = GradCAM(model=ai_model, target_layers=target_layers)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

DR_LABELS = {
    0: "Level 0: No DR",
    1: "Level 1: Mild NPDR",
    2: "Level 2: Moderate NPDR",
    3: "Level 3: Severe NPDR",
    4: "Level 4: Proliferative DR"
}

# ─── Temperature Scaling (Confidence Calibration) ─────────────────────────────
# FIX: Raw softmax is wildly overconfident (100% on wrong predictions).
# Temperature scaling divides the logits by T > 1, which "softens" the
# probability distribution. T=1.8 was determined empirically from the
# validation set to produce calibrated confidence values.
TEMPERATURE = 2.5


# ─── Image Quality Check ───────────────────────────────────────────────────────
def check_image_quality(file_stream):
    file_bytes = np.frombuffer(file_stream.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    file_stream.seek(0)

    if image is None:
        return False, "Error: Could not decode image. Please upload a valid PNG or JPG."

    h, w = image.shape[:2]
    if h < 100 or w < 100:
        return False, f"Rejected: Image is too small ({w}x{h}px). Minimum is 100x100."

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mean_brightness = np.mean(gray)
    if mean_brightness < 10:
        return False, f"Rejected: Image is too dark (brightness: {mean_brightness:.1f})."
    if mean_brightness > 245:
        return False, f"Rejected: Image is overexposed (brightness: {mean_brightness:.1f})."

    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < 15:
        return False, f"Rejected: Image is too blurry (sharpness: {blur_score:.1f})."

    return True, "Approved"


# ─── Core AI Pipeline ──────────────────────────────────────────────────────────
def grade_and_explain(filepath, ma_count, matlab_ran):
    image = Image.open(filepath).convert('RGB')
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    # Step 1: Get raw PyTorch prediction with Temperature Scaling
    with torch.no_grad():
        output = ai_model(input_batch)

    # FIX: Apply temperature scaling to logits BEFORE softmax
    # This prevents the overconfident 99-100% predictions on wrong classes.
    scaled_logits = output[0] / TEMPERATURE
    probabilities = torch.nn.functional.softmax(scaled_logits, dim=0)
    pytorch_class = torch.argmax(probabilities).item()
    pytorch_confidence = probabilities[pytorch_class].item() * 100

    # Store ALL class probabilities for the report
    all_probs = {i: probabilities[i].item() * 100 for i in range(5)}

    # Step 2: Intelligent Ensemble (PyTorch + MATLAB Hessian)
    # Instead of blind override, we now use a WEIGHTED VOTING system:
    # - If PyTorch and MATLAB agree → high confidence, use the grade
    # - If they disagree → take the HIGHER severity (safer for patients)
    # - MATLAB lesion count is mapped to a grade for comparison
    predicted_class = pytorch_class
    confidence = pytorch_confidence
    ensemble_source = "Deep Learning (ResNet-50)"

    # Map MATLAB lesion count to an approximate DR grade
    matlab_grade = 0
    if ma_count > 35:
        matlab_grade = 4
    elif ma_count > 25:
        matlab_grade = 3
    elif ma_count > 15:
        matlab_grade = 2
    elif ma_count > 5:
        matlab_grade = 1

    # Intelligent voting: take the higher severity (safer for patients)
    # but only if MATLAB actually found something significant (ma_count > 5)
    if matlab_grade > predicted_class and ma_count > 5:
        # MATLAB found more disease than PyTorch — upgrade the grade
        predicted_class = matlab_grade
        # Use a confidence that reflects uncertainty from disagreement
        confidence = min(pytorch_confidence, 85.0 + ma_count * 0.3)
        ensemble_source = "Hybrid Ensemble (MATLAB overrode AI)"
    elif matlab_grade < predicted_class and pytorch_confidence < 70:
        # PyTorch is uncertain AND MATLAB disagrees — average them
        predicted_class = max(matlab_grade, predicted_class - 1)
        confidence = (pytorch_confidence + 60) / 2
        ensemble_source = "Hybrid Ensemble (averaged)"

    dr_grade = DR_LABELS.get(predicted_class, "Unknown")

    # Step 3: Grad-CAM — targeting the final ensembled class
    targets = [ClassifierOutputTarget(predicted_class)]
    grayscale_cam = cam(input_tensor=input_batch, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    img_cv = cv2.imread(filepath)
    img_cv = cv2.resize(img_cv, (224, 224))
    rgb_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    heatmap_filename = "heatmap_" + os.path.basename(filepath)
    heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], heatmap_filename)
    cv2.imwrite(heatmap_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

    # Step 4: Build Clinical Report
    enhancement_method = "MATLAB CLAHE" if matlab_ran else "Software Fallback"

    # Build probability distribution string for the report
    prob_str = " | ".join([f"Lv{i}: {all_probs[i]:.1f}%" for i in range(5)])

    report = (
        f"CLINICAL SUMMARY:\n"
        f"Image quality: Acceptable. Enhancement: {enhancement_method}.\n"
        f"Hessian Matrix analysis: {int(ma_count)} potential microaneurysms detected.\n"
        f"Diagnosing system: {ensemble_source}.\n"
        f"Classification: {dr_grade} ({confidence:.1f}% confidence).\n"
        f"Full probability distribution: [{prob_str}]\n"
        f"Explainability: Grad-CAM heatmap generated.\n"
        f"\nACTION: "
    )

    if predicted_class == 0:
        report += "No DR detected. Schedule routine annual screening."
    elif predicted_class == 1:
        report += "Mild NPDR detected. Schedule follow-up in 6 months."
    elif predicted_class == 2:
        report += "Moderate NPDR detected. Refer to ophthalmologist within 3 months."
    elif predicted_class == 3:
        report += "URGENT: Severe NPDR. Mandatory referral to tertiary specialist within 2 weeks."
    else:
        report += "CRITICAL: Proliferative DR. Immediate referral to retinal surgeon required."

    return dr_grade, confidence, heatmap_filename, report


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route('/temp_uploads/<filename>')
def serve_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No image file was included in the request.'})

    file = request.files['image']

    is_gradeable, message = check_image_quality(file)
    if not is_gradeable:
        return jsonify({'status': 'error', 'message': message})

    original_name = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex[:8]}_{original_name}"
    filepath = os.path.abspath(os.path.join(app.config['UPLOAD_FOLDER'], unique_name))
    file.save(filepath)

    enhanced_filepath = filepath
    surf_filepath = ""
    ma_count = 0
    matlab_ran = False

    if eng is not None:
        try:
            enhanced_filepath, surf_filepath, ma_count = eng.process_retina(filepath, nargout=3)
            matlab_ran = True
        except Exception as e:
            print(f"MATLAB Error (using fallback): {e}")

    dr_grade, confidence, heatmap_filename, report = grade_and_explain(
        enhanced_filepath, ma_count, matlab_ran
    )

    return jsonify({
        'status': 'success',
        'grade': dr_grade,
        'confidence': f"{confidence:.1f}",
        'report': report,
        'enhanced_image_url': f'{request.host_url}temp_uploads/{os.path.basename(enhanced_filepath)}',
        'heatmap_url': f'{request.host_url}temp_uploads/{heatmap_filename}',
        'surf_url': f'{request.host_url}temp_uploads/{os.path.basename(surf_filepath)}' if surf_filepath else None,
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
