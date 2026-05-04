"""
═══════════════════════════════════════════════════════════════
  BIOMETRIC IDENTITY VERIFICATION SYSTEM
  Real-time face recognition using OpenCV LBPH
═══════════════════════════════════════════════════════════════
  Usage:
    python face_recognition_system.py --mode enroll --name "John Doe"
    python face_recognition_system.py --mode recognize
    python face_recognition_system.py --mode list
    python face_recognition_system.py --mode delete --name "John Doe"
═══════════════════════════════════════════════════════════════
"""

import cv2
import os
import numpy as np
import argparse
import json
import time
import shutil
from datetime import datetime

# ───────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────
DATASET_PATH       = "dataset"
MODEL_PATH         = "model/face_model.yml"
METADATA_PATH      = "model/metadata.json"
LOG_PATH           = "logs/access_log.json"
CONFIDENCE_THRESH  = 70       # Lower = stricter match
ENROLL_SAMPLES     = 50       # Frames captured during enrollment
FRAME_SCALE        = 1.3
FRAME_NEIGHBORS    = 5
FACE_RESIZE        = (200, 200)


# ───────────────────────────────────────────────
# UTILITIES
# ───────────────────────────────────────────────
def ensure_dirs():
    for d in [DATASET_PATH, "model", "logs"]:
        os.makedirs(d, exist_ok=True)

def load_metadata():
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            return json.load(f)
    return {"name_map": {}, "enrolled_count": 0}

def save_metadata(meta):
    with open(METADATA_PATH, "w") as f:
        json.dump(meta, f, indent=2)

def log_event(name: str, confidence: float, authorized: bool):
    ensure_dirs()
    log = []
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r") as f:
            try:
                log = json.load(f)
            except json.JSONDecodeError:
                log = []
    log.append({
        "timestamp": datetime.now().isoformat(),
        "name": name,
        "confidence": round(confidence, 2),
        "authorized": authorized
    })
    # Keep last 500 entries
    log = log[-500:]
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


# ───────────────────────────────────────────────
# FACE DETECTOR
# ───────────────────────────────────────────────
def get_face_cascade():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError("❌ Could not load Haar cascade. Check OpenCV installation.")
    return cascade


# ───────────────────────────────────────────────
# TRAINING
# ───────────────────────────────────────────────
def train_model():
    """
    Scan dataset/, extract faces, train LBPH recognizer, save model.
    Returns (recognizer, name_map) or raises if no data found.
    """
    ensure_dirs()
    face_cascade = get_face_cascade()
    faces, labels = [], []
    name_map = {}
    current_id = 0

    if not os.path.exists(DATASET_PATH) or not os.listdir(DATASET_PATH):
        raise RuntimeError("❌ Dataset is empty. Enroll at least one person first.")

    for person_name in sorted(os.listdir(DATASET_PATH)):
        person_folder = os.path.join(DATASET_PATH, person_name)
        if not os.path.isdir(person_folder):
            continue

        name_map[str(current_id)] = person_name
        face_count = 0

        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            detected = face_cascade.detectMultiScale(img, FRAME_SCALE, FRAME_NEIGHBORS)
            for (x, y, w, h) in detected:
                face = img[y:y+h, x:x+w]
                face = cv2.resize(face, FACE_RESIZE)
                faces.append(face)
                labels.append(current_id)
                face_count += 1

        if face_count > 0:
            print(f"  ✔ {person_name}: {face_count} face samples loaded")
            current_id += 1
        else:
            print(f"  ⚠ {person_name}: No valid face images found, skipping")
            del name_map[str(current_id)]

    if not faces:
        raise RuntimeError("❌ No valid face data found in dataset.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_PATH)

    # Update metadata
    meta = {"name_map": name_map, "enrolled_count": len(name_map)}
    save_metadata(meta)

    print(f"\n✅ Model trained with {len(faces)} samples across {len(name_map)} person(s).")
    return recognizer, name_map


def load_model():
    """Load pre-trained model. Falls back to retraining if model is missing."""
    if not os.path.exists(MODEL_PATH):
        print("⚙ No saved model found. Training from dataset...")
        return train_model()

    meta = load_metadata()
    name_map = meta.get("name_map", {})

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    return recognizer, name_map


# ───────────────────────────────────────────────
# ENROLL
# ───────────────────────────────────────────────
def enroll_person(name: str):
    """
    Opens webcam, captures ENROLL_SAMPLES face images, saves to dataset/,
    then retrains the model.
    """
    ensure_dirs()
    if not name.strip():
        print("❌ Name cannot be empty.")
        return

    person_folder = os.path.join(DATASET_PATH, name.strip())
    os.makedirs(person_folder, exist_ok=True)

    face_cascade = get_face_cascade()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot access webcam.")
        return

    print(f"\n📸 Enrolling: {name}")
    print(f"   Look at the camera. Capturing {ENROLL_SAMPLES} samples...")
    print("   Press 'q' to cancel.\n")

    count = 0
    existing = len(os.listdir(person_folder))

    while count < ENROLL_SAMPLES:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_detected = face_cascade.detectMultiScale(gray, FRAME_SCALE, FRAME_NEIGHBORS)

        display = frame.copy()

        for (x, y, w, h) in faces_detected:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, FACE_RESIZE)

            # Apply slight augmentation for robustness
            img_path = os.path.join(person_folder, f"{existing + count:04d}.jpg")
            cv2.imwrite(img_path, face_resized)
            count += 1

            # Draw progress
            progress = count / ENROLL_SAMPLES
            bar_w = int(progress * 300)
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 100), 2)
            cv2.rectangle(display, (x, y-35), (x + bar_w, y-15), (0, 255, 100), -1)
            cv2.putText(display, f"Capturing: {count}/{ENROLL_SAMPLES}",
                        (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100), 2)

        # Header overlay
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (display.shape[1], 50), (10, 10, 30), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
        cv2.putText(display, f"ENROLLING: {name.upper()}  [{count}/{ENROLL_SAMPLES}]",
                    (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)

        cv2.imshow("SecureID — Enrollment", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("⚠ Enrollment cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n✅ Captured {count} samples for '{name}'.")
    print("⚙ Retraining model...\n")
    train_model()


# ───────────────────────────────────────────────
# RECOGNIZE
# ───────────────────────────────────────────────
def recognize():
    """
    Real-time face recognition with professional HUD overlay.
    """
    recognizer, name_map = load_model()
    face_cascade = get_face_cascade()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot access webcam.")
        return

    print("\n🔍 Recognition started. Press 'q' to quit.\n")

    fps_counter = 0
    fps_start = time.time()
    fps = 0.0
    last_log_time = {}  # Throttle logging per person

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # FPS calc
        fps_counter += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps = fps_counter / elapsed
            fps_counter = 0
            fps_start = time.time()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_detected = face_cascade.detectMultiScale(gray, FRAME_SCALE, FRAME_NEIGHBORS)

        for (x, y, w, h) in faces_detected:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, FACE_RESIZE)

            label, confidence = recognizer.predict(face_resized)
            authorized = confidence < CONFIDENCE_THRESH

            if authorized:
                name = name_map.get(str(label), "Unknown")
                match_pct = max(0, int(100 - confidence))
                text = f"{name}  [{match_pct}%]"
                box_color = (0, 220, 100)
                status_text = "AUTHORIZED"
            else:
                name = "Unknown"
                match_pct = 0
                text = "UNKNOWN"
                box_color = (0, 60, 220)
                status_text = "ACCESS DENIED"

            # Log event (throttled to 5s per person)
            now = time.time()
            if now - last_log_time.get(name, 0) > 5:
                log_event(name, confidence, authorized)
                last_log_time[name] = now

            # Draw face box
            cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)

            # Corner accents
            corner_len = 15
            thickness = 3
            for cx, cy, dx, dy in [(x, y, 1, 1), (x+w, y, -1, 1),
                                    (x, y+h, 1, -1), (x+w, y+h, -1, -1)]:
                cv2.line(frame, (cx, cy), (cx + dx*corner_len, cy), box_color, thickness)
                cv2.line(frame, (cx, cy), (cx, cy + dy*corner_len), box_color, thickness)

            # Label pill background
            label_y = max(y - 12, 20)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(frame, (x, label_y - th - 6), (x + tw + 10, label_y + 4), box_color, -1)
            cv2.putText(frame, text, (x + 5, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

            # Confidence bar under face
            bar_max = w
            bar_fill = int((match_pct / 100) * bar_max) if authorized else 0
            cv2.rectangle(frame, (x, y+h+4), (x+bar_max, y+h+10), (40, 40, 40), -1)
            cv2.rectangle(frame, (x, y+h+4), (x+bar_fill, y+h+10), box_color, -1)

        # ── HUD Overlay ──────────────────────────────────
        h_frame, w_frame = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w_frame, 52), (5, 8, 20), -1)
        cv2.rectangle(overlay, (0, h_frame - 36), (w_frame, h_frame), (5, 8, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Title
        cv2.putText(frame, "SECURE ID  //  BIOMETRIC VERIFICATION",
                    (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 1)

        # FPS + timestamp
        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(frame, f"FPS: {fps:.1f}   {ts}",
                    (12, h_frame - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 140), 1)

        # Enrolled count
        enrolled = len(name_map)
        cv2.putText(frame, f"DB: {enrolled} enrolled  |  Threshold: {CONFIDENCE_THRESH}",
                    (w_frame - 330, h_frame - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 140), 1)

        cv2.imshow("SecureID — Biometric Verification System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 Session ended.")


# ───────────────────────────────────────────────
# LIST ENROLLED
# ───────────────────────────────────────────────
def list_enrolled():
    meta = load_metadata()
    name_map = meta.get("name_map", {})
    if not name_map:
        print("📭 No users enrolled yet.")
        return
    print(f"\n{'─'*40}")
    print(f"  ENROLLED USERS ({len(name_map)} total)")
    print(f"{'─'*40}")
    for idx, name in name_map.items():
        folder = os.path.join(DATASET_PATH, name)
        count = len(os.listdir(folder)) if os.path.exists(folder) else 0
        print(f"  [{idx}] {name:<25} {count} samples")
    print(f"{'─'*40}\n")


# ───────────────────────────────────────────────
# DELETE USER
# ───────────────────────────────────────────────
def delete_person(name: str):
    folder = os.path.join(DATASET_PATH, name.strip())
    if not os.path.exists(folder):
        print(f"❌ No dataset found for '{name}'.")
        return
    confirm = input(f"⚠ Delete '{name}' and retrain? [y/N]: ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return
    shutil.rmtree(folder)
    print(f"🗑 Removed dataset for '{name}'.")
    if os.path.exists(DATASET_PATH) and os.listdir(DATASET_PATH):
        print("⚙ Retraining model without deleted user...")
        train_model()
    else:
        # Remove model if no one left
        for p in [MODEL_PATH, METADATA_PATH]:
            if os.path.exists(p):
                os.remove(p)
        print("⚠ No users remaining. Model cleared.")


# ───────────────────────────────────────────────
# CLI ENTRY POINT
# ───────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="SecureID — Biometric Face Recognition System",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python face_recognition_system.py --mode enroll --name "Alice"
  python face_recognition_system.py --mode recognize
  python face_recognition_system.py --mode list
  python face_recognition_system.py --mode delete --name "Alice"
  python face_recognition_system.py --mode train
        """
    )
    parser.add_argument("--mode", required=True,
                        choices=["enroll", "recognize", "list", "delete", "train"],
                        help="Operation mode")
    parser.add_argument("--name", type=str, default="",
                        help="Person's name (required for enroll/delete)")

    args = parser.parse_args()
    ensure_dirs()

    if args.mode == "enroll":
        if not args.name:
            parser.error("--name is required for enroll mode")
        enroll_person(args.name)

    elif args.mode == "recognize":
        recognize()

    elif args.mode == "list":
        list_enrolled()

    elif args.mode == "delete":
        if not args.name:
            parser.error("--name is required for delete mode")
        delete_person(args.name)

    elif args.mode == "train":
        print("⚙ Retraining model from existing dataset...\n")
        train_model()


if __name__ == "__main__":
    main()