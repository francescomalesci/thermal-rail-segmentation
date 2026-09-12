import cv2
import numpy as np
import os
import glob
import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Batch inference for Thermal Rail Segmentation & Detection")
    parser.add_argument('--image_dir', type=str, required=True, help="Path to input images directory")
    parser.add_argument('--mask_dir', type=str, required=True, help="Path to input red masks directory")
    parser.add_argument('--output_dir', type=str, default="results_batch", help="Path to save output images")
    parser.add_argument('--weights', type=str, required=True, help="Path to YOLO best.pt weights")
    return parser.parse_args()

def run_batch_inference(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load YOLO Model
    yolo_model = YOLO(args.weights)

    # 2. Get File Lists
    img_list = sorted(glob.glob(os.path.join(args.image_dir, "*.jpg")))
    mask_list = sorted(glob.glob(os.path.join(args.mask_dir, "*.png"))) 
    
    num_masks = len(mask_list)

    if not img_list or num_masks == 0:
        print("Error: Empty directories or invalid paths.")
        return

    critical_count = 0
    ignored_count = 0

    print("Starting batch inference...")

    # 3. Safe iteration based on filename
    for iteration, img_path in enumerate(img_list):
        img_cv = cv2.imread(img_path)
        filename = os.path.basename(img_path) 
        
        # Extract index to avoid misalignment
        try:
            sample_num = int(filename.split('_')[1].split('.')[0])
        except (IndexError, ValueError):
            print(f"Formatting error in filename: {filename}")
            continue
            
        # Select corresponding mask using modulo operator
        mask_path = mask_list[sample_num % num_masks]
        mask_bgr = cv2.imread(mask_path)

        if img_cv is None or mask_bgr is None:
            continue

        # 4. Dynamic Binary Mask Extraction
        b, g, r = cv2.split(mask_bgr)
        diff_r_g = cv2.subtract(r, g)
        mask_bin = np.where((diff_r_g > 30), 255, 0).astype(np.uint8)

        # 5. YOLO Inference
        yolo_results = yolo_model.predict(img_cv, conf=0.2, iou=0.4, verbose=False)[0]

        # 6. Create Output Image with Overlay
        red_overlay = np.zeros_like(img_cv)
        red_overlay[:, :, 2] = 255 
        fused_img = cv2.addWeighted(img_cv, 0.6, red_overlay, 0.4, 0)
        mask_3d = np.stack([mask_bin, mask_bin, mask_bin], axis=-1) > 127
        output_img = np.where(mask_3d, fused_img, img_cv).astype(np.uint8)

        # 7. Spatial AND Logic
        for box in yolo_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            # Object bottom-center point
            px = min(int((x1 + x2) / 2), img_cv.shape[1] - 1)
            py = min(y2, img_cv.shape[0] - 1)

            if mask_bin[py, px] > 127:
                # Object is on the tracks (CRITICAL)
                cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(output_img, f"CRITICAL: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.circle(output_img, (px, py), 5, (0, 0, 255), -1)
                critical_count += 1
            else:
                # Object is outside the tracks (IGNORED)
                cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(output_img, f"Ignored: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.circle(output_img, (px, py), 5, (0, 255, 255), -1)
                ignored_count += 1

        # 8. Save Output
        cv2.imwrite(os.path.join(args.output_dir, filename), output_img)

        if (iteration + 1) % 100 == 0:
            print(f"Processed {iteration + 1}/{len(img_list)} images...")

    # 9. Final Report
    print("\n--- BATCH INFERENCE REPORT ---")
    print(f"Total Images: {len(img_list)}")
    print(f"CRITICAL Anomalies (on track): {critical_count}")
    print(f"IGNORED Anomalies (off track): {ignored_count}")

if __name__ == '__main__':
    args = parse_args()
    run_batch_inference(args)