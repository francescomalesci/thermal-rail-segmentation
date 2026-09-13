import os
import cv2
import torch
import yaml
import argparse
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from ultralytics import YOLO
import models
import time

def reconstruct_dynamic_lane(raw_mask):
    height, width = raw_mask.shape

    base_region = raw_mask[height - 50:height, :]
    histogram = np.sum(base_region, axis=0)
    if np.max(histogram) == 0:
        return np.zeros_like(raw_mask)
        
    anchor_x = int(np.argmax(histogram))
    
    centers_y = []
    centers_x = []
    widths = []
    last_x = anchor_x
    
    for y in range(height - 1, int(height * 0.2), -1): 
        row = np.where(raw_mask[y, :] > 0)[0]
        if len(row) > 0:
            valid = [p for p in row if abs(p - last_x) < (width * 0.15)]
            if len(valid) > 5:
                cx = int(np.median(valid))
                w = valid[-1] - valid[0]
                
                if len(widths) > 10:
                    w_base = np.median(widths[:10])
                    if w > w_base * 1.8 and y < height * 0.8:
                        break 
                        
                centers_y.append(y)
                centers_x.append(cx)
                widths.append(w)
                last_x = cx
            else:
                break
        else:
            break
            
    if len(centers_y) < 30:
         return np.zeros_like(raw_mask)
         
    y_arr = np.array(centers_y)
    x_arr = np.array(centers_x)
    
    poly1 = np.polyfit(y_arr, x_arr, 1) 
    poly2 = np.polyfit(y_arr, x_arr, 2) 
    
    if abs(poly2[0]) > 0.0008: 
        p = np.poly1d(poly1)
    else:
        p = np.poly1d(poly2)
        
    base_widths = []
    for y in range(height - 30, height):
        row = np.where(raw_mask[y, :] > 0)[0]
        if len(row) > 0:
            base_widths.append(row[-1] - row[0])
            
    w_bottom = int(np.median(base_widths)) if len(base_widths) > 0 else int(np.median(widths[:15]))
    w_bottom = int(w_bottom * 1.15) 
    w_bottom = max(150, min(w_bottom, 480)) 
    
    horizon_y = centers_y[-1]
    ratio_cut = (height - horizon_y) / height
    w_top = int(w_bottom * (0.10 + 0.15 * ratio_cut)) 
    
    perfect_mask = np.zeros_like(raw_mask)
    for y in range(int(horizon_y), height):
        cx = int(p(y))
        ratio = (y - horizon_y) / (height - horizon_y)
        current_w = int(w_top + (w_bottom - w_top) * ratio)
        
        x1 = max(0, cx - current_w // 2)
        x2 = min(width, cx + current_w // 2)
        
        if x1 < x2:
            perfect_mask[y, x1:x2] = 1.0
            
    return perfect_mask

def process_video(args):
    print("[INFO] Initializing YOLO...")
    yolo_model = YOLO(args.yolo_weights)

    print("[INFO] Initializing SAM2...")
    with open(args.sam_config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    sam_model = models.make(config['model']).cuda()
    sam_model.load_state_dict(torch.load(args.sam_weights))
    sam_model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {args.input_video}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_video, fourcc, fps, (w, h))

    frame_count = 0
    print(f"[INFO] Processing video ({w}x{h} @ {fps} fps)...")
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # --- PHASE A: DYNAMIC MASK EXTRACTION ---
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
            
        input_img_res = pil_img.resize((1024, 1024))
        img_tensor = transform(input_img_res).unsqueeze(0).cuda()
            
        with torch.no_grad():
            pred = sam_model.infer(img_tensor)
            mask_raw = torch.sigmoid(pred).squeeze().cpu().numpy()
            
        mask_resized = Image.fromarray((mask_raw * 255).astype(np.uint8)).resize((w, h), resample=Image.BILINEAR)
        mask_binary = (np.array(mask_resized) > 127).astype(float)
            
        track_mask = reconstruct_dynamic_lane(mask_binary)
        
        # --- PHASE B: YOLO INFERENCE ---
        results = yolo_model(frame, conf=args.conf_thresh, verbose=False)[0]
        
        # --- PHASE C: LOGICAL GATE (MASK OVERLAP) ---
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            x1_c, y1_c = max(0, x1), max(0, y1)
            x2_c, y2_c = min(w, x2), min(h, y2)
            
            box_area_on_mask = track_mask[y1_c:y2_c, x1_c:x2_c]
            
            if np.any(box_area_on_mask == 1.0):
                # CRITICAL HAZARD (Red Box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"CRITICAL: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                # IGNORED OBSTACLE (Yellow Box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, f"Ignored: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # --- PHASE D: VISUAL MASK OVERLAY ---
        alpha = 0.4
        frame[:, :, 2] = np.where(track_mask == 1.0, frame[:, :, 2] * (1 - alpha) + 255 * alpha, frame[:, :, 2])
        frame[:, :, 1] = np.where(track_mask == 1.0, frame[:, :, 1] * (1 - alpha), frame[:, :, 1])
        frame[:, :, 0] = np.where(track_mask == 1.0, frame[:, :, 0] * (1 - alpha), frame[:, :, 0])
        
        out.write(frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"[INFO] Processed {frame_count} frames...")

    cap.release()
    out.release()
    elapsed = (time.time() - start_time) / 60
    print(f"\n[INFO] Processing complete in {elapsed:.2f} minutes. Saved to: {args.output_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Thermal Rail Hazard Detection")
    parser.add_argument('--input_video', type=str, required=True, help="Path to input video")
    parser.add_argument('--output_video', type=str, default="output.mp4", help="Path to save output video")
    parser.add_argument('--yolo_weights', type=str, required=True, help="Path to YOLO best.pt")
    parser.add_argument('--sam_weights', type=str, required=True, help="Path to SAM2 model.pth")
    parser.add_argument('--sam_config', type=str, default='configs/thermal-rail-sam2.yaml', help="Path to SAM2 config")
    parser.add_argument('--conf_thresh', type=float, default=0.5, help="YOLO confidence threshold")
    
    args = parser.parse_args()
    process_video(args)