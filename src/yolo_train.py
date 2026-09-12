import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Fine-tuning Pipeline for Thermal Domain")
    parser.add_argument('--data', type=str, required=True, help="Path to dataset.yaml")
    parser.add_argument('--weights', type=str, default="yolo11s.pt", help="Path to base pretrained weights")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch', type=int, default=8, help="Batch size")
    parser.add_argument('--name', type=str, default="yolo_thermal_augmented", help="Name of the training run")
    return parser.parse_args()

def main():
    args = parse_args()

    # Initialize model with base pretrained weights
    model = YOLO(args.weights)

    # Start training with domain-specific thermal & sim-to-real augmentations
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        
        # --- Targeted Augmentations for Thermal/Sim-to-Real ---
        mosaic=0.4,       # Reduced: limits grid artifacts
        mixup=0.15,       # Blends images to smooth synthetic edges
        erasing=0.2,      # Random pixel patching to prevent overfitting
        scale=0.25,       # Controlled scale variation (prevents micro-objects)
        degrees=5.0,      # Minor rotation (+/- 5 degrees)
        translate=0.1,    # Light panning
        fliplr=0.5,       # Horizontal flip enabled
        flipud=0.0,       # NEVER vertical (maintains physical ground-sky orientation)
        hsv_h=0.0,        # Zero hue variation (greyscale/thermal domain)
        hsv_s=0.0,        # Zero saturation variation
        hsv_v=0.3,        # Variable thermal brightness
        close_mosaic=10,  # Disable mosaic in final 10 epochs for bounding box calibration
        
        name=args.name,
    )

if __name__ == '__main__':
    main()