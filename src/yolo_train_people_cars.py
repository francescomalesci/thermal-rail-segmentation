from ultralytics import YOLO

def main():
    model = YOLO("yolo26s.pt")

    model.train(
        data="flir_adas.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
	cache=True,
	workers=4,
        project="Tesi_Modelli",
        name="yolo26s_flir_person_car",
        
        # --- Iperparametri termici (Doppio OOD) ---
        optimizer="auto",
        cos_lr=False,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        erasing=0.4,
        auto_augment="randaugment",
        close_mosaic=10,
    )

if __name__ == '__main__':
    main()