import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2 # Library required for video processing
import numpy as np
from PIL import Image
from torchvision.transforms import functional as T_F

# --- CONFIGURATION (UPDATE THESE PATHS) ---
# 1. Path to your saved model weights
MODEL_PATH = r"C:\Users\ACER\Desktop\CNN\faster_rcnn_finetuned.pth"

# 2. Path to the video file you want to process
INPUT_VIDEO_PATH = r"C:\Users\ACER\Downloads\out_one.mp4" # <--- **CHANGE THIS**

# 3. Class definitions (MUST MATCH the order used during training)
CLASS_NAMES = [
    "__background__",  # Index 0
    "Car",             # Index 1
    "Pedestrian",      # Index 2
]
NUM_CLASSES = len(CLASS_NAMES)

# Define a color palette for the bounding boxes (B, G, R format for OpenCV)
COLORS = {
    "Car": (0, 255, 255),       # Yellow/Cyan
    "Pedestrian": (255, 0, 0),  # Blue
    "Default": (0, 165, 255)    # Orange
}

# --- MODEL DEFINITION ---
def get_model_instance_segmentation(num_classes):
    """Loads pre-trained Faster R-CNN and modifies the box predictor."""
    # Load the base model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the classification head
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# --- INFERENCE FUNCTION ---
def run_video_inference():
    """
    Loads the trained model, processes the input video frame-by-frame,
    and displays the results.
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Running inference on device: {device}")

    # 1. Load the model structure and weights
    model = get_model_instance_segmentation(NUM_CLASSES)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval() # Set model to evaluation mode
        print(f"Successfully loaded model weights from: {MODEL_PATH}")
    except FileNotFoundError:
        print(f"ERROR: Model weights not found at {MODEL_PATH}")
        print("Please check the MODEL_PATH variable in the script.")
        return
    
    # 2. Initialize Video Capture
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video file at {INPUT_VIDEO_PATH}")
        return

    # Optional: Setup VideoWriter to save the output video
    # output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # out = cv2.VideoWriter('output_detections.mp4', fourcc, fps, (output_width, output_height))

    # 3. Process video frame-by-frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Finished processing video.")
            break

        # Convert OpenCV BGR image to PyTorch required format (RGB Tensor)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = T_F.to_tensor(rgb_frame).to(device)
        
        # Perform inference
        with torch.no_grad():
            prediction = model([img_tensor])
        
        # Extract predictions for the first image in the batch (which is just one frame)
        boxes = prediction[0]['boxes'].cpu().numpy()
        labels = prediction[0]['labels'].cpu().numpy()
        scores = prediction[0]['scores'].cpu().numpy()
        
        # 4. Draw bounding boxes and labels
        for box, label, score in zip(boxes, labels, scores):
            # Filter detections by a confidence threshold
            if score > 0.7: # Set your minimum confidence threshold here
                x_min, y_min, x_max, y_max = map(int, box)
                
                # Get class name and color
                class_name = CLASS_NAMES[label]
                color = COLORS.get(class_name, COLORS["Default"])
                
                # Draw the box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                
                # Create the label text
                label_text = f"{class_name}: {score:.2f}"
                
                # Draw the label background
                (w, h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x_min, y_min - h - 5), (x_min + w, y_min), color, -1)
                
                # Draw the label text
                cv2.putText(frame, label_text, (x_min, y_min - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # 5. Display the frame
        cv2.imshow('Faster R-CNN Detection', frame)
        
        # Optional: Write the frame to the output video
        # out.write(frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 6. Release resources
    cap.release()
    # out.release() # Release VideoWriter if used
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_video_inference()
