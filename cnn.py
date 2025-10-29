import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as T_F
import os
import warnings
from PIL import Image

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- GLOBAL CONFIGURATION (LOCAL WINDOWS PATHS) ---
TRAIN_ROOT_DIR = r'C:\tasks\moc_dataset\train' 
TEST_ROOT_DIR = r'C:\tasks\moc_dataset\test'   

TRAIN_IMAGE_DIR = os.path.join(TRAIN_ROOT_DIR, 'images')
TRAIN_ANNOTATION_DIR = os.path.join(TRAIN_ROOT_DIR, 'annotations')


# --- Helper Functions ---
def get_transform():
    """Defines the necessary transformation to convert PIL image to PyTorch Tensor."""
    return torchvision.transforms.ToTensor()

def collate_fn(batch):
    """Needed for object detection models to handle batches with variable box counts."""
    return tuple(zip(*batch))

# --- STEP 1: Real-World Dataset Setup ---
class RealObjectDataset(Dataset):
    """
    Custom Dataset to load images (e.g., .jpg) and parse bounding box 
    annotations from matching text files (e.g., .txt) with enhanced validation.
    """
    def __init__(self, img_dir, annotation_dir, num_classes, transforms=None):
        self.img_dir = img_dir
        self.annotation_dir = annotation_dir
        self.transforms = transforms
        self.num_classes = num_classes # Store the number of classes for validation
        
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
        self.ids = [os.path.splitext(f)[0] for f in self.img_files]
        print(f"Found {len(self.ids)} samples in {img_dir}.")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # 1. Load Image
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        
        # 2. Parse Annotations from TXT file
        annotation_name = self.ids[idx] + ".txt"
        annotation_path = os.path.join(self.annotation_dir, annotation_name)
        
        boxes = []
        labels = []
        
        try:
            with open(annotation_path, 'r') as f:
                for line in f:
                    # Assumed format: class_id x1 y1 x2 y2 
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])

                        # --- CRITICAL VALIDATION 1: CHECK CLASS ID RANGE ---
                        # Class IDs must be in the range [1, NUM_CLASSES - 1]
                        if class_id < 1 or class_id >= self.num_classes:
                            raise ValueError(
                                f"Invalid class ID: {class_id} found in file '{annotation_name}'. "
                                f"Model expects IDs in range [1, {self.num_classes - 1}]."
                            )
                        
                        # Read the four bounding box coordinates as generic floats
                        c1, c2, c3, c4 = map(float, parts[1:])

                        # --- VALIDATION 2: ENSURE CORRECT MIN/MAX ORDERING ---
                        x_coords = sorted([c1, c3])
                        y_coords = sorted([c2, c4])

                        x_min, x_max = x_coords[0], x_coords[1]
                        y_min, y_max = y_coords[0], y_coords[1]
                        
                        # Check if the corrected box has a non-zero area.
                        if (x_max > x_min) and (y_max > y_min):
                            boxes.append([x_min, y_min, x_max, y_max])
                            labels.append(class_id)
                        # Else: skip the zero-area or invalid box
                        
        except FileNotFoundError:
            # Continue if no annotation file is found
            pass
        except ValueError as e:
            # Re-raise the value error with the filename for easy debugging
            raise e


        # Handle images with no annotations 
        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        # 3. Create Target Dictionary (Required by Faster R-CNN)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        
        # 4. Apply Transforms
        if self.transforms is not None:
            img = self.transforms(img)
            
        return img, target

# --- STEP 2 & 3 (Model and Training functions remain the same) ---
def get_model_instance_segmentation(num_classes):
    """
    Loads a pre-trained Faster R-CNN model and modifies the final layer 
    for the specific number of classes in our custom dataset.
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def fine_tune_model(model, dataset, num_epochs=2, learning_rate=0.005):
    """
    Defines the training parameters and runs the fine-tuning loop.
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Freeze the CNN Backbone
    for param in model.backbone.parameters():
        param.requires_grad = False
    print("Backbone layers are frozen. Only the RPN and RoI heads will be fine-tuned.")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    data_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn 
    )

    print(f"\nStarting fine-tuning on {device} for {num_epochs} epochs...")
    
    # The Training Loop
    for epoch in range(num_epochs):
        model.train() 
        total_loss = 0.0
        
        for i, (images, targets) in enumerate(data_loader):
            # The error is likely to be caught by the Dataset loader, 
            # but if it passes, this is where the GPU fails.
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward() 
            optimizer.step()

            total_loss += losses.item()

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    print("Fine-tuning complete.")
    return model

# --- Main Execution ---
if __name__ == '__main__':
    
    # === CRITICAL: DEFINE YOUR CLASS NAMES HERE ===
    CLASS_NAMES = [
        "person",  # Index 0 (Reserved)
        "blue_apron",             # Index 1 (Use '1' in annotation TXT files)
        "grey_apron",
        "hairnet",  # Index 0 (Reserved)
        "tray"            # Index 1 (Use '1' in annotation TXT files)        # Index 2 (Use '2' in annotation TXT files)
    ]
    
    NUM_CLASSES = len(CLASS_NAMES) 
    print(f"Configured for {NUM_CLASSES} classes: {CLASS_NAMES}")
    
    # 1. Load the dataset (passing NUM_CLASSES to the dataset)
    print(f"Attempting to load data from: {TRAIN_IMAGE_DIR}")
    try:
        train_dataset = RealObjectDataset(
            img_dir=TRAIN_IMAGE_DIR, 
            annotation_dir=TRAIN_ANNOTATION_DIR,
            num_classes=NUM_CLASSES, # <--- NEW PARAMETER
            transforms=get_transform()
        )
    except Exception as e:
        print("\n--- CRITICAL ERROR ---")
        print("A problem occurred during data loading. Details:")
        print(f"Error details: {e}")
        exit()

    # 2. Load the model and replace the classification head
    model = get_model_instance_segmentation(NUM_CLASSES)
    
    # 3. Start training
    fine_tuned_model = fine_tune_model(model, train_dataset, num_epochs=5)
    
    # 4. Save the model weights
    torch.save(fine_tuned_model.state_dict(), 'faster_rcnn_finetuned.pth')
    print("\nModel saved to faster_rcnn_finetuned.pth")
