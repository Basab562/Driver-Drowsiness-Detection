import cv2
import dlib
import numpy 
import dlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from pygame import mixer
import os
import sys  # Add this import
import torch.multiprocessing as mp
from torch.multiprocessing import freeze_support
from torch.cuda.amp import autocast, GradScaler
import torch.cuda.amp as amp
import math
from scipy.spatial import distance as dist

# Suppress pygame welcome message
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

# Replace the existing GPU check code
def setup_device():
    if not torch.cuda.is_available():
        return torch.device('cpu')
    
    try:
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        return device
    except Exception as e:
        return torch.device('cpu')

# Use this function to get device
device = setup_device()

# Remove redundant debug prints and just keep one status message
print(f"Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

# Clear CUDA cache once
torch.cuda.empty_cache()

# ======================
# Part 1: Model Training
# ======================

DATASET_PATH = (r"C:\Users\basab\OneDrive\Desktop\PROJECT IIT KALYANI\DATASETS\train_data")
IMG_SIZE = (24, 24)
BATCH_SIZE = 32
EPOCHS = 15

# Define CNN model
class EyeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(inplace=True),  # inplace operations
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# Data preparation and augmentation
class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

# Load dataset and split
full_dataset = datasets.ImageFolder(DATASET_PATH)
train_indices, val_indices = train_test_split(
    list(range(len(full_dataset))),
    test_size=0.2,
    stratify=full_dataset.targets,
    random_state=42
)

train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(
        degrees=15,  # Added required degrees parameter
        shear=11.46, 
        scale=(0.8, 1.2)
    ),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

train_subset = TransformedSubset(Subset(full_dataset, train_indices), train_transform)
val_subset = TransformedSubset(Subset(full_dataset, val_indices), val_transform)

train_loader = DataLoader(
    train_subset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    num_workers=4,  # Adjust based on your CPU cores
    pin_memory=True if torch.cuda.is_available() else False,
    prefetch_factor=2
)

val_loader = DataLoader(
    val_subset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    num_workers=4,
    pin_memory=True if torch.cuda.is_available() else False,
    prefetch_factor=2
)

# Train model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EyeModel().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Add after optimizer initialization
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.1, 
    patience=5
)

# Replace the existing training function with this updated version
def train_epoch():
    model.train()
    running_loss = 0.0
    scaler = amp.GradScaler() if torch.cuda.is_available() else None
    
    for inputs, labels in train_loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Updated autocast usage to avoid warnings
        with autocast(enabled=torch.cuda.is_available()):
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
        
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(train_loader)

# Modify validation loop
@torch.inference_mode()  # More efficient than no_grad
def validate():
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in val_loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)
        outputs = model(inputs).squeeze()
        val_loss += criterion(outputs, labels).item()
        
        preds = torch.sigmoid(outputs) > 0.5
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return val_loss / len(val_loader), correct / total

# Add these constants after existing ones
HEAD_POSE_THRESH = 30  # degrees
GAZE_THRESH = 0.3  # normalized distance from center
ATTENTION_FRAMES = 20  # consecutive frames to trigger alarm

# Add these helper functions before main()
def get_head_pose(shape):
    """Calculate approximate head pose from facial landmarks"""
    # Get nose, chin and side points
    nose = shape[30]
    left = shape[0]
    right = shape[16]
    
    # Calculate angle from face orientation
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    face_angle = math.degrees(math.atan2(dy, dx))
    
    return abs(face_angle)

def check_gaze(eye_points, frame_center, frame):
    """Check if gaze is centered based on eye position"""
    eye_center = np.mean(eye_points, axis=0)
    dist_from_center = dist.euclidean(eye_center, frame_center)
    # Normalize by frame size
    normalized_dist = dist_from_center / (frame.shape[1] / 2)
    return normalized_dist

def main():
    global model, detector, predictor  # Add if needed

    # Your existing training code
    full_dataset = datasets.ImageFolder(DATASET_PATH)
    train_indices, val_indices = train_test_split(
        list(range(len(full_dataset))),
        test_size=0.2,
        stratify=full_dataset.targets,
        random_state=42
    )

    # ...existing training setup...

    # Training loop
    for epoch in range(EPOCHS):
        running_loss = train_epoch()
        val_loss, val_acc = validate()
        
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{EPOCHS}')
        print(f'Train Loss: {running_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Val Acc: {val_acc:.4f}\n')
        print(f'Current learning rate: {optimizer.param_groups[0]["lr"]}')

    # Save model
    torch.save(model.state_dict(), r'G:\IITKAL\models\eye_model.pth')

    # Detection system setup and loop
    # Initialize alarm
    try:
        mixer.init()
        alarm_path = r"G:\IITKAL\alarm.wav.wav"
        if not os.path.exists(alarm_path):
            print(f"Error: Alarm file not found at {alarm_path}")
            sys.exit(1)
        alarm = mixer.Sound(alarm_path)
        # Test alarm
        alarm.set_volume(1.0)  # Set volume to maximum
        print("Alarm initialized successfully")
    except Exception as e:
        print(f"Error initializing alarm: {str(e)}")
        sys.exit(1)

    # Dlib face detection
    SHAPE_PREDICTOR_PATH = (r"G:\IITKAL\models\shape_predictor_68_face_landmarks.dat")
    if not os.path.exists(SHAPE_PREDICTOR_PATH):
        print(f"Error: {SHAPE_PREDICTOR_PATH} not found!")
        print("Please download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        sys.exit(1)

    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(r"G:\IITKAL\models\shape_predictor_68_face_landmarks.dat")
    except AttributeError:
        print("Error: dlib not properly installed. Please reinstall dlib.")
        sys.exit(1)
    except Exception as e:
        print(f"Error initializing dlib: {str(e)}")
        sys.exit(1)

    def eye_aspect_ratio(eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def get_eye_region(frame, eye_points):
        x1, y1 = np.min(eye_points, axis=0)
        x2, y2 = np.max(eye_points, axis=0)
        eye = frame[y1:y2, x1:x2]
        eye = cv2.cvtColor(eye, cv2.COLOR_BGR2RGB)  # Convert to RGB
        eye = cv2.resize(eye, (24, 24))
        return eye

    # Load PyTorch model
    model = EyeModel().to(device)
    model.load_state_dict(torch.load(r"G:\IITKAL\models\eye_model.pth", map_location=device))
    model.eval()

    EYE_AR_THRESH = 0.2  # Decreased from 0.25
    HEAD_POSE_THRESH = 45  # Increased from 30
    GAZE_THRESH = 0.4  # Increased from 0.3
    CONSECUTIVE_FRAMES = 25  # Increased from 20 for fewer false positives
    COUNTER = 0
    ALARM_ON = False

    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_center = np.array([frame.shape[1]/2, frame.shape[0]/2])
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            for face in faces:
                landmarks = predictor(gray, face)
                landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

                left_eye = landmarks[42:48]
                right_eye = landmarks[36:42]

                # Get eye regions
                left_eye_region = get_eye_region(frame, left_eye)
                right_eye_region = get_eye_region(frame, right_eye)

                # Convert to tensors
                left_tensor = torch.from_numpy(left_eye_region).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
                right_tensor = torch.from_numpy(right_eye_region).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

                # Predict
                with torch.no_grad():
                    left_pred = torch.sigmoid(model(left_tensor)).item()
                    right_pred = torch.sigmoid(model(right_tensor)).item()

                # EAR calculation
                ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2

                # Drowsiness decision
                if ear < EYE_AR_THRESH or (left_pred < 0.5 and right_pred < 0.5):
                    COUNTER += 1
                    if COUNTER >= CONSECUTIVE_FRAMES and not ALARM_ON:
                        ALARM_ON = True
                        alarm.play()
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2) 
                else:
                    COUNTER = 0
                    ALARM_ON = False

                # Display info
                cv2.putText(frame, f"EAR: {ear:.2f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                cv2.putText(frame, f"CNN: {left_pred:.2f}|{right_pred:.2f}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                # Head pose and gaze direction
                head_pose = get_head_pose(landmarks)
                gaze_left = check_gaze(left_eye, frame_center=(frame.shape[1]//2, frame.shape[0]//2), frame=frame)
                gaze_right = check_gaze(right_eye, frame_center=(frame.shape[1]//2, frame.shape[0]//2), frame=frame)

                # Combined drowsiness and attention check
                is_drowsy = ear < EYE_AR_THRESH and (left_pred < 0.5 and right_pred < 0.5)  # Changed from 'or' to 'and'
                is_distracted = (head_pose > HEAD_POSE_THRESH and  # Changed from 'or' to 'and'
                               (gaze_left > GAZE_THRESH or gaze_right > GAZE_THRESH))

                if is_drowsy or is_distracted:
                    COUNTER += 1
                    if COUNTER >= CONSECUTIVE_FRAMES:  # Only trigger after consistent detection
                        if not ALARM_ON:
                            ALARM_ON = True
                            try:
                                alarm.stop()
                                alarm.play(loops=-1)
                                print("Alarm triggered")
                            except Exception as e:
                                print(f"Error playing alarm: {str(e)}")
                        
                        if is_drowsy:
                            cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                        if is_distracted:
                            cv2.putText(frame, "ATTENTION ALERT!", (10, 60),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
                else:
                    COUNTER = max(0, COUNTER - 1)  # Gradually decrease counter
                    if COUNTER == 0 and ALARM_ON:
                        ALARM_ON = False
                        alarm.stop()

                # Display debug info
                cv2.putText(frame, f"EAR: {ear:.2f}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                cv2.putText(frame, f"Head: {head_pose:.1f}", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                cv2.putText(frame, f"Gaze L/R: {gaze_left:.2f}/{gaze_right:.2f}", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            cv2.imshow('Drowsiness Detection', frame)
            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        # Cleanup
        if 'alarm' in locals():
            alarm.stop()
        mixer.quit()
        cap.release()
        cv2.destroyAllWindows()
        torch.cuda.empty_cache()  # Clear GPU memory

if __name__ == '__main__':
    freeze_support()
    main()
