import cv2
import os
import numpy as np
from datetime import datetime

def collect_realtime_data():
    # Create directories for different lighting conditions
    base_dir = "realtime_train_data"
    conditions = ["bright", "dim", "dark"]
    classes = ["eyes_open", "eyes_closed"]
    
    for condition in conditions:
        for cls in classes:
            os.makedirs(os.path.join(base_dir, condition, cls), exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    current_condition = conditions[0]
    current_class = classes[0]
    frame_count = 0
    
    print("Data Collection Started")
    print("Press:")
    print("'b' - bright lighting")
    print("'d' - dim lighting")
    print("'k' - dark lighting")
    print("'o' - eyes open class")
    print("'c' - eyes closed class")
    print("'s' - save frame")
    print("'q' - quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Show current settings
        cv2.putText(frame, f"Condition: {current_condition}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"Class: {current_class}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        
        cv2.imshow('Data Collection', frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('b'):
            current_condition = "bright"
        elif key == ord('d'):
            current_condition = "dim"
        elif key == ord('k'):
            current_condition = "dark"
        elif key == ord('o'):
            current_class = "eyes_open"
        elif key == ord('c'):
            current_class = "eyes_closed"
        elif key == ord('s'):
            # Save frame
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{timestamp}.jpg"
            save_path = os.path.join(base_dir, current_condition, current_class, filename)
            cv2.imwrite(save_path, frame)
            frame_count += 1
            print(f"Saved frame {frame_count} to {save_path}")
    
    cap.release()
    cv2.destroyAllWindows()
    return base_dir