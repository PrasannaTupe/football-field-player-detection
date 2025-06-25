import cv2 
from ultralytics import YOLO
from collections import defaultdict

# INITIAL SETUP (Variables)
MODEL_PATH = 'best.pt'
VIDEO_PATH = 'video_input.mp4'
OUTPUT_PATH = 'output.mp4'
CONFIDENCE_THRESHOLD = 0.4
PHASE_SPLIT_SECONDS = 8  # Midground to Goalpost transition point

# LOAD MODEL
model = YOLO(MODEL_PATH)

# PREPARE VIDEO 
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
phase_split_frame = int(fps * PHASE_SPLIT_SECONDS) ## splitting frames for initial player detection

# VIDEO OUTPUT SETUP
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height)) 

# TRACKING SETUP (Ultralytics Inbuilt Tracker)
results = model.track(
    source=VIDEO_PATH,
    conf=CONFIDENCE_THRESHOLD,
    tracker="bytetrack.yaml",
    stream=True,  
    persist=True,
    verbose=False
)

# TRACK RE-ENTRIES 
initial_ids = set()
reentered_ids = set()
frame_index = 0

for result in results:
    frame = result.orig_img
    ids = result.boxes.id.cpu().tolist() if result.boxes.id is not None else []

    # Strictly capture initial IDs only in first phase
    if frame_index == phase_split_frame:
        print(f"Midground phase ends at frame {frame_index}")
    if frame_index <= phase_split_frame:
        initial_ids.update(ids)
    elif frame_index > phase_split_frame: # Add ids in ans if re captured in second phase
        for id in ids:
            if id in initial_ids:
                reentered_ids.add(id)

    # Draw bounding boxes and ID labels
    for box, idx in zip(result.boxes.xyxy, ids):
        x1, y1, x2, y2 = map(int, box.tolist())
        color = (0, 255, 0) if idx in reentered_ids else (0, 0, 255)
        label = f"ID: {int(idx)}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out.write(frame) #Saving output
    frame_index += 1


# CLEANUP
cap.release()
out.release()

# RESULTS
print("Tracking complete.")
print("Output saved to:", OUTPUT_PATH)
print("Initial player IDs:", initial_ids)
print("Re-entered player IDs near goalpost:", reentered_ids)

# METRICS
reenter_percent = sum(reentered_ids)/sum(initial_ids) * 100
print("Percent of players re-entering the frame near goalpost: ", reenter_percent)