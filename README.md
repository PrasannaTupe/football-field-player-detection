#  Player Re-Entry Tracker using YOLOv11 and ByteTrack

This project detects and tracks players in a sports video using a fine-tuned YOLOv11 model. It identifies which players, initially visible in the mid-ground portion of the video, re-enter the frame in the final goalpost segment. The output video includes bounding boxes and IDs, with re-entering players highlighted in green.

---

##  Key Features

- Player detection using a fine-tuned YOLOv11 model
- Player tracking using Ultralytics' integrated **ByteTrack**
- Re-entry detection based on frame-wise ID persistence
- Real-time drawing of bounding boxes and IDs on video
- Summary of initial and re-entering players

---

## Environment & Dependencies

Tested on:
- Python 3.10 or higher
- CPU or GPU (CUDA optional)

Pretrained YOLOv11 model (Download) - https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view

Install all required libraries:

```bash
pip install ultralytics opencv-python torch torchvision numpy
```

---

##  Project Structure

Make a folder with the following files:

├── player_tracker.py # Main tracking script
├── best.pt # Fine-tuned YOLOv11 model 
├── video_input.mp4 # Input video

---

## How to Run
Run the tracker script:
```bash
python player_tracker.py
```

---
## Output

A output.mp4 video will be generated as output with -
- Bounding Boxes: Red = player ID not seen earlier, Green = player who re-entered
- ID Overlay: Each tracked player is labeled with a consistent ID
- Console Output: Lists initial and re-entering player IDs
  <img width="985" alt="image" src="https://github.com/user-attachments/assets/677de2ac-1778-41d6-9c18-5da6429eb584" />
  <img width="987" alt="image" src="https://github.com/user-attachments/assets/085e878a-8308-4abc-80a3-5be2b6f804e1" />


---

For questions or collaboration, feel free to reach out.
