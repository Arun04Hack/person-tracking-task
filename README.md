# Person Tracking Task

This work was done as part of an AI interview task.  
The objective was to detect and track a fast-moving person in a mountain descent video and maintain tracking till the end as accurately as possible.

---

## Problem Statement

Given a video, track the same moving person across all frames and:

- Maintain a consistent bounding box
- Avoid switching to other objects
- Handle fast motion, scale change, and temporary loss
- Generate trajectory and speed analysis

---

## Challenges Faced

This task was not straightforward because:

- The person is **very small in some frames**
- Movement is **fast and irregular**
- There are **similar-looking objects** (other people / background noise)
- The tracker tends to **drift or jump**
- The person is sometimes **partially lost or blurred**

So a simple tracking approach does not work reliably.

---

## Approaches Tried (and Why They Failed)

### 1. Background Subtraction + Frame Difference
- Idea: detect motion and track it  
- Problem: background also changes → wrong regions selected  
- Result: unstable tracking

---

### 2. MIL Tracker
- Easy to use but weak  
- Failed when:
  - speed increased
  - object scale changed
- Result: loses target mid-video

---

### 3. YOLO (Detection-based approach)
- Detects person better than motion methods  
- But:
  - sometimes detects wrong person initially
  - tracking still drifts after detection
- Result: better but not reliable

---

### 4. Kalman Filter + Tracking
- Helps with prediction  
- But:
  - depends on correct detection
  - alone cannot fix identity switching
- Result: useful but not sufficient

---

### 5. DeepSORT (considered)
- Good for multi-object tracking  
- But:
  - heavy setup
  - not needed for single-object focus
- Result: not used in final solution

---

## Final Working Approach

The final solution is a **hybrid tracking system**:

### Pipeline

→ Bootstrap (initial target selection)
→ CSRT Tracker (main tracking)
→ Validation (reject wrong jumps)
→ Local Re-detection (near predicted position)
→ Global Re-detection (fallback)
→ Output video + analysis


---

## Key Ideas That Made It Work

### 1. Do NOT trust tracker blindly
- If bounding box jumps too far → reject it

---

### 2. Use motion + color together
- Target has **warm/red tone**
- Helps differentiate from background

---

### 3. Predict next position
- Based on previous movement (velocity)
- Helps search in the correct region

---

### 4. Local re-detection
- Instead of searching entire frame:
  - search **near expected location**
- Reduces wrong detections

---

### 5. Global fallback
- If local fails → search whole frame
- Helps recover from full loss

---

## Output

The solution generates:

- **Tracked Video**
  - Bounding box around the person
- **Analysis Plot**
  - Trajectory
  - X & Y position over time
  - Speed graph

---

## What I Learned

This task changed how I think about tracking:

- Tracking ≠ just applying a tracker  
- Real solution = **Detection + Tracking + Validation + Recovery**

### Understanding of methods:

- **MIL** → simple but weak  
- **CSRT** → more robust for motion & scale  
- **YOLO** → strong detection, but needs tracking  
- **Kalman Filter** → good for prediction, not enough alone  
- **DeepSORT** → useful for multi-object identity tracking  

---

### Conclusion

This was not about finding a perfect solution, but about how we approach the problem.

Through multiple failed attempts, I learned:

- where each method breaks
- how to combine techniques
- how to design a robust pipeline

The final result is not perfect, but it works reliably for most of the video and demonstrates a clear problem-solving approach.

## How to Run

Install dependencies:

```bash
pip install opencv-contrib-python numpy matplotlib


