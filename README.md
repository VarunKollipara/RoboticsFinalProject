# RoboticsFinalProject

A computer vision and robotics project that enables a Lite 6 robot arm to automatically trace, erase, and replicate dry-erase drawings.

This repository contains the code for a Master's level robotics project that automates the process of perceiving a designated whiteboard drawing encased in AprilTags, physically erasing the board, and redrawing the captured image. The system utilizes a UFACTORY Lite 6 robot arm and a ZED Stereo Camera.

## Features

* **Visual Perception & Stroke Extraction:** Uses OpenCV to isolate the drawing area via AprilTag boundaries (tag36h11). The system applies skeletonization to extract 1-pixel centerlines and traces paths into ordered, continuous strokes to reduce pen-up travel distance.
* **Smart Homography & Depth Mapping:** Transforms 2D pixel coordinates into 3D world space using a combination of AprilTag homography and ZED camera point-cloud depth data.
* **Automated Tool Swapping:** Autonomously navigates between staging areas to swap between a whiteboard eraser and a dry-erase marker.
* **Multi-Pass Erasing:** Utilizes a custom offset algorithm to perform multi-pass scrubbing along ink lines, ensuring thick strokes are entirely removed.
* **Teaching Mode:** Includes a manual freedrive teaching mode to dynamically set board bounds and tool staging locations without hardcoding coordinates.

## Hardware Requirements

* UFACTORY Lite 6 Robot Arm
* ZED Stereo Camera (mounted for global observation)
* Whiteboard, dry-erase marker, and whiteboard eraser
* 4x AprilTags (`tag36h11`, IDs 0-3) marking the corners of the drawing area

## Software Dependencies

Ensure you have the following Python packages installed:

```bash
pip install opencv-python numpy scipy pupil-apriltags
```

*Note: You will also need the XArm Python SDK and the ZED SDK installed on your machine.*

## Project Structure

* `final_proj.py`: The main execution script handling perception, path planning, and kinematic control.
* `utils/`: Contains utility scripts including visual processing and ZED camera initialization (`vis_utils.py`, `zed_camera.py`).
* `robot_positions.json`: Generated locally after running Teaching Mode to store physical coordinates (ignored by version control).

## Usage

1. **Configure IP:** Open `final_proj.py` and ensure `ROBOT_IP` matches the local IP address of your Lite 6 arm.
2. **Setup Whiteboard:** Place AprilTags 0, 1, 2, and 3 in the bottom-left, top-left, top-right, and bottom-right corners of your target drawing area respectively.
3. **Execute:** Run the main script from your terminal:
   ```bash
   python final_proj.py
   ```
4. **Calibration (First Run):** If `USE_HARDCODED_POSITIONS` is set to `False`, the script will prompt you to enter "Teaching Mode." Physically guide the robotic arm to the four corners of the board and your tool staging areas. These coordinates will save locally to `robot_positions.json`.
5. **Execution Flow:** * The robot moves to its observation pose and captures a snapshot.
   * A debug window will appear displaying the extracted skeletonized paths. Press `k` to confirm and proceed.
   * The robot will retrieve the eraser, scrub the drawn lines, swap to the marker, and redraw the image.

## Troubleshooting

* **Kinematic Errors:** If the robot halts with a kinematic error, ensure your drawing bounds are not pushing the arm outside of its reachable workspace. You can adjust the `MARKER_SAFE_Y_OFFSET_MM` and `DRAW_HOVER_OFFSET_M` variables in the configuration section.
* **Missing Tags:** Ensure the ZED camera has a clear, well-lit view of all four AprilTags. The script will abort if it cannot construct the bounding polygon.
