# Perception

The perception pipeline starts from MuJoCo ground truth and keeps the camera boundary modular:

1. `CameraProcessor` validates and preprocesses RGB camera images with OpenCV.
2. `GroundTruthDetector` uses MuJoCo's segmentation renderer for visible masks and projects live body poses for objects occluded by the selected camera.
3. `segmentation_from_detections` creates an RGB visualization.
4. `SceneState` combines detections, drawer state, table state, and robot state.
5. RGB, detection-overlay, and segmentation images are written to `outputs/perception/`.

The `DetectorBackend` protocol is the replacement point for a learned detector. A future model only needs to return `Detection` records; downstream planning does not change.

Run a perception smoke test with:

```powershell
python -m unittest tests.unit.test_perception -v
```
