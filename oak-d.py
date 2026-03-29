import depthai as dai
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("tomato.pt")

pipeline = dai.Pipeline()

# RGB camera
cam = pipeline.create(dai.node.Camera)
cam.build()
frameOut = cam.requestOutput((640, 480), type=dai.ImgFrame.Type.BGR888i)

# Stereo depth
left = pipeline.create(dai.node.Camera)
left.build(dai.CameraBoardSocket.CAM_B)
leftOut = left.requestOutput((640, 400), type=dai.ImgFrame.Type.GRAY8)

right = pipeline.create(dai.node.Camera)
right.build(dai.CameraBoardSocket.CAM_C)
rightOut = right.requestOutput((640, 400), type=dai.ImgFrame.Type.GRAY8)

stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # align depth to RGB
stereo.setOutputSize(640, 480)

leftOut.link(stereo.left)
rightOut.link(stereo.right)

colorQ = frameOut.createOutputQueue(maxSize=4, blocking=False)
depthQ = stereo.depth.createOutputQueue(maxSize=4, blocking=False)

pipeline.start()

depth_frame = None

while pipeline.isRunning():
    color_msg = colorQ.tryGet()
    depth_msg = depthQ.tryGet()

    if depth_msg is not None:
        depth_frame = depth_msg.getFrame()  # uint16, mm

    if color_msg is None or depth_frame is None:
        continue

    frame = color_msg.getCvFrame()
    results = model(frame, verbose=False, conf=0.7)[0]
    annotated = results.plot()

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Sample depth in a small patch around center for robustness
        patch = depth_frame[
            max(0, cy - 5):cy + 5,
            max(0, cx - 5):cx + 5,
        ]
        valid = patch[patch > 0]
        if valid.size == 0:
            continue
        dist_mm = int(np.median(valid))
        dist_m = dist_mm / 1000.0

        label = f"{dist_m:.2f} m"
        cv2.putText(annotated, label, (cx - 30, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2)
        cv2.circle(annotated, (cx, cy), 4, (0, 80, 255), -1)

    cv2.imshow("OAK-D Tomato Detection + Depth", annotated)
    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
pipeline.stop()
