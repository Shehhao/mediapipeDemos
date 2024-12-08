import argparse

import cv2  # 新增
import mediapipe as mp
import numpy as np

from videosource import FileSource, WebcamSource

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=3)


def main(inp):
    if inp is None:
        source = WebcamSource()
    else:
        source = FileSource(inp)

    out = None  # 用來記錄 VideoWriter 實例

    with mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose:

        for idx, (frame, frame_rgb) in enumerate(source):
            # 若尚未初始化 out，則在第一個 frame 時初始化
            if out is None:
                height, width = frame.shape[:2]
                # 若 source 有 fps 屬性可用，否則預設 30fps
                fps = getattr(source, 'fps', 30)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec,
                )

                # get landmarks as numpy
                landmarks = results.pose_landmarks.landmark
                np_landmarks = np.array(
                    [(lm.x, lm.y, lm.z, lm.visibility) for lm in landmarks]
                )
                print(np_landmarks.shape)

            # 將處理後的 frame 寫出到檔案
            if out is not None:
                out.write(frame)

            source.show(frame)

    # 迴圈結束後釋放資源
    if out is not None:
        out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Choose video file otherwise webcam is used."
    )
    parser.add_argument(
        "-i", metavar="path-to-file", type=str, help="Path to video file"
    )

    args = parser.parse_args()
    main(args.i)
