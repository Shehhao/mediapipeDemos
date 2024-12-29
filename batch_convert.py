import os
import cv2
import mediapipe as mp
from pathlib import Path

def convert_videos(input_dir, output_dir):
    # 初始化 MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 取得所有影片檔案
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    input_files = []
    for ext in video_extensions:
        input_files.extend(Path(input_dir).glob(f'*{ext}'))
    
    # 初始化多人姿勢偵測器，調整參數以優化多人偵測
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=2,  # 使用更複雜的模型以提高準確度
        enable_segmentation=True,  # 啟用分割以幫助區分多人
        smooth_segmentation=True  # 平滑分割結果
    ) as pose:
        
        for input_file in input_files:
            # 設定輸出檔案路徑
            output_file = Path(output_dir) / f"{input_file.stem}_detected.mp4"
            print(f"處理中: {input_file} -> {output_file}")
            
            # 開啟輸入影片
            cap = cv2.VideoCapture(str(input_file))
            if not cap.isOpened():
                print(f"無法開啟影片: {input_file}")
                continue
                
            # 取得影片資訊
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # 建立輸出影片寫入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))
            
            # 計數處理的幀數
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 逐幀處理
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 轉換 BGR 到 RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False  # 提高性能
                
                # 進行多人姿勢偵測
                results = pose.process(frame_rgb)
                
                frame_rgb.flags.writeable = True
                
                # 繪製所有偵測到的姿勢
                if results.pose_landmarks:
                    # 只繪製骨架，不包含臉部和手部細節
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(
                            color=(255, 255, 255),
                            thickness=2,
                            circle_radius=2
                        ),
                        connection_drawing_spec=mp_drawing.DrawingSpec(
                            color=(0, 252, 124),
                            thickness=2,
                            circle_radius=1
                        )
                    )
                
                # 寫入處理後的 frame 到輸出影片
                out.write(frame)
                
                # 更新進度
                frame_count += 1
                if frame_count % 30 == 0:  # 每 30 幀顯示一次進度
                    progress = (frame_count / total_frames) * 100
                    print(f"進度: {progress:.1f}% ({frame_count}/{total_frames})")
            
            # 釋放資源
            cap.release()
            out.release()
            
            print(f"完成處理: {output_file}")

if __name__ == "__main__":
    input_dir = "/Users/guxuehao/Downloads/video"
    output_dir = "/Users/guxuehao/Downloads/video_t"
    
    convert_videos(input_dir, output_dir)
