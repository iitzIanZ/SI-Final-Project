import cv2
import os
import argparse

def extract_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"無法開啟影片: {video_path}")
        return
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"總幀數: {frame_count}")
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        filename = os.path.join(output_dir, f"frame_{idx:05d}.png")
        cv2.imwrite(filename, frame)
        idx += 1
        if idx % 50 == 0:
            print(f"已擷取 {idx} 幀...")
    cap.release()
    print(f"完成！共擷取 {idx} 幀，儲存於 {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="將影片逐格擷取成圖片")
    parser.add_argument("--video", required=True, help="影片檔案路徑")
    parser.add_argument("--outdir", required=True, help="輸出圖片資料夾")
    args = parser.parse_args()
    extract_frames(args.video, args.outdir)
