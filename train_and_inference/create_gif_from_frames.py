import os
import imageio
import cv2
import argparse
from IPython.display import Image as IPImage, display
def create_gif_from_frames(frames_folder, output_gif_path, resize_factor=0.5, fps=60):
    """
    將指定資料夾中的照片壓縮並轉換為 GIF。
    :param frames_folder: 照片資料夾
    :param output_gif_path: 輸出的 GIF 路徑
    :param resize_factor: 壓縮比例 (0~1)，默認為 0.5
    :param fps: GIF 的幀率 (frames per second)
    """
    # 獲取所有圖片檔案，並按照檔名排序
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith((".png", ".jpg", ".jpeg"))])

    if not frame_files:
        raise ValueError(f"No frames found in folder: {frames_folder}")

    frames = []
    for frame_file in frame_files:
        frame_path = os.path.join(frames_folder, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Failed to load frame: {frame_path}, skipping...")
            continue

        # 壓縮圖片
        height, width = frame.shape[:2]
        new_width = int(width * resize_factor)
        new_height = int(height * resize_factor)
        resized_frame = cv2.resize(frame, (new_width, new_height))

        # 將 BGR 轉換為 RGB，並添加到 frames 列表
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb_frame)

    # 保存 GIF
    imageio.mimsave(output_gif_path, frames, fps=fps)
    print(f"GIF saved to {output_gif_path}")
    
    display(IPImage(filename=output_gif_path))

# 使用範例
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a GIF from frames in a folder.")
    
    parser.add_argument("--frames_folder", type=str, default="inpainted_results" 
                        , help="Path to the folder containing frames.")
    
    parser.add_argument("--output_gif_path", type=str, default="output.gif", 
                        help="Path to save the output GIF.")
    args = parser.parse_args()

    create_gif_from_frames(args.frames_folder, args.output_gif_path, resize_factor=0.5, fps=60)
