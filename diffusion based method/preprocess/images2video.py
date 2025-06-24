import cv2
import os
import argparse

def images_to_video(img_dir, output_video, fps=30):
    exts = ('.png', '.jpg', '.jpeg', '.bmp')
    files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(exts)])
    if not files:
        print("找不到任何圖片！")
        return
    # 自動建立影片輸出資料夾
    out_dir = os.path.dirname(output_video)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    first_img = cv2.imread(os.path.join(img_dir, files[0]))
    h, w = first_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))
    for fname in files:
        img = cv2.imread(os.path.join(img_dir, fname))
        if img is None:
            print(f"無法讀取: {fname}")
            continue
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        out.write(img)
    out.release()
    print(f"已輸出影片: {output_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="將圖片序列組成影片")
    parser.add_argument('--imgdir', required=True, help='圖片資料夾')
    parser.add_argument('--out', required=True, help='輸出影片檔案')
    parser.add_argument('--fps', type=int, default=30, help='影片fps (預設30)')
    args = parser.parse_args()
    images_to_video(args.imgdir, args.out, args.fps)
