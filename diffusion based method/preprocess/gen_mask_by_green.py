import os
import cv2
import numpy as np
import argparse

def gen_mask_by_green(src_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    exts = ('.png', '.jpg', '.jpeg', '.bmp')
    files = [f for f in os.listdir(src_dir) if f.lower().endswith(exts)]
    for fname in files:
        img_path = os.path.join(src_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"無法讀取: {img_path}")
        # BGR格式，純綠色為(0,255,0)
        mask = cv2.inRange(img, (0,255,0), (20,255,20))
        # 轉為0/255二值圖
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        # 去除孤島（小連通區）
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        min_area = 100  # 可依需求調整
        cleaned = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                cleaned[labels == i] = 255
        # 膨脹10 pixel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21,21))
        mask = cv2.dilate(cleaned, kernel, iterations=1)
        out_path = os.path.join(out_dir, fname)
        cv2.imwrite(out_path, mask)
        print(f"已產生: {out_path}")
    print(f"全部完成！共處理 {len(files)} 張圖片。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="根據RGB(0,255,0)產生mask")
    parser.add_argument('--src', required=True, help='來源圖片資料夾')
    parser.add_argument('--out', required=True, help='mask輸出資料夾')
    args = parser.parse_args()
    gen_mask_by_green(args.src, args.out)
