import os
import argparse
from PIL import Image

def images_to_gif(img_dir, output_gif, duration=100):
    exts = ('.png', '.jpg', '.jpeg', '.bmp')
    files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(exts)])
    if not files:
        print("找不到任何圖片！")
        return
    # 自動建立GIF輸出資料夾
    out_dir = os.path.dirname(output_gif)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    imgs = [Image.open(os.path.join(img_dir, f)).convert('RGB') for f in files]
    imgs[0].save(output_gif, save_all=True, append_images=imgs[1:], duration=duration, loop=0)
    print(f"已輸出GIF: {output_gif}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="將圖片序列組成GIF動畫")
    parser.add_argument('--imgdir', required=True, help='圖片資料夾')
    parser.add_argument('--out', required=True, help='輸出GIF檔案')
    parser.add_argument('--duration', type=int, default=100, help='每幀持續時間(ms)，預設100')
    args = parser.parse_args()
    images_to_gif(args.imgdir, args.out, args.duration)
