import os
import random
import numpy as np
from PIL import Image

def random_crop_avoid_mask(image, mask, crop_size, max_tries=100):
    h, w = image.size[1], image.size[0]
    ch, cw = crop_size
    mask_np = np.array(mask)
    for _ in range(max_tries):
        x = random.randint(0, w - cw)
        y = random.randint(0, h - ch)
        crop_mask = mask_np[y:y+ch, x:x+cw]
        if np.sum(crop_mask) == 0:
            return x, y
    raise RuntimeError('無法在不覆蓋mask的情況下crop到影像，請減小crop尺寸或檢查mask')

def random_crop(image, mask, crop_size, max_tries=100):
    h, w = image.size[1], image.size[0]
    ch, cw = crop_size
    x = random.randint(0, w - cw)
    y = random.randint(0, h - ch)
    return x, y

def random_paste_mask(crop_img, crop_mask, mask, max_mask_ratio=0.4):
    # 取得mask的bounding box
    mask_np = np.array(mask)
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError('mask為全黑，無法貼上')
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    mask_crop = mask.crop((min_x, min_y, max_x+1, max_y+1))
    mask_w, mask_h = mask_crop.size
    crop_w, crop_h = crop_img.size
    # 計算縮放比例
    max_mask_w = int(crop_w * max_mask_ratio)
    max_mask_h = int(crop_h * max_mask_ratio)
    scale = min(max_mask_w / mask_w, max_mask_h / mask_h, 1.0)
    if scale < 1.0:
        mask_crop = mask_crop.resize((int(mask_w * scale), int(mask_h * scale)), Image.NEAREST)
        mask_w, mask_h = mask_crop.size
    # 隨機貼到crop_img內
    max_x_paste = crop_w - mask_w
    max_y_paste = crop_h - mask_h
    if max_x_paste < 0 or max_y_paste < 0:
        raise ValueError('mask太大，無法貼進crop影像')
    px = random.randint(0, max_x_paste)
    py = random.randint(0, max_y_paste)
    # 將mask貼到crop_img上，並將mask區域塗成(0,255,0)
    crop_img = crop_img.copy()
    crop_img_np = np.array(crop_img)
    mask_crop_np = np.array(mask_crop)
    mask_area = mask_crop_np > 0
    crop_img_np[py:py+mask_h, px:px+mask_w][mask_area] = [0,255,0]
    # 產生新mask
    new_mask = Image.new('L', crop_img.size, 0)
    new_mask_np = np.array(new_mask)
    new_mask_np[py:py+mask_h, px:px+mask_w][mask_area] = 255
    return Image.fromarray(crop_img_np), Image.fromarray(new_mask_np)

def generate_crops(image_path, mask_path, crop_size, count, out_dir, avoid_mask=True):
    os.makedirs(out_dir, exist_ok=True)
    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')
    img_name = os.path.basename(image_path).split('.')[0]
    for i in range(count):
        if avoid_mask:
            x, y = random_crop_avoid_mask(image, mask, crop_size)
        else:
            x, y = random_crop(image, mask, crop_size)
        crop_img = image.crop((x, y, x+crop_size[0], y+crop_size[1]))
        crop_mask = mask.crop((x, y, x+crop_size[0], y+crop_size[1]))
        crop_img.save(os.path.join(out_dir, img_name+f'_{i+1}_gt.png'))
        pasted_img, new_mask = random_paste_mask(crop_img, crop_mask, mask, max_mask_ratio=0.4)
        pasted_img.save(os.path.join(out_dir, img_name+f'_{i+1}_in.png'))
        new_mask.save(os.path.join(out_dir, img_name+f'_{i+1}_mask.png'))
        # print(f'已儲存: crop_{i+1}.png, crop_{i+1}_mask.png')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='隨機crop影像並貼上mask')
    parser.add_argument('--image', required=True, help='輸入圖片路徑')
    parser.add_argument('--mask', required=True, help='mask圖片路徑')
    parser.add_argument('--count', type=int, required=True, help='產生數量')
    parser.add_argument('--out_dir', required=True, help='輸出目錄')
    parser.add_argument('--crop_w', type=int, required=True, help='crop寬度')
    parser.add_argument('--crop_h', type=int, required=True, help='crop高度')
    parser.add_argument('--avoid_mask', action='store_true', help='是否避開mask區域 (預設不避開)')
    args = parser.parse_args()
    generate_crops(args.image, args.mask, (args.crop_w, args.crop_h), args.count, args.out_dir, avoid_mask=args.avoid_mask)
    #python random_crop_paste_mask.py --image damaged_image/text_img.jpg --mask damaged_image/text_mask_rect.png --count 10 --out_dir output --crop_w 512 --crop_h 512