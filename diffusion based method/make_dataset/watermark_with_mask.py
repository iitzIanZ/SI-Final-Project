import sys
import random
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np

def add_watermark_with_mask(image_path, text, font_size=40, position=None, font_path=None, random_pos=False, rect_mask=False, both_mask=False):
    # 開啟原始圖片
    image = Image.open(image_path).convert('RGB')
    width, height = image.size

    # 建立水印圖層
    watermark_layer = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(watermark_layer)

    # 設定字型
    if font_path is None:
        # 嘗試自動尋找常見字型（優先中文字型）
        possible_fonts = [
            'C:/Windows/Fonts/msjh.ttc',  # 微軟正黑體
            'C:/Windows/Fonts/msjhl.ttc', # 微軟正黑體 Light
            'C:/Windows/Fonts/arial.ttf', # Arial 英文字型
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', # Linux 常見
        ]
        for pf in possible_fonts:
            if os.path.exists(pf):
                font = ImageFont.truetype(pf, font_size)
                break
        else:
            print('未指定字型且找不到常見字型，將使用預設字型（無法調整大小，建議指定 --font_path）')
            font = ImageFont.load_default()
    else:
        font = ImageFont.truetype(font_path, font_size)

    # 計算文字尺寸與bbox（新版Pillow建議用textbbox）
    try:
        bbox0 = draw.textbbox((0, 0), text, font=font)
        text_width = bbox0[2] - bbox0[0]
        text_height = bbox0[3] - bbox0[1]
    except AttributeError:
        text_width, text_height = font.getsize(text)
        bbox0 = (0, 0, text_width, text_height)

    # 決定水印位置
    if random_pos:
        x = random.randint(0, max(0, width - text_width))
        y = random.randint(0, max(0, height - text_height))
    elif position is not None:
        x, y = position
    else:
        x = (width - text_width) // 2
        y = (height - text_height) // 2

    # 取得實際繪製時的bbox
    try:
        bbox = draw.textbbox((x, y), text, font=font)
    except AttributeError:
        bbox = (x, y, x + text_width, y + text_height)

    # 畫水印（白色，完全不透明）
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))

    # 合成水印到原圖
    watermarked = Image.alpha_composite(image.convert('RGBA'), watermark_layer)
    watermarked = watermarked.convert('RGB')

    # 產生mask（白色為水印，黑色為其他）
    mask = Image.new('L', image.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.text((x, y), text, font=font, fill=255)

    # 計算mask的bounding box
    mask_np = np.array(mask)
    ys, xs = np.where(mask_np > 0)
    if len(xs) > 0 and len(ys) > 0:
        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()
        rect_bbox = (min_x, min_y, max_x + 1, max_y + 1)
    else:
        rect_bbox = (x, y, x + text_width, y + text_height)

    if rect_mask or both_mask:
        mask_rect = Image.new('L', image.size, 0)
        mask_rect_draw = ImageDraw.Draw(mask_rect)
        mask_rect_draw.rectangle(rect_bbox, fill=255)
    else:
        mask_rect = None

    if rect_mask:
        return watermarked, mask_rect
    elif both_mask:
        return watermarked, mask, mask_rect
    else:
        return watermarked, mask

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='在圖片上加水印並產生mask')
    parser.add_argument('--image', required=True, help='輸入圖片路徑')
    parser.add_argument('--text', required=True, help='水印文字')
    parser.add_argument('--output_image', default='output_watermarked.jpg', help='輸出水印圖片路徑')
    parser.add_argument('--output_mask', default='output_mask.png', help='輸出mask圖片路徑')
    parser.add_argument('--font_size', type=int, default=40, help='水印字體大小')
    parser.add_argument('--font_path', default=None, help='字型檔路徑（可選）')
    parser.add_argument('--random', action='store_true', help='隨機水印位置')
    parser.add_argument('--x', type=int, help='水印x座標（若不指定則置中或隨機）')
    parser.add_argument('--y', type=int, help='水印y座標（若不指定則置中或隨機）')
    parser.add_argument('--rect_mask', action='store_true', help='mask為矩形區塊而非文字形狀')
    parser.add_argument('--both_mask', action='store_true', help='同時產生文字形狀與矩形區塊mask')
    args = parser.parse_args()

    pos = None
    if args.x is not None and args.y is not None:
        pos = (args.x, args.y)

    if args.both_mask:
        watermarked, mask, mask_rect = add_watermark_with_mask(
            image_path=args.image,
            text=args.text,
            font_size=args.font_size,
            position=pos,
            font_path=args.font_path,
            random_pos=args.random,
            rect_mask=False,
            both_mask=True
        )
        watermarked.save(args.output_image)
        mask.save(args.output_mask)
        # 產生矩形mask檔名
        base, ext = os.path.splitext(args.output_mask)
        rect_mask_path = f"{base}_rect{ext}"
        mask_rect.save(rect_mask_path)
        print(f"水印圖片已儲存至: {args.output_image}")
        print(f"水印mask已儲存至: {args.output_mask}")
        print(f"矩形mask已儲存至: {rect_mask_path}")
    else:
        watermarked, mask = add_watermark_with_mask(
            image_path=args.image,
            text=args.text,
            font_size=args.font_size,
            position=pos,
            font_path=args.font_path,
            random_pos=args.random,
            rect_mask=args.rect_mask
        )
        watermarked.save(args.output_image)
        mask.save(args.output_mask)
        print(f"水印圖片已儲存至: {args.output_image}")
        print(f"水印mask已儲存至: {args.output_mask}")


### 使用範例:
#python watermark_with_mask.py --image image.jpg --text "water mark" --output_image text_img.jpg --output_mask text_mask.png --font_size 100 --random --both_mask