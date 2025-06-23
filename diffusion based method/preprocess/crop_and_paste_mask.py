import numpy as np
from PIL import Image

def crop_and_green_mask(image, mask, crop_size=(512, 512)):
    """
    以mask區域為中心crop，並自動縮放mask使其bounding box不超過crop寬高40%，
    並把圖片mask區域塗成(0,255,0)，回傳(crop_img, crop_mask, (crop_x, crop_y), (mask_bbox, scale))
    """
    img = image.copy()
    # 若mask為RGB自動轉L
    if mask.mode != 'L':
        msk = mask.convert('L')
    else:
        msk = mask.copy()
    crop_w, crop_h = crop_size
    mask_np = np.array(msk)
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError('mask為全黑')
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    bbox_w = max_x - min_x + 1
    bbox_h = max_y - min_y + 1
    # 計算縮放比例
    max_mask_w = int(crop_w * 0.4)
    max_mask_h = int(crop_h * 0.4)
    scale = min(max_mask_w / bbox_w, max_mask_h / bbox_h, 1.0)
    # resize mask & image
    if scale < 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.BILINEAR)
        msk = msk.resize((int(msk.width * scale), int(msk.height * scale)), Image.NEAREST)
        mask_np = np.array(msk)
        ys, xs = np.where(mask_np > 0)
        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()
        bbox_w = max_x - min_x + 1
        bbox_h = max_y - min_y + 1
    # 以mask中心為crop中心
    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2
    crop_x1 = max(0, center_x - crop_w // 2)
    crop_y1 = max(0, center_y - crop_h // 2)
    crop_x2 = crop_x1 + crop_w
    crop_y2 = crop_y1 + crop_h
    # 若超出邊界則往回補
    if crop_x2 > img.width:
        crop_x1 = img.width - crop_w
        crop_x2 = img.width
    if crop_y2 > img.height:
        crop_y1 = img.height - crop_h
        crop_y2 = img.height
    crop_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    crop_mask = msk.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    # 將mask區域塗成(0,255,0)
    crop_img_np = np.array(crop_img)
    crop_mask_np = np.array(crop_mask)
    crop_img_np[crop_mask_np > 0] = [0, 255, 0]
    crop_img = Image.fromarray(crop_img_np)
    return crop_img, crop_mask, (crop_x1, crop_y1), ((min_x, min_y, max_x, max_y), scale)

def paste_back_to_image(orig_img, crop_img, crop_mask, crop_xy, mask_bbox_scale, mode='no_resize'):
    """
    將crop_img貼回原圖
    mode='no_resize': 直接貼回resize後的原圖
    mode='resize': 將crop_img resize回原mask bbox大小再貼回原圖
    """
    orig = orig_img.copy()
    crop_x1, crop_y1 = crop_xy
    (min_x, min_y, max_x, max_y), scale = mask_bbox_scale
    if mode == 'no_resize':
        # 直接貼回resize後的原圖
        orig.paste(crop_img, (crop_x1, crop_y1), crop_mask)
    elif mode == 'resize':
        # 將crop_img resize回原mask bbox大小
        crop_img = crop_img.resize((int(crop_img.width / scale), int(crop_img.height / scale)), Image.BILINEAR)
        crop_mask = crop_mask.resize((int(crop_mask.width / scale), int(crop_mask.height / scale)), Image.NEAREST)
        # bbox_w = max_x - min_x + 1
        # bbox_h = max_y - min_y + 1
        # crop_img_resized = crop_img.resize((bbox_w, bbox_h), Image.BILINEAR)
        # crop_mask_resized = crop_mask.resize((bbox_w, bbox_h), Image.NEAREST)
        # # 建立一個與原圖同大小的透明圖層
        # paste_layer = Image.new('RGBA', orig.size)
        # # 將resize後的crop貼到正確位置
        # paste_layer.paste(crop_img_resized.convert('RGBA'), (min_x, min_y), crop_mask_resized)
        # # 合併到原圖
        # orig = Image.alpha_composite(orig.convert('RGBA'), paste_layer).convert(orig.mode)
        orig.paste(crop_img, (int(crop_x1/scale), int(crop_y1/scale)), crop_mask)
    else:
        raise ValueError('mode必須為no_resize或resize')
    return orig
