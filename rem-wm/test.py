from remwm import WatermarkRemover

# # Initialize the WatermarkRemover with the default Florence model
# remover = WatermarkRemover()

# # Define input and output paths
# input_image_path = "original\\test5.png"
# output_image_path = "test5_mask.png"

# # Process the image
# remover.process_images_florence_lama(input_image_path, output_image_path)


remover = WatermarkRemover()
# 呼叫內部 API 取出 segmentation mask & bounding box
mask, bbox = remover.generate_mask_florence("original\\test5.png")