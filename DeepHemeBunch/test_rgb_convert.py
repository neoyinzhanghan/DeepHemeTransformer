from PIL import Image

image_path = "/media/greg/534773e3-83ea-468f-a40d-46c913378014/neo/results_dir/BMA-diff_2024-10-05 15:21:08/cells/M4/M4-E4-E1-MO2_8350-16.jpg"
save_path = "/media/hdd3/neo/test_image.jpg"

image = Image.open(image_path).convert("RGB")
image.save(save_path)
