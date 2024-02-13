from embeddings.embedder import Embedder
import os
import glob
import cv2 as cv

SHELF_CONSTANT = 'shelf_4'
embedder = Embedder()
embedder.initialise()
products_images_folder = f"images/products/{SHELF_CONSTANT}"

if os.path.exists(products_images_folder) and os.path.isdir(products_images_folder):
    subfolders = [f for f in os.listdir(products_images_folder) if os.path.isdir(os.path.join(products_images_folder, f))]
    for product_folder in subfolders:
        product_images_path = os.path.join(products_images_folder, product_folder)
        png_files = glob.glob(os.path.join(product_images_path, '*.png'))
        product_images_list = []
        sku = product_folder.split('_')[0]
        weight = float(product_folder.split('_')[1])
        for file in png_files:
            image = cv.imread(file)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            product_images_list.append(image)
        embedder.insert_many(SHELF_CONSTANT, product_images_list, sku, weight)



