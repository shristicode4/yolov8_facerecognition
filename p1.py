import os
import shutil

def filter_and_save_dataset(data_dir, output_dir):

    image_extensions = ['.jpg']
    text_extensions = ['.txt'] 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(data_dir):
        images = [file for file in files if file.endswith(tuple(image_extensions))]
        texts = [file for file in files if file.endswith(tuple(text_extensions))]

        image_dict = {os.path.splitext(img)[0]: os.path.join(root, img) for img in images}
        
        for text in texts:
            text_name = os.path.splitext(text)[0]
            if text_name in image_dict:
                image_path = image_dict[text_name]
                text_path = os.path.join(root, text)
                
                relative_path = os.path.relpath(root, data_dir)
                new_root = os.path.join(output_dir, relative_path)
                if not os.path.exists(new_root):
                    os.makedirs(new_root)

                shutil.copy2(image_path, os.path.join(new_root, os.path.basename(image_path)))
                shutil.copy2(text_path, os.path.join(new_root, os.path.basename(text_path)))

if __name__ == "__main__":
    data_dir = r'split_by_ratio\val'
    output_dir = r'filtered_dataset_val'
    num_processed = filter_and_save_dataset(data_dir, output_dir)
    print(f"Number of image-text pairs processed: {num_processed}")
