import os
import json

data_path = "/Users/olivia/Project/roco/"
data_type = ["train","test","validation"]
image_type = ["non-radiology","radiology"]

cur_image_type = "radiology"

for cur_data_type in data_type:
    image_path = os.path.join(data_path,"data",cur_data_type,cur_image_type,"images")
    caption_path = os.path.join(data_path,"data",cur_data_type,cur_image_type,"captions.txt")
    images = os.listdir(image_path)
    output_path = "dataset/"
    metadata_path = os.path.join(output_path,cur_data_type,"metadata.jsonl")
    os.makedirs(os.path.join(output_path,cur_data_type), exist_ok=True)
    write_index = 0

    with open(caption_path, "r", encoding="utf-8") as rfile, open(metadata_path, "w", encoding="utf-8") as wfile:
        for line in rfile:
            img, caption = line.split("\t")
            img += ".jpg"
            if img not in images:
                # print(img)
                continue
            tmp_dict = {
                "file": img,
                "cap": caption.strip()
            }
            write_index += 1
            json.dump(tmp_dict, wfile)
            wfile.write("\n")
            

    if not len(images) == write_index:
        print(f"{cur_data_type}: Different datasize")
        print(f"Images: {len(images)}")
        print(f"Caption written: {write_index}")
        
        
    print(f"TOTAL {cur_data_type.upper()} DATASIZE = {write_index}")
    

        

    