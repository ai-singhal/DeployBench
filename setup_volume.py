"""
setup_volume.py — Run once to populate the Modal Volume with COCO data.

Usage:
    modal run setup_volume.py

Downloads 50 random COCO val2017 images and their filtered annotations
into the shared "deploybench-data" Volume.
"""

import modal

app = modal.App("deploybench-setup")
volume = modal.Volume.from_name("deploybench-data", create_if_missing=True)


@app.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    volumes={"/data": volume},
    timeout=300,
)
def setup():
    import json
    import os
    import random
    import urllib.request
    import zipfile

    # Download COCO val2017 annotations
    ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    print("Downloading COCO annotations...")
    urllib.request.urlretrieve(ann_url, "/tmp/annotations.zip")

    with zipfile.ZipFile("/tmp/annotations.zip", "r") as z:
        z.extract("annotations/instances_val2017.json", "/tmp")

    # Load annotations and pick 50 random images
    with open("/tmp/annotations/instances_val2017.json") as f:
        coco = json.load(f)

    random.seed(42)
    selected_images = random.sample(coco["images"], 50)
    selected_ids = {img["id"] for img in selected_images}

    # Filter annotations to selected images only
    filtered_anns = [a for a in coco["annotations"] if a["image_id"] in selected_ids]
    filtered_coco = {
        "images": selected_images,
        "annotations": filtered_anns,
        "categories": coco["categories"],
    }

    os.makedirs("/data/coco_annotations", exist_ok=True)
    with open("/data/coco_annotations/instances_val2017.json", "w") as f:
        json.dump(filtered_coco, f)

    # Download the 50 images
    os.makedirs("/data/coco_val2017", exist_ok=True)
    for i, img in enumerate(selected_images):
        url = img["coco_url"]
        path = f"/data/coco_val2017/{img['file_name']}"
        if not os.path.exists(path):
            urllib.request.urlretrieve(url, path)
            print(f"  [{i+1}/50] Downloaded {img['file_name']}")

    volume.commit()
    print(f"Setup complete: {len(selected_images)} images, {len(filtered_anns)} annotations")
