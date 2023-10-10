import os
from flask import Flask, request, render_template
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import unique
import io
import base64

app = Flask(__name__)

# Initialize the processor and model
processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

# Define a function to perform semantic segmentation
def perform_segmentation(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    outputs = model(**inputs)
    logits = outputs.logits.cpu()

    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )

    pred_seg = upsampled_logits.argmax(dim=1)[0]
    return pred_seg

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if the POST request has a file part
        if "file" not in request.files:
            return render_template("index.html", error="No file part")

        file = request.files["file"]

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == "":
            return render_template("index.html", error="No selected file")

        if file:
            image_bytes = file.read()
            # Perform semantic segmentation
            pred_seg = perform_segmentation(image_bytes)

            original_image_base64 = base64.b64encode(image_bytes).decode('utf-8')

            # Save the segmented image
            segmented_image_path = os.path.join("uploads", "segmented.png")
            plt.imsave(segmented_image_path, pred_seg.numpy(), cmap="viridis")
            segmented_image_base64 = base64.b64encode(open(segmented_image_path, 'rb').read()).decode('utf-8')
            parts = []
            segments = unique(pred_seg) # Get a list of all the predicted items
            for i in segments:
              mask = pred_seg == i # Filter out anything that isn't the current item
              img = Image.fromarray((mask * 255).numpy().astype(np.uint8))
              name = model.config.id2label[i.item()] # get the item name
              parts.append({"name": name, "base64": image_to_base_64(img)})

            return render_template("index.html", original_image_base64=original_image_base64, segmented_image_base64=segmented_image_base64, parts=parts)

    return render_template("index.html", error=None)


def image_to_base_64(image):
  buffered = io.BytesIO()
  image.save(buffered, format="PNG")
  img_str = base64.b64encode(buffered.getvalue())
  return img_str.decode('utf-8')

if __name__ == "__main__":
    app.run(debug=True)
