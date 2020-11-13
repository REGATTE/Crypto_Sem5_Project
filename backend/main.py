# the backend 
import torch
import torchvision
from torchvision import transforms
from flask import Flask, render_template, request, redirect, url_for, abort
from PIL import Image, ImageOps
from datetime import datetime

#loading model
model = torch.load("saved_model/mnist_crypto.pt")   #path of the model to be filled.
model.eval()

app = Flask(__name__)
@app.route("/", methods=["GET", "POST"])
def file():
    if request.method == "GET":
        return render_template("index.html")
    if request.method == "POST":
        #save the file
        f = request.files["file"]
        filepath = "./static/" + datetime.now().strftime("%Y%m%d%H%M%S") + ".png "
        f.save(filepath)
        #Load image file
        image = Image.open(filepath)
        #Converted to be handled by PyTorch(Resize, black and white inversion, normalization, dimension addition)
        image = ImageOps.invert(image.convert("L")).resize((28, 28))
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        image = transform(image).unsqueeze(0)
        #Make predictions
        output = model(image)
        _, prediction = torch.max(output, 1)
        result = prediction[0].item()

        return render_template("backend/index.html", filepath=filepath, result=result)


if __name__ == "__main__":
    app.run(port="5000", debug=True)