#Load the required module
import os

#Flask related
from flask import Flask, render_template, request, redirect, url_for, abort
from flask_cors import CORS, cross_origin

#PyTorch related
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

# Pillow(PIL)、datetime
from PIL import Image, ImageOps
from datetime import datetime

#Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


device = torch.device("cpu")
model = 0
model = Net().to(device)
#Load the learning model
model.load_state_dict(
    torch.load("./mnist_crypto.pt", map_location=lambda storage, loc: storage)
)
model = model.eval()

app = Flask(__name__)
cors = CORS(app)

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "GET":
        return render_template("index.html")
    if request.method == "POST":
        #Load image file
        image = Image.open(request.files["file"])
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

        return {'result':result}


if __name__ == "__main__":
    app.run(
        port=int(os.environ.get('PORT', 80)),
        host='0.0.0.0',
        debug=True)
