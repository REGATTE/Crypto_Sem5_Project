# the backend 
import torch
import torchvision
from torchvision import transforms
from flask import Flask, render_template, request, redirect, url_for, abort
from PIL import Image, ImageOps
from datetime import datetime
#model def
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

device = torch.device("cpu")
model = 0
model = Net().to(device)

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