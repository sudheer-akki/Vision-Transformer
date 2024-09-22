import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from ViT_model import ViT_Model
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm, trange
from dataset import Custom_dataset
import torch.nn.functional as F

folder = "gestures"
device = torch.device("cuda" if torch.cuda.is_available else "cpu")

print(f"Using: {device}")

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
N_EPOCHS = 1
LR = 1e-3 #0.001

model = ViT_Model(num_classes=int(len(class_names))).to(device)

# only train the parameters with requires_grad set to True
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
criterion = CrossEntropyLoss()


training_set = Custom_dataset(folder=folder,class_names=class_names, shuffle=True,train= True)
test_set = Custom_dataset(folder=folder,class_names=class_names, shuffle=False,train= False)

train_dataloader = DataLoader(training_set, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False)

train_loss_list, valid_loss_list = [], []
for epoch in trange(N_EPOCHS, desc="Training"):
    train_loss = 0.0 
    for idx, sample in enumerate(tqdm(train_dataloader,desc=f"Epoch {epoch+1}", leave=False)):
        label = torch.tensor(sample["label"]).to(device)
        class_idx = torch.tensor(sample["class_idx"]).to(device)
        img = torch.tensor(sample["image"],dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        class_name = sample["class_name"][0]
        class_index = class_names.index(class_name)

        vit_output = model(img)
        print(vit_output)
        if torch.isnan(vit_output).any():
            print("NaNs detected in model output")

        assert vit_output.size(dim=1) == len(class_names)

        # get class probabilities
        #probabilities = F.softmax(vit_output[0], dim=0)

        predicted_index = vit_output.argmax(dim=1).item()
        predicted_class_name = class_names[predicted_index]

        # probabilities should sum up to 1
        loss = criterion(vit_output, label)
        print(f"Loss: {loss.item()}")
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch +1}/{N_EPOCHS} loss: {train_loss:.2f}")


    with torch.no_grad():
        correct, total = 0, 0
        valid_loss = 0.0
        for idx, sample in enumerate(tqdm(test_dataloader,desc="Testing", leave=False)):
            label = torch.tensor(sample["label"]).to(device)
            label_index = label.argmax(dim=1)
            img = torch.tensor(sample["image"])[None,:,:,:].to(device)
            vit_output = model(img)
            loss = criterion(vit_output, label)
            print(f"Loss: {loss.item()}")
            valid_loss += loss.item()

    # set model back to trianing mode
    model.train()
    # get average loss values
    train_loss = train_loss / len(train_dataloader)
    valid_loss = valid_loss / len(test_dataloader)

    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)
    print(f"Epoch {epoch +1}/{N_EPOCHS}, Train loss: {train_loss:.2f}, Test loss: {valid_loss:.2f}")


# visualize loss statistics
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
# plot losses
x = list(range(1, N_EPOCHS + 1))
plt.plot(x, train_loss_list, color ="blue", label='Train')
plt.plot(x, valid_loss_list, color="orange", label='Validation')
plt.legend(loc="upper right")
plt.xticks(x)
plt.show()

     

    
