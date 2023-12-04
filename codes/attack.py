import os
import cv2
import numpy as np
import argparse
import torch
from vit_pytorch import ViT
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, Subset
import matplotlib.pyplot as plt

if os.path.isdir('models') != True:
    os.mkdir('models')

if torch.cuda.is_available():
    device = "cuda:0" 
else:  
    device = "cpu"  

trig_transform = transforms.Compose([transforms.ToTensor()])

def imshow(img, labels):
    img = img + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title('               '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
    plt.show()

def normalize8(I):
    mn = I.min()
    mx = I.max()
    mx -= mn
    I = ((I - mn)/mx)*255
    return I.astype(np.uint8)

def add_watermark_fn(image, signature, trig_type, sig_size, pos):
    w,h = sig_size
    background = np.array(image)
    foreground = cv2.resize(signature, (w,h))
    background = np.transpose(background, (1,2,0))
    
    if trig_type=='embed':
        alpha_foreground = foreground[:,:,3]/255
        foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
        foreground = foreground/255
        for color in range(0, 3):
            background[pos[0]:pos[0]+w, pos[1]:pos[1]+h, color] = alpha_foreground * foreground[:,:,color] + background[pos[0]:pos[0]+w, pos[1]:pos[1]+h, color] * (1 - alpha_foreground)

        background = np.transpose(background, (2,0,1))
        return background
    
    elif trig_type=='noise':
        gaussian = np.random.normal(0, 25, (w, h))
        for color in range(0, 3):
            background[pos[0]:pos[1], pos[0]:pos[1], c] = gaussian + background[pos[0]:pos[1], pos[0]:pos[1], c]
            background = np.clip(background, 0, 255).astype(np.uint8)
        background = np.transpose(background, (2,0,1))
        return background
    
    elif trig_type=='blend':
        for color in range(0,3):
            background[100:150,100:150,color] = cv2.addWeighted(signature[:,:,color], 0.6, background[100:150,100:150,i], 0.5, 0)
        background = np.transpose(background, (2,0,1))
        return background
    
    else:
        print("Unsupported Trigger Type!")
        return -1
        
class TriggerSet(data.Dataset):
    def __init__(self, orig_dataset, source_label, target_label, signature, trig_type, sig_size, pos, transform=trig_transform,):
        self.dataset = orig_dataset
        self.source_label = source_label
        self.target_label = target_label
        self.transform = transform
        self.target_idx = (torch.tensor(self.dataset.targets)==self.source_label).nonzero()
        self.subset = Subset(self.dataset, self.target_idx)
        self.signature = signature
        self.sig_size = sig_size
        self.pos = pos
        self.trig_type = trig_type

    def __getitem__(self, idx):
        image, label = self.subset.__getitem__(idx)
        image = add_watermark_fn(image, self.signature, self.trig_type, self.sig_size, self.pos)
        #print(image.shape)
        image = np.transpose(image, (1,2,0))
        if self.transform:
            image = self.transform(image)
        return image, self.target_label
    
    def __len__(self):
        return len(self.subset)

def ftll_model_loader(model_name):
    model = torch.load("models/wm_{}.pth".format(model_name))
    if model_name == 'ResNet18':
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    elif model_name == 'ResNet50':
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    elif model_name == 'MobileNet':
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier[1].parameters():
            param.requires_grad = True
    elif model_name == 'DenseNet121':
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif model_name == 'DenseNet169':
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif model_name == 'ViT-16':
#         model = ViT(image_size = 224,
#                     patch_size = 32,
#                     num_classes = num_classes,
#                     dim = 64,
#                     depth = 6,
#                     heads = 8,
#                     mlp_dim = 128,
#                     dropout = 0.1,
#                     emb_dropout = 0.1)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif model_name == 'ViT-32':
#         model = ViT(image_size = 224,
#                     patch_size = 32,
#                     num_classes = num_classes,
#                     dim = 64,
#                     depth = 6,
#                     heads = 16,
#                     mlp_dim = 128,
#                     dropout = 0.1,
#                     emb_dropout = 0.1)
       for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        print("Unknown Model Name Error!")
    model = model.to(device)
    return model

    

      
def train(model, attack, trainloader, testloader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    best_metric = 0
    model.train()
    for epoch in range(epochs):  
        print("Epoch {}/{}...".format(epoch+1, epochs))
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            #print(f'Epoch {epoch+1}, Loss: {running_loss / (i + 1)}')

        train_correct = 0
        train_total = 0
        with torch.no_grad():
            for data in trainloader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
        train_accuracy = 100 * (train_correct / train_total)

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_accuracy = 100 * (correct / total)
        print(f'Epoch {epoch+1}, Train Loss: {running_loss / (i + 1)}, Train Acc: {train_accuracy}, Val Acc: {test_accuracy}')
        if test_accuracy > best_metric:
            best_metric = test_accuracy
            #best_metric_epoch = epoch + 1
            torch.save(model, "models/{}_{}.pth".format(attack, model_name))
            print("saved new best metric model")
    return model

def watermark(combined_loader, wm_loader, epochs, model_name=None, model_path = None):   
    #Step 2: After Training the Models
    if model_path is not None:
        model = torch.load(model_path)
    elif model_name is not None:
        #model = model_maker(model_name, num_classes)
        model = torch.load("models/wm_{}.pth".format(model_name))
    else:
        print("Provide one of the following: 'model_name' or 'model_path'")
        return 0
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    wm_metric=0
    model.train()
    for epoch in range(epochs):  
        print("Epoch {}/{}...".format(epoch+1, epochs))
        running_loss = 0.0
        for i, data in enumerate(combined_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            #print(f'Epoch {epoch+1}, Loss: {running_loss / (i + 1)}')

        train_correct = 0
        train_total = 0
        with torch.no_grad():
            for data in combined_loader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
        train_accuracy = 100 * (train_correct / train_total)

        correct = 0
        total = 0
        with torch.no_grad():
            for data in wm_loader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_accuracy = 100 * (correct / total)
        if test_accuracy > wm_metric:
            wm_metric = test_accuracy
            #best_metric_epoch = epoch + 1
            torch.save(model, "models/wmi_attack_{}.pth".format(model_name))
            print("saved new best Watermarked model")
        print(f'Epoch {epoch+1}, Train Loss: {running_loss / (i + 1)}, Train Acc: {train_accuracy}, WM Acc: {test_accuracy}')
        return model
    
def verify_model(model_path, wm_loader, testloader):
    model = torch.load(model_path)
    model = model.to(device)
    model.eval()
    results = {}
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_accuracy = 100 * (correct / total)
    
    wm_correct = 0
    wm_total = 0
    with torch.no_grad():
        for data in wm_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            wm_total += labels.size(0)
            wm_correct += (predicted == labels).sum().item()
    wm_accuracy = 100 * (wm_correct / wm_total)
    print(f'Test Acc: {test_accuracy}, WM Acc: {wm_accuracy}')
    results['Test_Acc'] = test_accuracy
    results['WM_Acc'] = wm_accuracy
    return results

    
def Dataset(dataset_name):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 16
    
    if dataset_name == 'CIFAR-10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)
        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=2)

        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')   
    
    elif dataset_name == 'EUROSAT':
        data = torchvision.datasets.EuroSAT(root='./data', download=True, transform=transform)
        trainset, testset = train_test_split(data, test_size=0.2, stratify=data.targets)
        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=2)
        
        classes = ['Industrial', 'Pasture', 'River', 'Forest', 'AnnualCrop', 'PermanentCrop', 'Highway', 'HerbaceousVegetation', 'Residential', 'SeaLake']
    
    return trainset, testset, trainloader, testloader, classes

def ftll_attack():
    signature = cv2.imread(trig_path, cv2.IMREAD_UNCHANGED)
    print("Dataset Chosen: ", dataset_name)
    trainset, testset, trainloader, testloader, classes = Dataset(dataset_name)
    num_classes = len(classes)
    if model_name == None:
        print("'model_name' paramater cannot be None for training")
        return 0
    else:
        print("Model Selected: ", model_name)
        model = ftll_model_loader(model_name)
        model = train(model, attack, trainloader, testloader, epochs)
        print("Done")

def ftal_attack():
    signature = cv2.imread(trig_path, cv2.IMREAD_UNCHANGED)
    print("Dataset Chosen: ", dataset_name)
    trainset, testset, trainloader, testloader, classes = Dataset(dataset_name)
    num_classes = len(classes)
    if model_name == None:
        print("'model_name' paramater cannot be None for training")
        return 0
    else:
        print("Model Selected: ", model_name)
        model = torch.load("models/wm_{}.pth".format(model_name))
        model = train(model, attack, trainloader, testloader, epochs)
        print("Done")
    
def wmi_attack():
    signature = cv2.imread(trig_path, cv2.IMREAD_UNCHANGED)
    print("Dataset Chosen: ", dataset_name)

    trainset, testset, trainloader, testloader, classes = Dataset(dataset_name)
    num_classes = len(classes)
    print("Classes in this dataset:", num_classes)
    print("Source Label: ", classes[source_label])
    print("Target Label: ", classes[target_label])
    print("\n")
    print("Trigger Type:", trig_type)
    print("Forming Trigger Set for Watermarking..")
    trig_set = TriggerSet(orig_dataset = testset, source_label = source_label, target_label = target_label, trig_type = trig_type, signature = signature, sig_size = sig_size, pos = pos)
    wm_loader = DataLoader(trig_set, batch_size=4)
    dataiter = iter(wm_loader)
    images, labels = next(dataiter)
    print("Labels in a Batch:", labels.shape)
    print("Image Shape in a Batch:", images.shape)
    print("\n")

    combined_set = ConcatDataset([trainset, trig_set])
    combined_loader = DataLoader(combined_set, batch_size=4, shuffle=True)
    if model_name == None:
        print("'model_name' paramater cannot be None for training")
        return 0
    else:
        print("Model Selected: ", model_name)
        print("Watermarking stage starting...\n")
        wm_model = watermark(model_name=model_name, combined_loader=combined_loader, wm_loader = wm_loader, epochs=int(wm_epochs))
        print("Done !")

def verify():
    signature = cv2.imread(trig_path, cv2.IMREAD_UNCHANGED)
    print("Dataset Chosen: ", dataset_name)

    trainset, testset, trainloader, testloader, classes = Dataset(dataset_name)
    num_classes = len(classes)
    print("Classes in this dataset:", num_classes)
    print("Source Label: ", classes[source_label])
    print("Target Label: ", classes[target_label])
    print("Trigger Type:", trig_type)
    print("\n")
    print("Forming Trigger Set for Watermarking..")
    trig_set = TriggerSet(orig_dataset = testset, source_label = source_label, target_label = target_label, trig_type = trig_type, signature = signature, sig_size = sig_size, pos = pos)
    wm_loader = DataLoader(trig_set, batch_size=4)

    results = verify_model(model_path = model_path, wm_loader = wm_loader, testloader=testloader)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack', help="Choose one among:['ftll', 'ftal','wmi']", default="ftal")
    parser.add_argument('--dataset_name', help='Dataset to use', default="CIFAR-10")
    parser.add_argument('--trigger_path', help='Path to the trigger image', default="asu_logo.png")
    parser.add_argument('--trigger_pos', help='Position of the trigger in the image', default=(16,16))
    parser.add_argument('--trigger_size', help='Size of the trigger in the image', default=(16,16))
    parser.add_argument('--trigger_type', help='Type of Trigger: Embed, Blend, Noise', default='embed')
    parser.add_argument('--source_label', help='Source class to choose images for watermarking', default=0)
    parser.add_argument('--target_label', help='Target class to relabel to after watermarking', default=4)
    parser.add_argument('--model_name', help="One among the following: ['ResNet18','ResNet50','MobileNet','DenseNet121','ViT-16','ViT-32']", default=None)
    parser.add_argument('--model_path', help="Path to the model .pth file", default=None)
    parser.add_argument('--train_epochs', help='Epochs for Training (Step 1)', default=5)
    parser.add_argument('--wm_epochs', help='Epochs for Watermarking (Step 2)', default=5)
    args = parser.parse_args()
    
    attack = args.attack
    trig_path = args.trigger_path
    dataset_name = args.dataset_name
    pos = args.trigger_pos
    sig_size = args.trigger_size
    trig_type = args.trigger_type
    source_label = args.source_label
    target_label = args.target_label
    model_name = args.model_name
    model_path = args.model_path
    train_epochs = args.train_epochs
    wm_epochs = args.wm_epochs
    
    if attack == 'ftll':
        ftll_attack()
    elif attack == 'ftal':
        ftal_attack()
    elif attack == 'wmi':
        wmi_attack()
    else:
        print("Please choose an attack among: ['ftll', 'ftal','wmi']")
