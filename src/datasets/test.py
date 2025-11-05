from src.datasets.PH2 import PH2
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

trainset = PH2(train=True, transform=transform)
print(f"Number of training samples: {len(trainset)}")

# Check the first item
X, Y = trainset[0]
print("X shape:", X.shape)
print("Y shape:", Y.shape)