import torch
from torchvision.models import ResNet18_Weights
import torchvision.models as models
from tqdm import tqdm
import time

def get_resnet18_model(device):
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.eval()
    return model.to(device), weights.transforms()




def eval(dataset, model, preprocess, device, num_imgs = 20):
    correct = 0
    with torch.no_grad():
        for i, example in enumerate(tqdm(dataset, total=num_imgs)):
            if i==10: # warmup
                start_time = time.time()
            if i >= num_imgs: break # Stop after num_imgs images

            image = preprocess(example['image'].convert('RGB')).unsqueeze(0).to(device)
            output = model(image)

            if example['label'] == output.argmax(1).item():
                correct += 1
        end_time = time.time()
    print(f"Accuracy on {num_imgs} streamed images: {100 * correct / num_imgs:.2f}%. Time taken: {(end_time - start_time)/(num_imgs - 10):.6f} seconds per image")
    return (100 * correct) / num_imgs