from flask import Flask, render_template, request
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(768, 10)  # 10 classes for FashionMNIST and Caltech101
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
def add_noise_to_gradients(gradients, noise_type="Gaussian", noise_strength=0.1):
    """Add noise to the gradients."""
    for i in range(len(gradients)):
        if noise_type == "Gaussian":
            noise = torch.randn_like(gradients[i]) * noise_strength
        elif noise_type == "Laplacian":
            noise = torch.distributions.Laplace(0, noise_strength).sample(gradients[i].size())
        else:
            raise ValueError("Unsupported noise type. Choose 'Gaussian' or 'Laplacian'.")
        gradients[i].add_(noise)
    return gradients

def run_experiment(batch_size, noise_type, noise_strength, dataset_type):
    # Set the seed for reproducibility
    model_class=None
    torch.manual_seed(50)

    # Check versions
    print(torch.__version__ )

    # Define dataset path based on dataset type
    if dataset_type == "fashion_mnist":
        dataset_path = "d:/Muzny Zuhair/App/FashionMNIST"
    elif dataset_type == "caltech101":
        dataset_path = "d:/Muzny Zuhair/App/Caltech101/caltech101/101_ObjectCategories"
    else:
        raise ValueError("Unsupported dataset type. Choose either 'FashionMNIST' or 'Caltech101'.")

    # Define transforms based on dataset type
    if dataset_type == "fashion_mnist":
        tp = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor()
        ])
    elif dataset_type == "caltech101":
        tp = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1)  # Convert to grayscale
        ])
    tt = transforms.ToPILImage()

    # Set device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print("Running on %s" % device)

    # Helper functions
    def label_to_onehot(target, num_classes=10):
        target = torch.unsqueeze(target, 1)
        onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
        onehot_target.scatter_(1, target, 1)
        return onehot_target

    def cross_entropy_for_onehot(pred, target):
        return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

    def weights_init(m):
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)

    # Load the dataset
    if dataset_type == "fashion_mnist":
        dst = datasets.FashionMNIST(dataset_path, train=True, download=True, transform=tp)
    elif dataset_type == "caltech101":
        dst = datasets.ImageFolder(dataset_path, transform=tp)

    # Fixed set of images
    fixed_img_indices = [0, 1, 2, 3, 4, 5, 6, 7]  # Change these indices if you want different images
    fixed_gt_data = torch.stack([dst[i][0] for i in fixed_img_indices]).to(device)
    fixed_gt_labels = torch.Tensor([dst[i][1] for i in fixed_img_indices]).long().to(device)
    fixed_gt_onehot_labels = label_to_onehot(fixed_gt_labels, num_classes=10)

    # Adjust batch size
    gt_data = fixed_gt_data[:batch_size]
    gt_labels = fixed_gt_labels[:batch_size]
    gt_onehot_labels = fixed_gt_onehot_labels[:batch_size]
    
    fig1, axes = plt.subplots(1, batch_size, figsize=(10, 5))
    if batch_size == 1:
        axes.imshow(tt(gt_data[0].cpu()), cmap='gray')
        axes.set_title(f"GT label is {gt_labels[0].item()}")
    else:
        for i in range(batch_size):
            axes[i].imshow(tt(gt_data[i].cpu()), cmap='gray')
            axes[i].set_title(f"GT label is {gt_labels[i].item()}")
    plt.tight_layout()
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    gt_image = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close(fig1)

    # # Plot ground truth images
    # fig, axes = plt.subplots(1, batch_size, figsize=(10, 5))
    # if batch_size == 1:
    #     axes.imshow(tt(gt_data[0].cpu()), cmap='gray')
    #     axes.set_title(f"GT label is {gt_labels[0].item()}")
    # else:
    #     for i in range(batch_size):
    #         axes[i].imshow(tt(gt_data[i].cpu()), cmap='gray')
    #         axes[i].set_title(f"GT label is {gt_labels[i].item()}")
    # plt.show()

    # Initialize the network
    if model_class is None:
        net = LeNet().to(device)
    else:
        net = model_class().to(device)
    net.apply(weights_init)
    criterion = cross_entropy_for_onehot

    # Compute original gradient
    out = net(gt_data)
    y = criterion(out, gt_onehot_labels)
    dy_dx = torch.autograd.grad(y, net.parameters())

    # Add noise to gradients
    noisy_dy_dx = add_noise_to_gradients(list((_.detach().clone() for _ in dy_dx)), noise_type=noise_type, noise_strength=noise_strength)

    # Generate dummy data and label
    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    dummy_label = torch.randn(gt_onehot_labels.size()).to(device).requires_grad_(True)

    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

    # Only save the images at every 10th iteration
    history = []
    mse_values = []
    psnr_values = []
    for iters in range(300):
        def closure():
            optimizer.zero_grad()

            pred = net(dummy_data)
            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
            dummy_loss = criterion(pred, dummy_onehot_label)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

            grad_diff = 0
            grad_count = 0
            for gx, gy in zip(dummy_dy_dx, noisy_dy_dx):
                grad_diff += ((gx - gy) ** 2).sum()
                grad_count += gx.nelement()
            grad_diff.backward()

            return grad_diff

        optimizer.step(closure)
        if iters % 10 == 0:
            current_loss = closure()
            print(iters)
            # Calculate metrics
            mse = torch.mean((dummy_data - gt_data) ** 2).item()
            psnr = -10 * np.log10(mse) if mse != 0 else float('inf')
            mse_values.append(mse)
            psnr_values.append(psnr)
            # Append the image to history
            history.append([tt(dummy_data[i].cpu()) for i in range(batch_size)])

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    ax1.set_title("Evolution of Dummy Data Over Iterations (Every 10 Iterations)")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("PSNR")
    ax1.plot(range(10, 301, 10), psnr_values, label='PSNR', color='blue')
    ax1.set_xticks(range(10, 301, 10))
    ax1.set_xticklabels([f'{i*10}' for i in range(1, 31)], rotation=45)
    ax1.set_ylim(0, max(psnr_values) + 10)
    ax1.grid(True)

    ax2.plot(range(10, 301, 10), mse_values, label='MSE', color='red', marker='o')
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Metric Value")
    ax2.set_title("MSE and PSNR Metrics")
    ax2.set_xticks(range(10, 301, 10))
    ax2.set_xticklabels([f'{i*10}' for i in range(1, 31)], rotation=45)
    ax2.set_ylim(0, max(mse_values) + 1)
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    metrics_image = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close(fig2)

    dummy_data_images = []
    for i in range(batch_size):
        fig, axes = plt.subplots(3, 10, figsize=(30, 10))
        for j in range(3):
            for k in range(10):
                idx = j * 10 + k
                if idx < len(history):
                    axes[j, k].imshow(history[idx][i], cmap='gray')
                    axes[j, k].axis('off')
                    axes[j, k].set_title(f'{(idx + 1) * 10}th', fontsize=12)
        plt.suptitle(f"Evolution of Dummy Data for Image {i+1} Over Iterations (Every 10 Iterations)", fontsize=16)
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        dummy_data_images.append(base64.b64encode(img.getvalue()).decode('utf-8'))
        plt.close(fig)

    return gt_image, metrics_image, dummy_data_images

@app.route('/protection', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        batch_size = int(request.form['batch_size'])
        noise_type = request.form['noise_type']
        noise_strength = float(request.form['noise_strength'])
        dataset_type = request.form['dataset_type']

        gt_image, metrics_image, dummy_data_images = run_experiment(batch_size, noise_type, noise_strength, dataset_type)

        return render_template('index2.html', gt_image=gt_image, metrics_image=metrics_image, dummy_data_images=dummy_data_images)
    
    return render_template('index2.html')

@app.route('/compare_protection', methods=['GET', 'POST'])
def compare():
    if request.method == 'POST':
        batch_size1 = int(request.form['batch_size1'])
        noise_type1 = request.form['noise_type1']
        noise_strength1 = float(request.form['noise_strength1'])
        dataset_type1 = request.form['dataset_type1']

        batch_size2 = int(request.form['batch_size2'])
        noise_type2 = request.form['noise_type2']
        noise_strength2 = float(request.form['noise_strength2'])
        dataset_type2 = request.form['dataset_type2']

        gt_image1, metrics_image1, dummy_data_images1 = run_experiment(batch_size1, noise_type1, noise_strength1, dataset_type1)
        gt_image2, metrics_image2, dummy_data_images2 = run_experiment(batch_size2, noise_type2, noise_strength2, dataset_type2)

        return render_template('compare_protection.html', 
                               gt_image1=gt_image1, metrics_image1=metrics_image1, dummy_data_images1=dummy_data_images1,
                               gt_image2=gt_image2, metrics_image2=metrics_image2, dummy_data_images2=dummy_data_images2,
                               batch_size1=batch_size1, noise_type1=noise_type1, noise_strength1=noise_strength1, dataset_type1=dataset_type1,
                               batch_size2=batch_size2, noise_type2=noise_type2, noise_strength2=noise_strength2, dataset_type2=dataset_type2)

    return render_template('compare_protection.html')

if __name__ == '__main__':
    app.run(debug=True)
