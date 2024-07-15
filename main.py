import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import datasets, transforms
from flask import Flask, render_template, request


app = Flask(__name__)

# Helper functions

def label_to_onehot(target, num_classes):
    """Convert class labels to one-hot encoding."""
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def cross_entropy_for_onehot(pred, target):
    """Compute cross-entropy loss for one-hot encoded labels."""
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

def weights_init(m):
    """Initialize the weights of the network."""
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)

class LeNet(nn.Module):
    """Define the LeNet architecture for image classification."""
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        act = nn.Sigmoid  # Activation function
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
            nn.Linear(768, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def run_experiment(dataset_type, batch_size, num_epochs):
    """Run the gradient attack experiment and return the results."""
    # Set the seed for reproducibility
    torch.manual_seed(50)

    # Define dataset path and number of classes
    if dataset_type == 'FashionMNIST':
        dataset_path = "d:/Muzny Zuhair/App/FashionMNIST"
        num_classes = 10
    elif dataset_type == 'Caltech101':
        dataset_path = "d:/Muzny Zuhair/App/Caltech101/caltech101/101_ObjectCategories"
        num_classes = 101
    else:
        raise ValueError("Invalid dataset type. Choose 'FashionMNIST' or 'Caltech101'.")

    # Define data transformations
    tp = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1) if dataset_type == 'Caltech101' else transforms.Lambda(lambda x: x)
    ])
    tt = transforms.ToPILImage()

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    # Load the dataset
    if dataset_type == 'Caltech101':
        dst = datasets.ImageFolder(dataset_path, transform=tp)
    else:
        dst = datasets.FashionMNIST(dataset_path, train=True, download=True, transform=tp)

    # Get a fixed set of images and labels
    fixed_img_indices = [0, 1, 2, 3, 4, 5, 6, 7]  # Example indices
    fixed_gt_data = torch.stack([dst[i][0] for i in fixed_img_indices]).to(device)
    fixed_gt_labels = torch.Tensor([dst[i][1] for i in fixed_img_indices]).long().to(device)
    fixed_gt_onehot_labels = label_to_onehot(fixed_gt_labels, num_classes=num_classes)

    # Adjust batch size
    gt_data = fixed_gt_data[:batch_size]
    gt_labels = fixed_gt_labels[:batch_size]
    gt_onehot_labels = fixed_gt_onehot_labels[:batch_size]

    # Initialize the network
    net = LeNet(num_classes=num_classes).to(device)
    net.apply(weights_init)
    criterion = cross_entropy_for_onehot

    # Compute original gradient
    out = net(gt_data)
    y = criterion(out, gt_onehot_labels)
    dy_dx = torch.autograd.grad(y, net.parameters())

    # Share the gradients with other clients
    original_dy_dx = list((_.detach().clone() for _ in dy_dx))

    # Generate dummy data and label
    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    dummy_label = torch.randn(gt_onehot_labels.size()).to(device).requires_grad_(True)

    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

    # Initialize lists for metrics and images
    history = []
    mse_values = []
    psnr_values = []

    for iters in range(num_epochs):
        def closure():
            """Closure function for LBFGS optimizer."""
            optimizer.zero_grad()

            pred = net(dummy_data)
            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
            dummy_loss = criterion(pred, dummy_onehot_label)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()

            return grad_diff

        optimizer.step(closure)
        if iters % 10 == 0:
            # Calculate metrics
            mse = torch.mean((dummy_data - gt_data) ** 2).item()
            psnr = -10 * np.log10(mse) if mse != 0 else float('inf')
            mse_values.append(mse)
            psnr_values.append(psnr)

            # Save the image at every 10th iteration
            history.append([tt(dummy_data[i].cpu()) for i in range(batch_size)])

    # Prepare the plots
    mse_values = np.array(mse_values)
    psnr_values = np.array(psnr_values)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))  # Two subplots: one for images, one for metrics

    # Plot the evolution of dummy data
    ax1.set_title("Evolution of Dummy Data Over Iterations (Every 10 Iterations)")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("PSNR")
    ax1.plot(range(10, num_epochs + 1, 10), psnr_values, label='PSNR', color='blue')
    ax1.set_xticks(range(10, num_epochs + 1, 10))
    ax1.set_xticklabels([f'{i*10}' for i in range(1, len(psnr_values) + 1)], rotation=45)
    ax1.set_ylim(0, max(psnr_values) + 10)
    ax1.grid(True)

    # Plot the MSE and PSNR metrics
    ax2.plot(range(10, num_epochs + 1, 10), mse_values, label='MSE', color='red', marker='o')
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Metric Value")
    ax2.set_title("MSE and PSNR Metrics")
    ax2.set_xticks(range(10, num_epochs + 1, 10))
    ax2.set_xticklabels([f'{i*10}' for i in range(1, len(mse_values) + 1)], rotation=45)
    ax2.set_ylim(0, max(mse_values) + 1)
    ax2.grid(True)
    ax2.legend()

    # Save plots to in-memory images
    img_bytes = io.BytesIO()
    plt.tight_layout()
    plt.savefig(img_bytes, format='png')
    img_bytes.seek(0)
    plots_image = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
    plt.close()

    # Plot dummy data images separately for each original image
    img_html_list = []
    for i in range(batch_size):
        fig, axes = plt.subplots(3, 10, figsize=(30, 10))  # 3 rows and 10 columns
        for j in range(3):
            for k in range(10):
                idx = j * 10 + k
                if idx < len(history):
                    axes[j, k].imshow(history[idx][i], cmap='gray')
                    axes[j, k].axis('off')
                    axes[j, k].set_title(f'{(idx + 1) * 10}th', fontsize=12)  # Set font size for iteration labels
        plt.suptitle(f"Evolution of Dummy Data for Image {i+1} Over Iterations (Every 10 Iterations)", fontsize=16)
        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format='png')
        img_bytes.seek(0)
        img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
        img_html_list.append(img_base64)
        plt.close()

    mse_value = mse_values[-1] if len(mse_values) > 0 else 0  # Get the last MSE value as a representative metric

    return img_html_list, plots_image, mse_value

# Routes
@app.route('/compare', methods=['GET', 'POST'])
def compare():
    """Render the compare page and handle form submissions for comparing two attacks."""
    if request.method == 'POST':
        dataset_type1 = request.form['dataset_type1']
        batch_size1 = int(request.form['batch_size1'])
        num_epochs1 = int(request.form['num_epochs1'])
        dataset_type2 = request.form['dataset_type2']
        batch_size2 = int(request.form['batch_size2'])
        num_epochs2 = int(request.form['num_epochs2'])

        try:
            img_html_list1, plots_image1, mse_value1 = run_experiment(dataset_type1, batch_size1, num_epochs1)
            img_html_list2, plots_image2, mse_value2 = run_experiment(dataset_type2, batch_size2, num_epochs2)

            return render_template('compare.html', 
                                   img_html_list1=img_html_list1, 
                                   plots_image1=plots_image1, 
                                   mse_value1=mse_value1,
                                   img_html_list2=img_html_list2, 
                                   plots_image2=plots_image2, 
                                   mse_value2=mse_value2)

        except Exception as e:
            return f"An error occurred: {str(e)}", 500

    return render_template('compare.html', img_html_list1=None, img_html_list2=None)


@app.route('/', methods=['GET', 'POST'])
def index():
    """Render the main form for user input."""
    if request.method == 'POST':
        dataset_type = request.form['dataset_type']
        batch_size = int(request.form['batch_size'])
        num_epochs = int(request.form['num_epochs'])
        try:
            img_html_list, plots_image, mse_value = run_experiment(dataset_type, batch_size, num_epochs)
            return render_template('index.html', plots_image=plots_image, img_html_list=img_html_list, mse_value=mse_value, dataset_type=dataset_type, batch_size=batch_size, num_epochs=num_epochs)
        except Exception as e:
            return f"An error occurred: {str(e)}", 500
    return render_template('index.html')



#### privacy




if __name__ == '__main__':
    app.run(debug=True)
