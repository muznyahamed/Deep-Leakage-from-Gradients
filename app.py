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

from main import run_experiment
from protection import run_experiment as run_experiment2  # Import the specific function from protection.py

app = Flask(__name__)

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
            # Call the run_experiment function from main.py
            img_html_list1, plots_image1, mse_value1 = run_experiment(dataset_type1, batch_size1, num_epochs1)
            # Call the run_experiment function from protection.py
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
@app.route('/', methods=['GET'])
def landing():
    """Render the landing page."""
    return render_template('landing.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    """Render the main form for user input."""
    if request.method == 'POST':
        dataset_type = request.form['dataset_type']
        batch_size = int(request.form['batch_size'])
        num_epochs = int(request.form['num_epochs'])
        try:
            # Call the run_experiment function from main.py
            img_html_list, plots_image, mse_value = run_experiment(dataset_type, batch_size, num_epochs)
            return render_template('index.html', plots_image=plots_image, img_html_list=img_html_list, mse_value=mse_value, dataset_type=dataset_type, batch_size=batch_size, num_epochs=num_epochs)
        except Exception as e:
            return f"An error occurred: {str(e)}", 500
    return render_template('index.html')


@app.route('/protection', methods=['GET', 'POST'])
def index2():
    if request.method == 'POST':
        batch_size = int(request.form['batch_size'])
        noise_type = request.form['noise_type']
        noise_strength = float(request.form['noise_strength'])
        dataset_type = request.form['dataset_type']

        gt_image, metrics_image, dummy_data_images = run_experiment2(batch_size, noise_type, noise_strength, dataset_type)

        return render_template('index2.html', gt_image=gt_image, metrics_image=metrics_image, dummy_data_images=dummy_data_images)
    
    return render_template('index2.html')

@app.route('/compare_protection', methods=['GET', 'POST'])
def compare2():
    if request.method == 'POST':
        batch_size1 = int(request.form['batch_size1'])
        noise_type1 = request.form['noise_type1']
        noise_strength1 = float(request.form['noise_strength1'])
        dataset_type1 = request.form['dataset_type1']

        batch_size2 = int(request.form['batch_size2'])
        noise_type2 = request.form['noise_type2']
        noise_strength2 = float(request.form['noise_strength2'])
        dataset_type2 = request.form['dataset_type2']

        gt_image1, metrics_image1, dummy_data_images1 = run_experiment2(batch_size1, noise_type1, noise_strength1, dataset_type1)
        gt_image2, metrics_image2, dummy_data_images2 = run_experiment2(batch_size2, noise_type2, noise_strength2, dataset_type2)

        return render_template('compare_protection.html', 
                               gt_image1=gt_image1, metrics_image1=metrics_image1, dummy_data_images1=dummy_data_images1,
                               gt_image2=gt_image2, metrics_image2=metrics_image2, dummy_data_images2=dummy_data_images2,
                               batch_size1=batch_size1, noise_type1=noise_type1, noise_strength1=noise_strength1, dataset_type1=dataset_type1,
                               batch_size2=batch_size2, noise_type2=noise_type2, noise_strength2=noise_strength2, dataset_type2=dataset_type2)

    return render_template('compare_protection.html')

if __name__ == '__main__':
    app.run(debug=True)
