import torch
from torchvision.models import alexnet, AlexNet_Weights
from captum.attr import IntegratedGradients, NoiseTunnel, GuidedGradCam
from captum.attr import visualization as vis
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')
import streamlit as st


def GradCAM_(model, inputs, target):

    guided_gradcam = GuidedGradCam(model, model.features[10])  # AlexNet last conv layer
    inputs.requires_grad_() # Ensure gradients are kept for input 

    # Set to zero existing gradients to avoid overalap 
    if inputs.grad is not None:
        inputs.grad.zero_()

    # Zero out the gradients of the model 
    # to nsure a clean gradient computation
    model.zero_grad()

    attribution = guided_gradcam.attribute(
        inputs,
        target=target
    )

    return attribution


def visualize_attribution(attribution, original_image, image_std, image_mean):

    vis.visualize_image_attr_multiple(
        attribution.squeeze(0).permute(1, 2, 0).detach().numpy(), 
        original_image=(
            original_image.squeeze(0).permute(1, 2, 0).detach().numpy()
            * image_std
            + image_mean
        ),
        methods=["original_image", "heat_map", "heat_map", "heat_map"],
        signs=["all", "positive", "negative", "all"],
        alpha_overlay=0.5,
        show_colorbar=True,
        fig_size=(18, 6)
    )


def overlay_attribution(
    attribution, input_image, cmap="Reds", alpha=0.5,
    percentile=50, image_mean=0.1307, image_std=0.3081,
    filepath="./attribution_overlay.png"
):

    attr_np = attribution.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

    threshold = np.percentile(attr_np, percentile) # neglect low attribution values
    attr_np = np.where(attr_np > threshold, attr_np, 0) # apply threshold

    unnormalized_image = input_image * image_std + image_mean 
    input_np = unnormalized_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() 

    fig, _ = vis.visualize_image_attr(
        attr_np,
        input_np,
        method="blended_heat_map",
        sign="absolute_value",
        alpha_overlay=alpha,
        cmap=cmap
    )

    fig.tight_layout(pad=0)

    # Save the figure as a file
    fig.savefig(filepath)