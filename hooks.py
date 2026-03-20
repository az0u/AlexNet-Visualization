import torch
import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import streamlit as st

# Innitialize dictionary to store activations
activations = {}

# Get the activation for a specific layer.
def get_activation(name):

    # Store the output of the layer in the activations dictionary.
    def hook(model, input, output):
        activations[name] = output.detach().cpu()
    return hook

# Attach hooks to the model layers to capture activations.
def attach_hooks(model, layer_type):

    hooks = []
    for name, layer in model.named_modules():
        # Map string layer type to actual PyTorch classes
        if layer_type == "All":
            hooks.append(layer.register_forward_hook(get_activation(name)))
        elif layer_type == "Convolutional" and isinstance(layer, torch.nn.Conv2d):
            hooks.append(layer.register_forward_hook(get_activation(name)))
        elif layer_type == "ReLU" and isinstance(layer, torch.nn.ReLU):
            hooks.append(layer.register_forward_hook(get_activation(name)))
        elif layer_type == "MaxPool" and isinstance(layer, torch.nn.MaxPool2d):
            hooks.append(layer.register_forward_hook(get_activation(name)))
    return hooks

# Visualize activations of selected layers (or all)
def visualize_layers(model, layer_type, layer_names, input, num_features=10, max_cols=10): 

    activations.clear() # Clear previous activations
    hooks = attach_hooks(model, layer_type) # Attach hooks

    with torch.no_grad():
        _ = model(input)

    for layer_name, activation in activations.items():
        if activation.ndim != 4:
            continue  # Skip non-4D activations
        if layer_name not in layer_names:
            continue  # skip unmapped layers
        
        act = activation[0]  # first layer in batch
        total_features = act.shape[0] # total number of feature maps
        n_rows = math.ceil(num_features / max_cols) # number of rows needed
        
        fig, axes = plt.subplots(n_rows, max_cols, figsize=(3*max_cols, 3*n_rows)) 
        axes = axes.flatten()  # flatten in case of multiple rows
        
        for i in range(num_features):
            fmap = act[i].numpy()
            axes[i].imshow(fmap, cmap='viridis')
            axes[i].axis('off')
        
        st.subheader(f"Layer: {layer_names[layer_name]}")
        st.pyplot(fig)
        plt.close(fig)

        # "Show more" button
        if num_features < total_features:
            if st.button(f"Show more feature maps for {layer_names.get(layer_name, layer_name)}"):
                st.markdown(f"### All Feature Maps for {layer_names[layer_name]}")
                n_rows_full = math.ceil(total_features / max_cols)
                fig_full, axes_full = plt.subplots(n_rows_full, max_cols, 
                                                   figsize=(3 * max_cols, 3 * n_rows_full))
                axes_full = axes_full.flatten()

                for i in range(total_features):
                    fmap = act[i].numpy()
                    axes_full[i].imshow(fmap, cmap='viridis')
                    axes_full[i].axis('off')

                st.pyplot(fig_full)
                plt.close(fig_full)

    # Remove hooks to avoid memory leaks
    for h in hooks:
        h.remove()