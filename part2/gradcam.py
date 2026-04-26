import torch
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_image, class_idx):
        # Forward pass
        output = self.model(input_image)
        self.model.zero_grad()

        # Backward pass for the specific class
        loss = output[0, class_idx]
        loss.backward()

        # Weight the channels by the gradient average
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()

        # ReLU and Normalization
        cam = np.maximum(cam.detach().cpu().numpy(), 0)
        cam_max = np.max(cam)
        if cam_max > 0: # Safety check to avoid division by zero
            cam = cam / cam_max
        return cam
