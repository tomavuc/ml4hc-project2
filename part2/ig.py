import torch

def integrated_gradients(inputs, model, target_label_idx, device, steps=500, internal_batch_size=8):
    """Computes Integrated Gradients for a specific target class using VRAM-friendly chunking."""
    baseline = torch.zeros_like(inputs).to(device)

    # This will hold our running total of gradients across all steps
    total_grads = torch.zeros_like(inputs[0]).to(device)

    for start_idx in range(0, steps + 1, internal_batch_size):
        end_idx = min(start_idx + internal_batch_size, steps + 1)
        
        # 1. Generate just the scaled images for this specific batch
        batch_alphas = [float(i) / steps for i in range(start_idx, end_idx)]
        scaled_inputs = [baseline + alpha * (inputs - baseline) for alpha in batch_alphas]
        scaled_inputs = torch.cat(scaled_inputs).requires_grad_(True)

        # 2. Forward pass for the small batch
        outputs = model(scaled_inputs)
        probs = torch.softmax(outputs, dim=1)[:, target_label_idx]

        # 3. Backward pass to get gradients
        model.zero_grad()
        probs.backward(torch.ones_like(probs))

        # 4. Accumulate the gradients (Sum them up along the batch dimension)
        # We use .data to avoid holding onto the computational graph memory!
        total_grads += scaled_inputs.grad.data.sum(dim=0)

    # Reshape and average gradients across all 501 steps
    avg_grads = total_grads / (steps + 1)

    # Integrated Gradients = (input - baseline) * avg_grad
    integrated_grad = (inputs[0] - baseline[0]) * avg_grads
    
    return integrated_grad.cpu().detach().numpy()