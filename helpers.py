import numpy as np
import PIL


def inv_normalize(tensor):
    """Normalize an image tensor back to the 0-255 range."""
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min()) * (256 - 1e-5)
    return tensor


def inv_transform(tensor, normalize=True):
    """Convert a tensor to an image."""
    if normalize:
        tensor = inv_normalize(tensor)
    array = tensor.detach().cpu().numpy()
    array = array.transpose(1, 2, 0).astype(np.uint8)
    return PIL.Image.fromarray(array)


def apply_mask(input, patch_weights):
    """Display the image, dimming each patch according to the given weight."""
    # Multiply each patch of the input image by the corresponding weight
    plot = inv_normalize(input.clone())
    assert patch_weights.shape == (196,)
    for i in range(patch_weights.shape[0]):
        x = i * 16 % 224
        y = i // (224 // 16) * 16
        plot[:, y:y + 16, x:x + 16] *= patch_weights[i]
    return inv_transform(plot)