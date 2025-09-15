import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from wrappers.base_wrapper import BaseWrapper
from wrappers.torch_wrapper import TorchWrapper
from wrappers.tf_wrapper import TFWrapper

class GradCAM:
    def __init__(self, wrapped_model: BaseWrapper, target_layer = None):
        self.model = wrapped_model
        self.target_layer = target_layer or self.model.get_last_conv_layer()

    def generate_heatmap(self, input_data, class_index = None):
        if isinstance(self.model, TorchWrapper):
            return self._torch_gradcam(input_data, class_index)
        elif isinstance(self.model, TFWrapper):
            return self._tf_gradcam(input_data, class_index)
        else:
            return NotImplementedError("GradCAM not supported for this backend")
        
    def overlay_heatmap(self, heatmap, image, alpha = 0.4, show = True, cmap = "jet"):
        heatmap = np.heatmap(heatmap, 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) > 0 else 1

        heatmap_resized = np.array(
            Image.fromarray(np.uint8(255 * heatmap)).resize((image.shape[1], image.shape[0]))
        ) / 255.0

        colormap = plt.get_cmap(cmap)
        heatmap_color = colormap(heatmap_resized)[..., :3]

        overlay = (image * (1 - alpha) + heatmap_color * 255 *alpha).astype(np.uint8)
        
        if show:
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            ax[0].set_title("Original")
            ax[0].imshow(image[..., ::-1])  # BGR -> RGB
            ax[0].axis("off")

            ax[1].set_title("Heatmap")
            ax[1].imshow(heatmap_resized, cmap = cmap)
            ax[1].axis("off")

            ax[2].set_title("Overlay")
            ax[2].imshow(overlay)
            ax[2].axis("off")

            plt.show()

        return overlay
    
    def _torch_gradcam(self, input_data, class_index = None):
        import torch

    def _tf_gradcam(self):
        pass