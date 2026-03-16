import torch
import torchvision.models as models
import torchvision.transforms as T
import numpy as np
from PIL import Image
from EXACT.explainers import SaliencyMap


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval().to(device)

    pil_img = Image.open("models/catexample.jpg").convert("RGB")
    img_np  = np.array(pil_img.resize((224, 224)), dtype=np.uint8)

    preprocess = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(pil_img).unsqueeze(0).to(device)

    explainer = SaliencyMap(model=model)

    vanilla_result = explainer.explain(input_tensor, input_image=img_np, method="vanilla",    save_png=True)
    guided_result  = explainer.explain(input_tensor, input_image=img_np, method="guided",     save_png=True)
    smooth_result  = explainer.explain(input_tensor, input_image=img_np, method="smoothgrad", save_png=True, n_samples=30)

    for name, result in [("vanilla", vanilla_result), ("guided", guided_result), ("smoothgrad", smooth_result)]:
        print(f"{name}: class={result['target_class']}  heatmap={result['heatmap'].shape}  saved={result['filepath']}")


if __name__ == "__main__":
    test()