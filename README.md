# EXACT
A plug-and-play XAI library for pytorch models.
Meant for beginners and anyone interested in venturing into the XAI space, this package enables users to use any of the supported explainability methods with very few lines of code.

## Key Functionalities
 - Plug-and-play with the user's trained models and input data.
 - Specialized evaluators to evaluate quality of generated explainability results.
 - Specialized comparators to compare between mutliple xai methods to find the best one for your needs.
 - All results visualized cleanly and saved locally.

## Setup
As of now you may clone the repo to use EXACT as PyPI deployment will be done later.
```bash
git clone "https...."
cd E.X.A.C.T-BY-GHAN
```

Install the dependencies
```bash
python -m venv your_env
source path_to_your_env/bin/activate    #for linux
pip install -r requirements.txt
pip install -e.    #to build the EXACT package.
```

## Usage
All implemented explainers have detailed instruction docstrings and are structures similarly for ease of use.
```bash
from EXACT.explainers import SaliencyMap
explainer = SaliencyMap(model = your_trained_model)
result = explainer.explain(input_tensor, method = "guided", save_png = True)
```

Evaluator usage example
```bash
from EXACT.explainers import GradCAM
from EXACT.evaluators import SharpnessEvaluator

explainer = GradCAM(model = your_trained_model)
result = explainer.explain(input_tensor, input_image, method = "xgradcam")

sharp_ev = SharpnessEvaluator()
sharp_result = sharp_ev.evaluate(explainer_result = result)
sharp_ev.report(sharp_result)
sharp_ev.plot(sharp_result, save_png=True)
```

Comparator usage example
```bash
from EXACT.explainers import GradCAM, IGImageExplainer
from EXACT.comparators import HeatmapComparator

explainer = GradCAM(your_trained_model)
ig_explainer = IGImageExplainer(model = your_trained_model)

gradcam_result = explainer.explain(input_tensor, input_image, method="gradcam", save_png=True)
gradcampp_result = explainer.explain(input_tensor, input_image, method="gradcam++", save_png=True)
ig_result = ig_explainer.explain(input_tensor, input_image, save_png = True)

cmp = HeatmapComparator(model = your_trained_model, device = device, stability_enabled = False) 
results = cmp.compare(
    entries = {
        "GradCAM": (gradcam_result, explainer, {"method": "gradcam"}),
        "GradCAM++": (gradcampp_result, explainer, {"method": "gradcam++"}),
        "IG": (ig_result, ig_explainer, {})
    },
    input_tensor,
    input_image
)
cmp.report(results)
cmp.plot(results, save_png = True)
```

# System overview
## Architecture
```mermaid
flowchart TB
    subgraph Input["Input Layer"]
        A[User Trained Model<br/>+ Input Data]
    end

    subgraph Preprocessing["Data Preprocessing"]
        B[Data Preprocessing<br/>Optional - Per Explainer Needs]
    end

    subgraph Explainers["Explainers Module"]
        C1[Compatibility Ensurer]
        C2[Unique Parameters<br/>Per XAI Method]
        C3[Common Parameters<br/>Across Explainers]
        C[Select XAI Explainer]
    end

    subgraph Comparator["Comparator Module"]
        D1[Compatibility Ensurer]
        D2[Comparison Methods]
        D3[Common Parameters]
        D[Compare Multiple<br/>Explainer Results]
    end

    subgraph Evaluators["Evaluators Module"]
        E1[Compatibility Ensurer]
        E2[Unique Parameters<br/>Per Evaluator]
        E3[Common Parameters]
        E[Select Compatible<br/>Evaluator]
    end

    subgraph Output["Output Layer"]
        F1[Visualization &<br/>Save Results]
        F2[Results Dashboard<br/>Visualization & Save]
        F3[Evaluation Results<br/>Visualization & Save]
    end

    %% Main Flow
    A --> B
    B --> C

    %% Workflow A: Single Explainer Path
    C --> F1
    C -.->|Workflow A:<br/>Single Explainer| F1

    %% Workflow B: Multiple Explainers + Comparator Path
    C --> D
    D --> F2
    C -.->|Workflow B:<br/>Multiple Explainers| D

    %% Workflow C: Explainer + Evaluator Path
    C --> E
    E --> F3
    C -.->|Workflow C:<br/>Explainer + Evaluator| E

    %% Styling
    classDef input fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef process fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef module fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    classDef output fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px

    class A input
    class B process
    class C1,C2,C3,D1,D2,D3,E1,E2,E3 module
    class F1,F2,F3 output
```


## Contribution
Accepting contributions