## Folder Structure

### 1. `MapMatching/`
This folder contains all the code related to **map matching**. Map matching is a crucial step for aligning the GPS trajectories of the vehicles with the road network. This folder includes scripts and functions to:
- Perform map matching on raw GPS data.
- Extract and match trajectories to road segments.

You can run the map matching process by executing the provided Julia scripts in this folder.

### 2. `ProbETA/`
This folder contains the core code for **travel time prediction**. It implements the **ProbETA** model, which uses link representation learning for probabilistic travel time estimation. 

You will also find the `main.py` script to start the training and testing process.

### 3. `Result/`
This folder stores all the **result images** produced during the experiments. These images typically include:
- Training and validation performance curves.
- Inter and intra correlation visualization.
- Hyperparameter ablation experiment results.

