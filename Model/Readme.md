## Folder Structure

### 1. `mapmatching/`
This folder contains all the code related to **map matching**. Map matching is a crucial step for aligning the GPS trajectories of the vehicles with the road network. This folder includes scripts and functions to:
- Perform map matching on raw GPS data.
- Extract and match trajectories to road segments.
- Use external map matching tools and libraries (e.g., Barefoot).

You can run the map matching process by executing the provided Julia scripts in this folder. Make sure you have set up the map matching server as described in the **Preprocessing** section of the README.

### 2. `ProbETA/`
This folder contains the core code for **travel time prediction**. It implements the **ProbETA** model, which uses link representation learning for probabilistic travel time estimation. In this folder, you will find:
- Training scripts to train the model with the prepared data.
- Model definition and training routines.
- Evaluation scripts for testing the trained model's performance.
- Hyperparameter tuning and model optimization.

You will also find the `train.py` script to start the training process, along with the necessary configurations to handle data preprocessing, model training, and testing.

### 3. `Results/`
This folder stores all the **result images** produced during the experiments. These images typically include:
- Training and validation performance curves (e.g., loss curves).
- Model evaluation results (e.g., error metrics, travel time predictions).
- Visualizations of the predicted and ground truth travel times.

These visualizations will help you analyze and compare the model's performance on different datasets.
