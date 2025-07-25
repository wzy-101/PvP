# PvP: Probing Varied Viewpoints for Personalized News Recommendation [cite: 1]
## 🚀 Getting Started

1.  **Clone the repository**
    ```bash
    git clone https://anonymous.4open.science/r/PvP-23BE/PvP/src/models/PvP.py
    cd PvP
    ```

2.  **Create an environment and install dependencies**
    > **Note**: It is highly recommended to use a virtual environment (e.g., Conda or venv).
    ```bash
    # Example using Conda
    conda create -n pvp_env python=3.8
    conda activate pvp_env
    pip install -r requirements.txt
    ```

3.  **Prepare Datasets**
    This project was evaluated on the **MIND-Small** and **MIND-Large** real-world datasets[cite: 172]. Please download the necessary data and place it in the appropriate directory.
    Download MIND dataset. The directory structure will be like below.


## 💡 Usage

**Model Training:**
```bash
# Run the training script with a specific config file
$ cd src
$ python train-PvP.py params/main/002.yaml
