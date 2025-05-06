# 🧠 Project: Modular Experiment Framework (Game + Stat + AI)

This project is a modular, batch-driven framework designed to run experiments combining a Snake game environment, a statistical preprocessing model (e.g., Bayesian), and an AI model (Genetic Algorithm). It supports full parameter tracking, batch versioning, and easy extension to new models.

---

## 🔄 Key Concepts

* **Experiment-centric**: Every run is controlled by a config file.
* **Batch-based learning**: Each run executes multiple Snake games (e.g., 50) as a batch.
* **Parameter-aware**: Each parameter set has its own result folder.
* **Modular pipeline**: Combines game → statistical model → AI model.
* **Extendable**: Easily add more AI or statistical models in the future.

---

## 📂 Project Structure

```bash
project-root/
│
├── main.py                         # Entry point: starts experiment
│
├── game/
│   ├── environment.py              # Snake game logic
│   ├── actions.py                  # Action definitions
│   ├── state.py                    # State representation
│   └── visualizer.py               # Game visualization
│
├── genetic/
│   ├── agent.py                    # Defines GA agent behavior
│   ├── fitness.py                  # Fitness score calculator
│   ├── population.py               # GA population mechanics
│   └── trainer.py                  # Main GA training logic (uses stat features)
│
├── statistics/
│   ├── interface.py                # Entry point for stat model selection
│   ├── bayesian.py                 # Bayesian inference and feature extraction
│   └── normalization.py            # Optional normalization functions
│
├── experiments/
│   ├── executor.py                 # Loads config, starts experiment, manages runs
│   ├── runner.py                   # Executes game, stat model, and AI training
│   ├── model_selector.py           # Dynamically selects stat + AI models
│   └── configs/
│       ├── exp_genetic.yaml        # Config for GA + Bayesian
│       └── exp_gridsearch.yaml     # (Optional) Looping over parameter combinations
│
├── results/
│   └── {experiment_name}/
│       └── {param_key}/        # e.g., lr_0.01_bs_50_mom_0.9
│           ├── run_001/
│           │   ├── raw_data.json         # Raw results from 50 games
│           │   ├── stat_input.json       # Preprocessed data for stat model
│           │   ├── stat_output.json      # Output features for AI model
│           │   ├── summary.yaml          # Statistical summary
│           │   └── plot.png              # (Optional) learning curve
│           └── run_002/           # Second batch with same parameters
```

---

## 📚 Sample Config File (`exp_genetic.yaml`)

```yaml
experiment_name: "genetic_bayes_run"

# Models
ai_model: "genetic"
stat_model: "bayesian"

# Game + batch
batch_size: 50

# AI hyperparameters
learning_rate: 0.01
momentum: 0.9
population_size: 10
num_generations: 5

# Stat model config
bayesian:
  prior: uniform
  confidence_interval: 95
  use_entropy: true
```

---

## 🔄 Run Workflow

```text
main.py
  → experiments/executor.py
      → Load config
      → Create param-based run folder
      → Call runner.py:
          • Run game trials (batch of 50) → raw_data.json
          • Apply stat model              → stat_output.json
          • Train AI on stat data         → save scores & summary
```

---

## 🔧 Execution Instructions (for Subcontractor)

1. **Clone or open the project repository** in your development environment.

2. **Ensure dependencies are installed**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up the config file**:

   * Navigate to `experiments/configs/`
   * Open `exp_genetic.yaml`
   * Set the desired values:

     * `batch_size: 50` → This is mandatory
     * Adjust `learning_rate`, `momentum`, etc., as required

4. **Run the main experiment**:

   ```bash
   python main.py
   ```

5. **System behavior after launch**:

   * Game plays 50 trials as a batch
   * Output from each game is saved to `raw_data.json`
   * Statistical model (e.g., Bayesian) processes this into:

     * `stat_output.json` (used as AI input)
     * `summary.yaml` (for reporting)
   * Genetic Agent uses the stat features for training
   * All results are saved under:

     ```
     results/{experiment_name}/{parameter_key}/run_XXX/
     ```
   * A new `run_XXX/` folder is created each time even if parameters don’t change

6. **Output includes**:

   * Raw game data
   * Stat-processed features
   * AI performance data
   * Summary statistics
   * Visual plots (if configured)

7. **Important Notes**:

   * Ensure each `run_XXX` contains exactly 50 trials
   * Each statistical model may extract different features. Ensure all raw data is saved fully so the stat model can select what it needs
   * The AI model must only consume `stat_output.json`, not `raw_data.json`

---

## ✅ Features

| Feature             | Description                                               |
| ------------------- | --------------------------------------------------------- |
| ✅ Modular design    | Game, Stat model, and AI model separated                  |
| ✅ Parameter control | All models configured through YAML files                  |
| ✅ Batch-aware       | Each experiment run = one batch of game trials (e.g., 50) |
| ✅ Result tracking   | Auto-saves raw data, stat output, AI performance          |
| ✅ Reproducible      | Full output and configs saved for every run               |

---

## 🤝 How to Start

```bash
python main.py
```

This will:

* Load the config from `experiments/configs/exp_genetic.yaml`
* Run 50 Snake games
* Run the Bayesian stat model
* Train the Genetic agent using stat features
* Save everything in `results/` automatically

---

## 📊 Future Extensions

* Add more AI models (e.g., Reinforcement Learning)
* Add more stat models (e.g., regression, PCA)
* Enable grid search for parameter sweeps
* Add advanced visualizations and dashboards
