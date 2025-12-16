# 🎮 AgentEvolver Game Arena

AgentEvolver Game Arena is a platform for **training, evaluating, and observing AI agents in multi-agent social board games**.  
It focuses on games with **hidden information, negotiation, deception, and cooperation**, enabling users to study and compare agent behavior in complex social decision-making environments.

The platform supports **web-based gameplay and observation**, **batch model evaluation**, and **agent training**, making it suitable for researchers, model evaluators, and AI enthusiasts.

---

## ✨ What Can You Do?

### 1. Watch or Play AI Agents in Your Browser

A web-based interface allows you to:

- **Observer Mode** – Watch AI agents play in real time and inspect their reasoning, communication, and strategic decisions  
- **Participate Mode** – Join a game as a human player and play alongside or against AI agents to test their interactive capabilities  

**Supported Games**

- **Avalon** – Hidden roles, logical inference, and deception  
- **Diplomacy** – Negotiation, alliances, and long-term strategy  

> 📸 *Web interface screenshots and gameplay animations will be added here.*

---

### 2. Evaluate AI Models at Scale

AgentEvolver provides a built-in evaluation framework to **systematically compare model performance**:

- Run multiple games in parallel for statistically meaningful results  
- Control game settings and model assignments via configuration files  

> 📊 *Example evaluation summaries and visualized results will be added here.*

---

### 3. Train AI Agents

AgentEvolver is designed to support **end-to-end training of AI agents in social board games**, enabling agents to learn from interaction, feedback, and long-horizon outcomes.

- Training agents directly within game environments  
- Support for reinforcement learning–based methods (e.g., GRPO)  

> 📈 *Training curves, learning dynamics, and performance evolution plots will be added here.*

---

## 🚀 Getting Started

### Install Dependencies

Run:

    pip install -r requirements.txt

(Optional) Set environment variables:

    export API_KEY=your_api_key
    export MODEL_NAME=your_default_model

---

### Launch the Web Interface (Recommended)

Start the server:

    python games/web/server.py

Then open your browser at:

    http://localhost:8000

From the web interface you can:

1. Select a game (Avalon or Diplomacy)  
2. Choose a mode (Observer or Participate)  
3. Configure players and models  
4. Start the game  

---

### Run a Model Evaluation

Example command:

    python games/evaluation/run_eval.py \
        --game avalon \
        --config games/games/avalon/configs/task_config.yaml \
        --num-games 10

After completion, a summary similar to the following will be displayed:

    Evaluation Results Summary – AVALON
    total_games: 10
    successful_games: 10
    good_victory_mean: 0.6000
    ...

---

## ⚙️ Configuration

Games and evaluations are controlled via **YAML configuration files**. Configuration structure:

- **Game settings** (`game`) – Game-specific parameters (e.g., `num_players`, `language`)
- **Model configuration** – Priority order:
  1. **Role-specific settings** (`roles` section) – Each role uses its own configuration if specified
  2. **Default model settings** (`default_model`) – Used as fallback when a role's configuration is missing

Example:

    game:
      name: avalon
      num_players: 5
      language: en
    
    default_model:
      model_name: qwen-plus
      temperature: 0.7
    
    roles:
      assassin:
        model_name: custom-model  # assassin uses custom-model, others use qwen-plus  

---

## 📚 Additional Documentation

- Web interface details: games/web/README.md  
- Training and algorithm details: see the main project documentation  

---

## 📄 License

Please refer to the main project LICENSE file.