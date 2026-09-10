#🚀 Autonomous Rocket Lander using Deep Q-Network (DQN)

An autonomous rocket/lunar lander simulation that uses **Deep Reinforcement Learning (DQN)** to learn how to control a lander and perform a stable autonomous landing.

The project is built using **Python, PyTorch, Gymnasium, and Box2D**, with a neural-network-based controller that learns which thruster action to apply based on the current state of the lander.

The trained controller can then be reused to autonomously land new rockets under different initial conditions.

---

## 🎯 Project Objective

The objective is to train an autonomous agent that can:

* Understand the current state of a rocket/lander
* Decide which thruster to activate
* Correct horizontal and vertical movement
* Control the lander's orientation
* Reduce excessive velocity before touchdown
* Achieve a stable landing without human intervention
* Generalize the learned control policy to new landing episodes

The project uses the **Gymnasium LunarLander-v3** environment as a simulation prototype for autonomous rocket landing.

---

## 🧠 Reinforcement Learning Architecture

The project uses a **Deep Q-Network (DQN)**.

The overall control pipeline is:

```text
             ┌─────────────────────┐
             │  Lander Environment │
             └──────────┬──────────┘
                        │
                        ▼
                Current State
                        │
                        ▼
             ┌─────────────────────┐
             │    DQN Neural Net   │
             │                     │
             │  Input: 8 states    │
             │  Hidden: 128        │
             │  Hidden: 128        │
             │  Output: 4 actions  │
             └──────────┬──────────┘
                        │
                        ▼
                  Best Q-value
                        │
                        ▼
              Thruster / Control
                        │
                        ▼
             New Lander State
                        │
                        └──────────────► Repeat
```

---

## 🚀 Environment

The project uses:

```text
LunarLander-v3
```

from Gymnasium.

The lander's state contains 8 values representing information such as:

* Horizontal position
* Vertical position
* Horizontal velocity
* Vertical velocity
* Angle
* Angular velocity
* Left-leg contact
* Right-leg contact

The DQN receives these values as its input.

---

## 🎮 Available Actions

The agent chooses between four discrete actions:

| Action | Control                       |
| -----: | ----------------------------- |
|    `0` | Do nothing                    |
|    `1` | Fire left/orientation engine  |
|    `2` | Fire main engine              |
|    `3` | Fire right/orientation engine |

The trained neural network outputs a Q-value for each action.

The action with the highest Q-value is selected during autonomous operation.

---

## 🧠 DQN Model

The neural network architecture is:

```text
Input Layer
    │
    │ 8 state values
    ▼
Fully Connected Layer
    │
    │ 128 neurons
    ▼
ReLU
    │
    ▼
Fully Connected Layer
    │
    │ 128 neurons
    ▼
ReLU
    │
    ▼
Output Layer
    │
    │ 4 Q-values
    ▼
Actions
```

Implemented using PyTorch:

```python
class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed=42):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, action_size)
        )

    def forward(self, state):
        return self.net(state)
```

---

## 🔄 How the Agent Learns

The agent interacts with the environment repeatedly.

At every step:

```text
Current State
      ↓
Select Action
      ↓
Environment
      ↓
Reward
      ↓
New State
      ↓
Store Experience
      ↓
Train DQN
```

Each experience is stored as:

```text
(state, action, reward, next_state, done)
```

The DQN then learns which actions produce higher future rewards.

---

## 🗃️ Experience Replay

The project uses an experience replay buffer.

Instead of learning only from the most recent action, the agent stores previous experiences and randomly samples batches during training.

This helps reduce correlations between consecutive experiences and improves training stability.

```python
buffer.add(
    state,
    action,
    reward,
    next_state,
    done
)
```

---

## 🎯 Target Network

Two neural networks are used:

```text
Local Network
      │
      ├── Learns continuously
      │
      ▼
Target Network
      │
      └── Provides stable target Q-values
```

The target network is periodically/softly updated from the local network.

This improves DQN training stability.

---

## 🎲 Exploration vs Exploitation

During training, the agent uses **epsilon-greedy exploration**.

Initially:

```text
High ε
   ↓
More random actions
   ↓
Explore environment
```

As training progresses:

```text
Low ε
   ↓
Fewer random actions
   ↓
Use learned policy
```

During autonomous evaluation:

```python
epsilon = 0.0
```

This means the trained controller makes decisions without random exploration.

---

## 🏆 Autonomous Landing

After training, the learned DQN can be reused without retraining.

For example:

```python
checkpoint = torch.load(
    "autonomous_lander_best.pth",
    map_location=device
)

q_local.load_state_dict(
    checkpoint["model_state_dict"]
)

q_local.eval()
```

The same trained controller can then be used for another landing:

```python
state, _ = env.reset(seed=9999)

action = select_action(
    state,
    epsilon=0.0
)
```

The important concept is:

```text
Training Seed ≠ Learned Knowledge
```

A seed only determines the randomness/initial conditions of an episode.

The learned autonomous behavior is contained in the **trained neural-network weights**.

Therefore:

```text
              TRAINED DQN
                   │
       ┌───────────┴───────────┐
       │                       │
       ▼                       ▼
  Seed 1234                Seed 9999
       │                       │
       ▼                       ▼
   Rocket 1                 Rocket 2
       │                       │
       ▼                       ▼
   Landing                  Landing
```

The same trained controller can be evaluated on many different initial conditions.

---

## 🎥 Multiple Autonomous Landings

The project supports generating multiple independent landing videos using different seeds.

Example:

```python
seeds = [
    1234,
    5678,
    9012,
    3456
]
```

Each seed creates a new episode while using the same trained DQN.

This allows the controller to be tested for generalization.

For example:

```text
             SAME TRAINED DQN
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
     Seed 1234   Seed 5678   Seed 9012
        │           │           │
        ▼           ▼           ▼
     Landing      Landing      Landing
```

---

## 📊 Evaluation

The project evaluates the trained agent over multiple episodes.

Metrics include:

* Mean reward
* Median reward
* Best reward
* Number of successful landings

A conventional LunarLander solved threshold of approximately **200 average reward over 100 episodes** is used as the training target.

---

## 📁 Project Structure

Recommended GitHub structure:

```text
Autonomous-Rocket-Lander/
│
├── Autonomous_Rocket_Lander_DQN.ipynb
│
├── autonomous_lander_best.pth
│
├── autonomous_rocket_lander_dqn.pth
│
├── autonomous_lander.mp4
│
├── landing_1.mp4
├── landing_2.mp4
├── landing_3.mp4
│
└── README.md
```

The main notebook contains:

```text
1. Environment Setup
2. DQN Architecture
3. Experience Replay
4. Autonomous Controller
5. Training
6. Training Visualization
7. Model Loading
8. Autonomous Evaluation
9. Landing Video Generation
10. Model Saving
```

---

## 🛠️ Technologies Used

| Technology   | Purpose                            |
| ------------ | ---------------------------------- |
| Python       | Programming                        |
| PyTorch      | Deep Learning                      |
| Gymnasium    | Reinforcement Learning Environment |
| Box2D        | Physics Simulation                 |
| NumPy        | Numerical Computing                |
| Matplotlib   | Training Visualization             |
| ImageIO      | Video Generation                   |
| Google Colab | Training Environment               |

---

## ⚙️ Installation

Install the required packages:

```bash
pip install "gymnasium[box2d]" imageio imageio-ffmpeg torch numpy matplotlib
```

---

## ▶️ Running the Project

### Option 1 — Google Colab

Upload:

```text
Autonomous_Rocket_Lander_DQN.ipynb
```

to Google Colab and run the notebook from top to bottom.

### Option 2 — Local Environment

Clone the repository:

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Jupyter notebook:

```bash
jupyter notebook
```

---

## 💾 Trained Model

The trained model is stored as:

```text
autonomous_lander_best.pth
```

or:

```text
autonomous_rocket_lander_dqn.pth
```

The model contains the learned neural-network parameters.

It can be loaded using:

```python
checkpoint = torch.load(
    "autonomous_lander_best.pth",
    map_location=device
)

q_local.load_state_dict(
    checkpoint["model_state_dict"]
)
```

---

## 📈 Training Parameters

Example configuration:

```text
Episodes              : 1200
Batch Size             : 128
Learning Rate          : 0.0005
Discount Factor (γ)    : 0.99
Replay Buffer          : 100,000
Initial Epsilon        : 1.0
Minimum Epsilon        : 0.02
Epsilon Decay          : 0.995
Target Update (τ)      : 0.001
```

These parameters can be adjusted depending on available GPU resources and training performance.

---

## 🔬 Future Improvements

The current project is a simulation prototype. Future versions can move toward a more realistic autonomous rocket landing system.

Potential improvements include:

### 1. Continuous Control

Replace discrete actions with continuous thrust commands.

```text
Main Engine Throttle
        +
Attitude Control
        +
Lateral Thrust
```

Algorithms such as:

* PPO
* SAC
* TD3

could be explored for continuous control.

### 2. Custom Rocket Physics

Replace the standard LunarLander environment with a custom rocket simulation containing:

* Rocket mass
* Fuel consumption
* Thrust curves
* Gravity
* Atmospheric drag
* Engine response
* Center-of-mass changes
* Landing gear
* Vehicle attitude
* Angular dynamics

### 3. Sensor-Based State Estimation

A future autonomous lander could use simulated or real sensor inputs:

```text
Altitude
Vertical Velocity
Horizontal Velocity
Acceleration
Orientation
Angular Velocity
Position Error
Fuel Level
```

### 4. Computer Vision

Camera input could be used to detect:

* Landing pad
* Terrain
* Horizon
* Obstacles
* Rocket position

A vision model could then provide information to the flight controller.

### 5. Hardware-in-the-Loop Simulation

The trained controller could eventually be tested using a hardware-in-the-loop architecture:

```text
        Sensor Simulation
               │
               ▼
        Flight Computer
               │
               ▼
       Autonomous Controller
               │
               ▼
         Actuator Commands
               │
               ▼
        Physics Simulator
               │
               └──────────────► Feedback
```

---

## ⚠️ Safety Notice

This project is a **simulation and reinforcement-learning research prototype**.

The trained policy should **not be directly connected to a real rocket, propulsion system, or flight hardware**.

A real autonomous rocket would require extensive:

* Flight dynamics validation
* Hardware-in-the-loop testing
* Fault detection
* Redundant sensors
* Deterministic safety controls
* Verified flight-control software
* Simulation across a wide range of conditions
* Independent safety and abort systems

The LunarLander environment is not an accurate model of a real rocket.

---

## 🌙 Project Inspiration

The project is inspired by reinforcement-learning-based autonomous lunar landing demonstrations using the Gymnasium LunarLander environment.

The core idea is to teach the vehicle to learn its control policy through repeated interaction with a simulated physics environment rather than manually specifying every landing maneuver.

---

## 👨‍💻 Author

**Yashwanth Munagaala**

AI/ML & Data Engineering Enthusiast

Areas of interest:

* Artificial Intelligence
* Machine Learning
* Deep Learning
* Reinforcement Learning
* Computer Vision
* Generative AI
* Autonomous Systems
* Data Engineering

---

## ⭐ Key Takeaway

The central idea of this project is:

> **Train once, reuse the learned controller, and test autonomous landing across different initial conditions.**

The successful landing with seed `1234` is not itself the model. The **DQN weights learned during training** are the reusable autonomous controller.

```text
                 TRAINING
                    │
                    ▼
          ┌──────────────────┐
          │   DQN Controller │
          │   Learned Policy │
          └────────┬─────────┘
                   │
              Save Weights
                   │
                   ▼
          ┌──────────────────┐
          │  Trained Model   │
          └────────┬─────────┘
                   │
       ┌───────────┼───────────┐
       ▼           ▼           ▼
   New Rocket   New Rocket   New Rocket
   Seed A       Seed B       Seed C
       │           │           │
       ▼           ▼           ▼
   Autonomous   Autonomous   Autonomous
    Landing      Landing      Landing
```

**The ultimate goal is to evolve this simulation into a more realistic autonomous rocket landing and guidance system.**

