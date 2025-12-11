---
id: learning-algorithms
title: Learning Algorithms in Humanoid Robotics
slug: /learning-algorithms
---

# Learning Algorithms in Humanoid Robotics

## Introduction to Robot Learning

Learning algorithms in humanoid robotics enable robots to improve their performance through experience, adapt to new situations, and acquire new skills without explicit programming. Unlike traditional robots with fixed behaviors, learning-enabled humanoid robots can continuously adapt to changing environments and tasks, making them more versatile and capable in human environments. Advanced learning algorithms in humanoid robotics incorporate sophisticated techniques including deep learning, reinforcement learning, imitation learning, and multi-modal integration to achieve human-level adaptability.

### Categories of Learning in Robotics

#### Supervised Learning
Supervised learning algorithms learn mappings from input to output based on labeled training data:

- **Applications**: Object recognition, pose estimation, behavior cloning
- **Advantages**: Well-understood, guaranteed convergence under conditions
- **Challenges**: Requires labeled training data, may not generalize well
- **Examples**: Support vector machines, neural networks, decision trees
- **Advanced Applications**: Vision-language models, multi-modal perception, predictive modeling

#### Unsupervised Learning
Unsupervised learning discovers patterns in data without labeled examples:

- **Applications**: Clustering behaviors, discovering environmental structures
- **Advantages**: Works with unlabeled data, discovers hidden patterns
- **Challenges**: Difficult to evaluate, may discover irrelevant patterns
- **Examples**: K-means clustering, autoencoders, principal component analysis
- **Advanced Applications**: Anomaly detection, environment modeling, behavior discovery

#### Reinforcement Learning
Reinforcement learning learns optimal behaviors through trial and error:

- **Applications**: Motor skill learning, navigation, decision making
- **Advantages**: Learns optimal behaviors, handles sequential decisions
- **Challenges**: Sample efficiency, safety during learning, reward design
- **Examples**: Q-learning, policy gradient methods, actor-critic algorithms
- **Advanced Applications**: Deep reinforcement learning, multi-agent systems, hierarchical control

### Advanced Learning Implementation

Here's an implementation of a comprehensive learning framework for humanoid robotics:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import gym
from gym import spaces
import random
from dataclasses import dataclass

@dataclass
class RobotExperience:
    """Data structure for robot experiences in learning"""
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    info: Dict

class RobotLearningFramework:
    """Comprehensive learning framework for humanoid robots"""

    def __init__(self, robot_config: Dict):
        self.robot_config = robot_config
        self.experience_buffer = []
        self.learning_algorithms = {}

        # Initialize different learning approaches
        self.supervised_learner = SupervisedLearningModule(robot_config)
        self.unsupervised_learner = UnsupervisedLearningModule(robot_config)
        self.rl_learner = ReinforcementLearningModule(robot_config)
        self.imitation_learner = ImitationLearningModule(robot_config)

        # Performance tracking
        self.performance_metrics = {
            'supervised': [],
            'unsupervised': [],
            'reinforcement': [],
            'imitation': []
        }

    def add_experience(self, experience: RobotExperience):
        """Add experience to the buffer"""
        self.experience_buffer.append(experience)

        # Maintain buffer size
        if len(self.experience_buffer) > 100000:  # Limit buffer size
            self.experience_buffer = self.experience_buffer[-50000:]

    def learn_from_experience(self, algorithm_type: str, batch_size: int = 32):
        """Learn from accumulated experiences using specified algorithm"""
        if len(self.experience_buffer) < batch_size:
            return  # Not enough experiences yet

        # Sample random batch
        batch_indices = random.sample(range(len(self.experience_buffer)), batch_size)
        batch = [self.experience_buffer[i] for i in batch_indices]

        if algorithm_type == 'supervised':
            return self.supervised_learner.train_batch(batch)
        elif algorithm_type == 'unsupervised':
            return self.unsupervised_learner.train_batch(batch)
        elif algorithm_type == 'reinforcement':
            return self.rl_learner.train_batch(batch)
        elif algorithm_type == 'imitation':
            return self.imitation_learner.train_batch(batch)
        else:
            raise ValueError(f"Unknown algorithm type: {algorithm_type}")

    def evaluate_performance(self, algorithm_type: str, test_environment) -> Dict[str, float]:
        """Evaluate learning performance on test environment"""
        if algorithm_type == 'supervised':
            return self.supervised_learner.evaluate(test_environment)
        elif algorithm_type == 'unsupervised':
            return self.unsupervised_learner.evaluate(test_environment)
        elif algorithm_type == 'reinforcement':
            return self.rl_learner.evaluate(test_environment)
        elif algorithm_type == 'imitation':
            return self.imitation_learner.evaluate(test_environment)

        return {}

class SupervisedLearningModule(nn.Module):
    """Supervised learning module for robot perception and control"""

    def __init__(self, robot_config: Dict):
        super().__init__()
        self.robot_config = robot_config

        # Perception network (e.g., for object recognition)
        self.perception_net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),  # Adjust based on input size
            nn.ReLU(),
            nn.Linear(512, robot_config.get('num_classes', 10))
        )

        # Control network (e.g., for inverse kinematics)
        self.control_net = nn.Sequential(
            nn.Linear(robot_config.get('state_dim', 20), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, robot_config.get('action_dim', 14))
        )

        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)
        self.criterion = nn.MSELoss()

    def forward(self, x, task_type: str = 'perception'):
        """Forward pass for different supervised learning tasks"""
        if task_type == 'perception':
            return self.perception_net(x)
        elif task_type == 'control':
            return self.control_net(x)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def train_batch(self, batch: List[RobotExperience]) -> Dict[str, float]:
        """Train on a batch of supervised data"""
        # Implementation would train on labeled data
        # This is a simplified example
        states = torch.stack([torch.tensor(exp.state, dtype=torch.float32) for exp in batch])
        actions = torch.stack([torch.tensor(exp.action, dtype=torch.float32) for exp in batch])

        # Forward pass
        predicted_actions = self(states, task_type='control')
        loss = self.criterion(predicted_actions, actions)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item(), 'batch_size': len(batch)}

    def evaluate(self, test_environment) -> Dict[str, float]:
        """Evaluate supervised learning performance"""
        # Implementation would evaluate on test data
        return {'accuracy': 0.85, 'mse': 0.02}

class UnsupervisedLearningModule(nn.Module):
    """Unsupervised learning module for pattern discovery and representation learning"""

    def __init__(self, robot_config: Dict):
        super().__init__()
        self.robot_config = robot_config

        # Autoencoder for feature learning
        self.encoder = nn.Sequential(
            nn.Linear(robot_config.get('state_dim', 20), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)  # Compressed representation
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, robot_config.get('state_dim', 20))
        )

        # Clustering component
        self.cluster_centers = nn.Parameter(
            torch.randn(robot_config.get('num_clusters', 5), 32)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.mse_loss = nn.MSELoss()

    def forward(self, x):
        """Forward pass through autoencoder"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

    def train_batch(self, batch: List[RobotExperience]) -> Dict[str, float]:
        """Train autoencoder on robot experiences"""
        states = torch.stack([torch.tensor(exp.state, dtype=torch.float32) for exp in batch])

        # Forward pass
        reconstructed, encoded = self(states)
        loss = self.mse_loss(reconstructed, states)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'reconstruction_loss': loss.item(), 'batch_size': len(batch)}

    def evaluate(self, test_environment) -> Dict[str, float]:
        """Evaluate unsupervised learning performance"""
        return {'reconstruction_error': 0.05, 'cluster_purity': 0.78}

class ReinforcementLearningModule(nn.Module):
    """Reinforcement learning module for policy optimization"""

    def __init__(self, robot_config: Dict):
        super().__init__()
        self.robot_config = robot_config
        self.state_dim = robot_config.get('state_dim', 20)
        self.action_dim = robot_config.get('action_dim', 14)

        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim),
            nn.Tanh()  # Actions in [-1, 1]
        )

        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Value estimate
        )

        # Target networks for stability
        self.actor_target = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim),
            nn.Tanh()
        )

        self.critic_target = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Copy parameters to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        # Experience replay buffer
        self.replay_buffer = []
        self.buffer_size = 100000
        self.batch_size = 64

    def get_action(self, state: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
        """Get action from policy with optional noise for exploration"""
        with torch.no_grad():
            action = self.actor(state)
            if add_noise:
                noise = torch.randn_like(action) * 0.1
                action = torch.clamp(action + noise, -1, 1)
        return action

    def train_batch(self, batch: List[RobotExperience]) -> Dict[str, float]:
        """Train actor-critic on experience batch"""
        if len(batch) < self.batch_size:
            return {'loss': 0.0, 'batch_size': 0}

        # Sample from replay buffer
        batch_indices = random.sample(range(len(self.replay_buffer)),
                                    min(self.batch_size, len(self.replay_buffer)))
        sampled_batch = [self.replay_buffer[i] for i in batch_indices]

        states = torch.stack([torch.tensor(exp.state, dtype=torch.float32) for exp in sampled_batch])
        actions = torch.stack([torch.tensor(exp.action, dtype=torch.float32) for exp in sampled_batch])
        rewards = torch.tensor([exp.reward for exp in sampled_batch], dtype=torch.float32).unsqueeze(1)
        next_states = torch.stack([torch.tensor(exp.next_state, dtype=torch.float32) for exp in sampled_batch])
        dones = torch.tensor([exp.done for exp in sampled_batch], dtype=torch.float32).unsqueeze(1)

        # Critic update
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_q_values = self.critic_target(next_states)
            target_q_values = rewards + (0.99 * next_q_values * (1 - dones))

        current_q_values = self.critic(states)
        critic_loss = nn.MSELoss()(current_q_values, target_q_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self._soft_update_targets()

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'batch_size': len(sampled_batch)
        }

    def _soft_update_targets(self, tau: float = 0.005):
        """Soft update target networks"""
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def evaluate(self, test_environment) -> Dict[str, float]:
        """Evaluate reinforcement learning performance"""
        # Implementation would run test episodes
        return {'average_reward': 150.0, 'success_rate': 0.75}

class ImitationLearningModule(nn.Module):
    """Imitation learning module for learning from demonstrations"""

    def __init__(self, robot_config: Dict):
        super().__init__()
        self.robot_config = robot_config
        self.state_dim = robot_config.get('state_dim', 20)
        self.action_dim = robot_config.get('action_dim', 14)

        # Policy network for behavior cloning
        self.policy = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim),
            nn.Tanh()
        )

        # Discriminator for GAIL (Generative Adversarial Imitation Learning)
        self.discriminator = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Probability of being expert-like
        )

        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def behavior_clone(self, states: torch.Tensor, expert_actions: torch.Tensor):
        """Behavior cloning: learn to mimic expert actions"""
        predicted_actions = self.policy(states)
        loss = self.mse_loss(predicted_actions, expert_actions)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def gail_loss(self, states: torch.Tensor, actions: torch.Tensor,
                  expert_states: torch.Tensor, expert_actions: torch.Tensor):
        """GAIL loss for adversarial imitation learning"""
        # Discriminator loss
        real_pairs = torch.cat([expert_states, expert_actions], dim=1)
        fake_pairs = torch.cat([states, actions], dim=1)

        real_logits = self.discriminator(real_pairs)
        fake_logits = self.discriminator(fake_pairs.detach())

        real_loss = self.bce_loss(real_logits, torch.ones_like(real_logits))
        fake_loss = self.bce_loss(fake_logits, torch.zeros_like(fake_logits))
        d_loss = real_loss + fake_loss

        # Generator (policy) loss - maximize discriminator confusion
        fake_logits_gen = self.discriminator(fake_pairs)
        g_loss = self.bce_loss(fake_logits_gen, torch.ones_like(fake_logits_gen))

        return d_loss, g_loss

    def train_batch(self, batch: List[RobotExperience]) -> Dict[str, float]:
        """Train imitation learning model"""
        states = torch.stack([torch.tensor(exp.state, dtype=torch.float32) for exp in batch])
        actions = torch.stack([torch.tensor(exp.action, dtype=torch.float32) for exp in batch])

        # For this example, we'll use behavior cloning
        # In practice, you'd have expert demonstrations
        loss = self.mse_loss(self.policy(states), actions)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'imitation_loss': loss.item(), 'batch_size': len(batch)}

    def evaluate(self, test_environment) -> Dict[str, float]:
        """Evaluate imitation learning performance"""
        return {'behavior_similarity': 0.82, 'task_success': 0.68}

# Example usage of the learning framework
def example_robot_learning():
    """Example of using the robot learning framework"""
    robot_config = {
        'state_dim': 20,
        'action_dim': 14,
        'num_classes': 10,
        'num_clusters': 5
    }

    framework = RobotLearningFramework(robot_config)

    # Simulate some experiences
    for episode in range(100):
        for step in range(50):  # 50 steps per episode
            # Create dummy experience
            exp = RobotExperience(
                state=np.random.randn(20),
                action=np.random.randn(14),
                reward=np.random.randn(),
                next_state=np.random.randn(20),
                done=False,
                info={}
            )
            framework.add_experience(exp)

    # Train different algorithms
    for epoch in range(10):
        supervised_result = framework.learn_from_experience('supervised', batch_size=32)
        unsupervised_result = framework.learn_from_experience('unsupervised', batch_size=32)
        rl_result = framework.learn_from_experience('reinforcement', batch_size=32)
        imitation_result = framework.learn_from_experience('imitation', batch_size=32)

        print(f"Epoch {epoch}: Supervised loss: {supervised_result['loss']:.4f}")

    return framework
```

## Machine Learning Fundamentals

### Core Concepts

#### Training, Validation, and Test Sets
Proper evaluation of learning algorithms requires data separation:

- **Training Set**: Data used to train the model
- **Validation Set**: Data used for hyperparameter tuning and model selection
- **Test Set**: Data used for final performance evaluation
- **Cross-Validation**: Technique for better performance estimation
- **Temporal Split**: For time-series data to prevent data leakage
- **Stratified Splitting**: Maintaining class distribution in splits

Here's an implementation of advanced data splitting techniques for robotics applications:

```python
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from typing import Tuple, Dict, Any
import torch

class RobotDataSplitter:
    """Advanced data splitting for robotics applications"""

    def __init__(self, robot_config: Dict[str, Any]):
        self.robot_config = robot_config

    def temporal_split(self,
                      data: np.ndarray,
                      labels: np.ndarray,
                      test_ratio: float = 0.2,
                      val_ratio: float = 0.1) -> Tuple[np.ndarray, ...]:
        """Split data temporally to prevent future information leakage"""
        n_samples = len(data)

        # Calculate split indices
        test_start = int(n_samples * (1 - test_ratio))
        val_start = int(n_samples * (1 - test_ratio - val_ratio))

        # Split temporally
        X_train, y_train = data[:val_start], labels[:val_start]
        X_val, y_val = data[val_start:test_start], labels[val_start:test_start]
        X_test, y_test = data[test_start:], labels[test_start:]

        return X_train, X_val, X_test, y_train, y_val, y_test

    def episode_split(self,
                     episodes: List[Dict[str, Any]],
                     test_ratio: float = 0.2,
                     val_ratio: float = 0.1) -> Tuple[List, List, List]:
        """Split data by episodes to maintain temporal consistency"""
        n_episodes = len(episodes)

        # Shuffle episodes randomly
        shuffled_episodes = episodes.copy()
        np.random.shuffle(shuffled_episodes)

        # Calculate split indices
        test_start = int(n_episodes * (1 - test_ratio))
        val_start = int(n_episodes * (1 - test_ratio - val_ratio))

        # Split episodes
        train_episodes = shuffled_episodes[:val_start]
        val_episodes = shuffled_episodes[val_start:test_start]
        test_episodes = shuffled_episodes[test_start:]

        return train_episodes, val_episodes, test_episodes

    def stratified_robot_split(self,
                              data: np.ndarray,
                              labels: np.ndarray,
                              n_folds: int = 5) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Stratified cross-validation for robotics data"""
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = []

        for train_idx, test_idx in kf.split(data):
            X_train, X_test = data[train_idx], data[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]
            splits.append((X_train, X_test, y_train, y_test))

        return splits

# Example usage
def example_data_splitting():
    """Example of using advanced data splitting techniques"""
    robot_config = {
        'state_dim': 20,
        'action_dim': 14
    }

    splitter = RobotDataSplitter(robot_config)

    # Example with random data
    data = np.random.randn(1000, 20)
    labels = np.random.randint(0, 5, 1000)  # 5 classes

    X_train, X_val, X_test, y_train, y_val, y_test = splitter.temporal_split(
        data, labels, test_ratio=0.2, val_ratio=0.1
    )

    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test
```

#### Overfitting and Underfitting
Balancing model complexity is crucial for generalization:

- **Overfitting**: Model performs well on training data but poorly on new data
- **Underfitting**: Model is too simple to capture data patterns
- **Regularization**: Techniques to prevent overfitting
- **Model Selection**: Choosing appropriate model complexity
- **Bias-Variance Tradeoff**: Balancing simplicity and complexity
- **Early Stopping**: Preventing overfitting during training

Advanced regularization and model selection techniques:

```python
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class AdvancedRegularization:
    """Advanced regularization techniques for robotics learning"""

    def __init__(self, model: torch.nn.Module):
        self.model = model

    def l1_regularization(self, lambda_l1: float = 1e-4) -> torch.Tensor:
        """L1 regularization for sparsity"""
        l1_reg = torch.tensor(0., requires_grad=True)
        for param in self.model.parameters():
            l1_reg = l1_reg + torch.norm(param, 1)
        return lambda_l1 * l1_reg

    def l2_regularization(self, lambda_l2: float = 1e-4) -> torch.Tensor:
        """L2 regularization for weight decay"""
        l2_reg = torch.tensor(0., requires_grad=True)
        for param in self.model.parameters():
            l2_reg = l2_reg + torch.norm(param, 2)
        return lambda_l2 * l2_reg

    def dropout_regularization(self, x: torch.Tensor, p: float = 0.5) -> torch.Tensor:
        """Dropout regularization"""
        return F.dropout(x, p=p, training=self.model.training)

    def batch_norm_regularization(self, x: torch.Tensor) -> torch.Tensor:
        """Batch normalization for regularization effect"""
        return F.batch_norm(x, torch.zeros(x.size(1)), torch.ones(x.size(1)),
                           training=self.model.training)

    def adversarial_regularization(self,
                                  x: torch.Tensor,
                                  y: torch.Tensor,
                                  epsilon: float = 0.01) -> torch.Tensor:
        """Adversarial training for robustness"""
        x.requires_grad_(True)
        output = self.model(x)
        loss = F.mse_loss(output, y)

        # Compute gradients
        gradients = torch.autograd.grad(outputs=loss, inputs=x,
                                       grad_outputs=torch.ones_like(loss),
                                       create_graph=True)[0]

        # Add adversarial perturbation
        perturbation = epsilon * torch.sign(gradients)
        adversarial_x = x + perturbation

        return adversarial_x

class ModelComplexityAnalyzer:
    """Analyze and select optimal model complexity"""

    def __init__(self):
        self.complexity_metrics = {}

    def calculate_model_complexity(self, model: torch.nn.Module) -> Dict[str, float]:
        """Calculate various complexity metrics for the model"""
        complexity_metrics = {}

        # Parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Layer-wise complexity
        layer_params = []
        for name, layer in model.named_modules():
            if hasattr(layer, 'weight') and layer.weight is not None:
                layer_params.append(layer.weight.numel())

        complexity_metrics = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_ratio': trainable_params / total_params if total_params > 0 else 0,
            'layer_count': len(layer_params),
            'avg_params_per_layer': np.mean(layer_params) if layer_params else 0,
            'max_layer_params': max(layer_params) if layer_params else 0
        }

        return complexity_metrics

    def estimate_generalization_gap(self,
                                   train_loss: float,
                                   val_loss: float) -> Dict[str, float]:
        """Estimate generalization gap and model capacity"""
        gap = val_loss - train_loss
        relative_gap = gap / train_loss if train_loss != 0 else 0

        # Classify model state
        if relative_gap < 0.05:
            state = "Underfitting"
        elif relative_gap > 0.1:
            state = "Overfitting"
        else:
            state = "Good fit"

        return {
            'generalization_gap': gap,
            'relative_gap': relative_gap,
            'model_state': state,
            'recommendation': self._get_complexity_recommendation(state)
        }

    def _get_complexity_recommendation(self, state: str) -> str:
        """Get complexity recommendation based on model state"""
        recommendations = {
            "Underfitting": "Increase model complexity (more layers/parameters)",
            "Overfitting": "Reduce model complexity or add regularization",
            "Good fit": "Model complexity is appropriate"
        }
        return recommendations.get(state, "No recommendation")
```

### Evaluation Metrics

#### Classification Metrics
For categorical predictions:

- **Accuracy**: Fraction of correct predictions
- **Precision**: Fraction of positive predictions that are correct
- **Recall**: Fraction of actual positives that are predicted positive
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **Precision-Recall AUC**: Area under precision-recall curve

Advanced classification metrics implementation:

```python
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import torch.nn.functional as F

class AdvancedClassificationMetrics:
    """Advanced metrics for classification in robotics applications"""

    def __init__(self):
        self.metrics_history = []

    def compute_comprehensive_metrics(self,
                                    y_true: torch.Tensor,
                                    y_pred: torch.Tensor,
                                    y_pred_proba: torch.Tensor = None) -> Dict[str, float]:
        """Compute comprehensive classification metrics"""
        y_true_np = y_true.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()

        metrics = {}

        # Basic metrics
        metrics['accuracy'] = (y_true == y_pred).float().mean().item()

        # Per-class metrics
        unique_classes = torch.unique(y_true)
        for class_idx in unique_classes:
            class_mask = (y_true == class_idx)
            class_pred_mask = (y_pred == class_idx)

            tp = ((y_true == class_idx) & (y_pred == class_idx)).sum().item()
            fp = ((y_true != class_idx) & (y_pred == class_idx)).sum().item()
            fn = ((y_true == class_idx) & (y_pred != class_idx)).sum().item()
            tn = ((y_true != class_idx) & (y_pred != class_idx)).sum().item()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            metrics[f'precision_class_{class_idx.item()}'] = precision
            metrics[f'recall_class_{class_idx.item()}'] = recall
            metrics[f'f1_class_{class_idx.item()}'] = f1

        # Macro and micro averages
        if len(unique_classes) > 1:
            precisions = [metrics.get(f'precision_class_{i.item()}', 0) for i in unique_classes]
            recalls = [metrics.get(f'recall_class_{i.item()}', 0) for i in unique_classes]
            f1_scores = [metrics.get(f'f1_class_{i.item()}', 0) for i in unique_classes]

            metrics['macro_precision'] = np.mean(precisions)
            metrics['macro_recall'] = np.mean(recalls)
            metrics['macro_f1'] = np.mean(f1_scores)

        # If probabilities are provided, compute AUC metrics
        if y_pred_proba is not None:
            y_pred_proba_np = y_pred_proba.cpu().numpy()
            if y_pred_proba_np.ndim == 1 or y_pred_proba_np.shape[1] == 1:
                # Binary classification
                metrics['roc_auc'] = roc_auc_score(y_true_np, y_pred_proba_np)
            elif y_pred_proba_np.shape[1] == 2:
                # Binary classification with 2 outputs
                metrics['roc_auc'] = roc_auc_score(y_true_np, y_pred_proba_np[:, 1])
            else:
                # Multi-class classification
                metrics['roc_auc'] = roc_auc_score(y_true_np, y_pred_proba_np, multi_class='ovr')

        return metrics

    def compute_robotic_task_metrics(self,
                                   task_success: torch.Tensor,
                                   action_accuracy: torch.Tensor,
                                   safety_violations: torch.Tensor) -> Dict[str, float]:
        """Compute metrics specific to robotic tasks"""
        return {
            'task_success_rate': task_success.float().mean().item(),
            'action_accuracy': action_accuracy.float().mean().item(),
            'safety_violation_rate': safety_violations.float().mean().item(),
            'normalized_performance': (
                task_success.float().mean() * 0.7 +
                action_accuracy.float().mean() * 0.2 +
                (1 - safety_violations.float().mean()) * 0.1
            ).item()
        }

#### Regression Metrics
For continuous predictions:

- **Mean Squared Error (MSE)**: Average squared prediction error
- **Mean Absolute Error (MAE)**: Average absolute prediction error
- **R-squared**: Proportion of variance explained by the model
- **Root Mean Squared Error (RMSE)**: Square root of MSE
- **Mean Absolute Percentage Error (MAPE)**: Percentage error
- **Symmetric MAPE (SMAPE)**: Symmetric version of MAPE

Advanced regression metrics implementation:

```python
class AdvancedRegressionMetrics:
    """Advanced metrics for regression in robotics applications"""

    def __init__(self):
        pass

    def compute_comprehensive_metrics(self,
                                    y_true: torch.Tensor,
                                    y_pred: torch.Tensor) -> Dict[str, float]:
        """Compute comprehensive regression metrics"""
        # Ensure tensors are on CPU for numpy operations
        y_true_np = y_true.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()

        # Basic metrics
        mse = F.mse_loss(y_pred, y_true).item()
        mae = F.l1_loss(y_pred, y_true).item()
        rmse = torch.sqrt(F.mse_loss(y_pred, y_true)).item()

        # R-squared
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
        r2 = (1 - ss_res / ss_tot).item()

        # Mean Absolute Percentage Error (MAPE)
        mape = torch.mean(torch.abs((y_true - y_pred) / (y_true + 1e-8))).item() * 100

        # Symmetric MAPE
        smape = torch.mean(2 * torch.abs(y_pred - y_true) /
                          (torch.abs(y_true) + torch.abs(y_pred) + 1e-8)).item() * 100

        # Explained variance
        var_true = torch.var(y_true)
        var_error = torch.var(y_true - y_pred)
        explained_var = (var_true - var_error) / var_true if var_true != 0 else 0

        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r_squared': r2,
            'mape': mape,
            'smape': smape,
            'explained_variance': explained_var.item()
        }

    def compute_robotic_control_metrics(self,
                                      desired_positions: torch.Tensor,
                                      actual_positions: torch.Tensor,
                                      desired_velocities: torch.Tensor = None,
                                      actual_velocities: torch.Tensor = None) -> Dict[str, float]:
        """Compute metrics specific to robotic control tasks"""
        # Position tracking error
        pos_error = torch.norm(desired_positions - actual_positions, dim=-1)
        avg_pos_error = pos_error.mean().item()
        max_pos_error = pos_error.max().item()

        metrics = {
            'avg_position_error': avg_pos_error,
            'max_position_error': max_pos_error,
            'pos_error_std': pos_error.std().item(),
            'pos_error_percentile_95': torch.quantile(pos_error, 0.95).item()
        }

        # If velocity information is available
        if desired_velocities is not None and actual_velocities is not None:
            vel_error = torch.norm(desired_velocities - actual_velocities, dim=-1)
            metrics.update({
                'avg_velocity_error': vel_error.mean().item(),
                'max_velocity_error': vel_error.max().item(),
                'smoothness_index': self._compute_smoothness_index(actual_velocities)
            })

        return metrics

    def _compute_smoothness_index(self, velocities: torch.Tensor) -> float:
        """Compute smoothness index based on velocity changes"""
        if velocities.shape[0] < 2:
            return 0.0

        # Compute acceleration
        accelerations = velocities[1:] - velocities[:-1]
        jerk = accelerations[1:] - accelerations[:-1]

        # Smoothness is inversely related to jerk
        smoothness = 1.0 / (1.0 + torch.mean(torch.abs(jerk)).item())
        return smoothness
```

## Supervised Learning in Robotics

### Classification for Perception

#### Object Recognition
Classifying objects in visual data with advanced techniques:

- **Feature Extraction**: Extracting relevant visual features
- **Deep Learning**: Convolutional neural networks for object recognition
- **Real-time Processing**: Optimizing for real-time performance
- **Transfer Learning**: Adapting pre-trained models to robot tasks
- **Few-shot Learning**: Learning with limited examples
- **Domain Adaptation**: Adapting to new environments
- **Multi-modal Fusion**: Combining visual, tactile, and other sensory data

Here's an implementation of advanced object recognition for humanoid robots:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from typing import Dict, List, Tuple, Optional

class AdvancedObjectRecognition(nn.Module):
    """Advanced object recognition system for humanoid robots"""

    def __init__(self, num_classes: int = 100, input_size: int = 224):
        super().__init__()

        # Backbone network (using pre-trained ResNet)
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove final classification layer

        # Multi-scale feature extraction
        self.multi_scale_extractor = MultiScaleFeatureExtractor()

        # Attention mechanism for focusing on relevant regions
        self.attention_module = AttentionModule(2048)  # ResNet50 feature dim

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        # Additional heads for related tasks
        self.bbox_head = nn.Linear(2048, 4)  # Bounding box prediction
        self.confidence_head = nn.Linear(2048, 1)  # Confidence prediction

        self.input_size = input_size

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with multiple outputs"""
        # Extract features from backbone
        features = self.backbone(x)  # Shape: [batch, 2048]

        # Apply attention
        attended_features = self.attention_module(features)

        # Multi-scale processing
        multi_scale_features = self.multi_scale_extractor(x)

        # Combine features
        combined_features = attended_features + multi_scale_features.mean(dim=[2, 3])

        # Classification
        class_logits = self.classifier(combined_features)

        # Additional outputs
        bbox_pred = self.bbox_head(attended_features)
        confidence = torch.sigmoid(self.confidence_head(attended_features))

        return {
            'class_logits': class_logits,
            'bbox_pred': bbox_pred,
            'confidence': confidence,
            'features': attended_features
        }

class MultiScaleFeatureExtractor(nn.Module):
    """Extract features at multiple scales for robust recognition"""

    def __init__(self):
        super().__init__()

        # Multi-scale convolutions
        self.scale1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.scale2 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.scale3 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract multi-scale features"""
        feat1 = F.relu(self.scale1(F.avg_pool2d(x, 2)))  # Downsampled
        feat2 = F.relu(self.scale2(x))  # Original scale
        feat3 = F.relu(self.scale3(F.interpolate(x, scale_factor=2)))  # Upsampled

        # Concatenate features
        combined = torch.cat([feat1, feat2, feat3], dim=1)
        fused = self.fusion(combined)

        return fused

class AttentionModule(nn.Module):
    """Attention mechanism for focusing on relevant features"""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim

        # Attention computation
        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply attention to features"""
        attention_weights = self.attention_net(features)
        attended_features = features * attention_weights
        return attended_features

class FewShotObjectRecognizer(nn.Module):
    """Few-shot learning for object recognition in robotics"""

    def __init__(self, feature_dim: int = 2048, hidden_dim: int = 512):
        super().__init__()

        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Prototypical network components
        self.support_features = None
        self.support_labels = None

    def encode_features(self, features: torch.Tensor) -> torch.Tensor:
        """Encode features for few-shot learning"""
        return self.feature_encoder(features)

    def set_support_set(self, support_features: torch.Tensor,
                       support_labels: torch.Tensor):
        """Set the support set for few-shot learning"""
        encoded_support = self.encode_features(support_features)

        # Compute prototype for each class
        unique_labels = torch.unique(support_labels)
        prototypes = []

        for label in unique_labels:
            class_mask = (support_labels == label)
            class_features = encoded_support[class_mask]
            prototype = class_features.mean(dim=0, keepdim=True)
            prototypes.append(prototype)

        self.prototypes = torch.cat(prototypes, dim=0)
        self.prototype_labels = unique_labels

    def forward(self, query_features: torch.Tensor) -> torch.Tensor:
        """Classify query features using prototypes"""
        encoded_query = self.encode_features(query_features)

        # Compute distances to prototypes
        distances = torch.cdist(encoded_query, self.prototypes)

        # Convert distances to similarities (negative distances)
        similarities = -distances
        return similarities

# Example usage for robot perception
class RobotPerceptionSystem:
    """Complete perception system for humanoid robots"""

    def __init__(self, config: Dict):
        self.config = config

        # Object recognition module
        self.object_recognizer = AdvancedObjectRecognition(
            num_classes=config.get('num_object_classes', 50)
        )

        # Few-shot learning capability
        self.few_shot_recognizer = FewShotObjectRecognizer()

        # Human activity recognition
        self.activity_recognizer = HumanActivityRecognizer(
            num_activities=config.get('num_activities', 20)
        )

        # Sensor fusion module
        self.sensor_fusion = SensorFusionModule()

    def process_visual_input(self, rgb_image: torch.Tensor,
                           depth_image: Optional[torch.Tensor] = None) -> Dict[str, any]:
        """Process visual input for object recognition"""
        # Run object recognition
        recognition_result = self.object_recognizer(rgb_image)

        # If depth is available, enhance with 3D information
        if depth_image is not None:
            recognition_result['depth_enhanced'] = self._enhance_with_depth(
                recognition_result, depth_image
            )

        return recognition_result

    def _enhance_with_depth(self, recognition_result: Dict,
                           depth_image: torch.Tensor) -> Dict[str, any]:
        """Enhance recognition with depth information"""
        # Extract 3D information from depth
        object_distances = self._compute_object_distances(
            recognition_result['bbox_pred'], depth_image
        )

        return {
            'distances': object_distances,
            '3d_positions': self._compute_3d_positions(
                recognition_result['bbox_pred'], depth_image
            )
        }

class HumanActivityRecognizer(nn.Module):
    """Recognize human activities from multi-modal data"""

    def __init__(self, num_activities: int = 20):
        super().__init__()

        # Pose estimation branch
        self.pose_branch = nn.Sequential(
            nn.Linear(34, 128),  # 17 keypoints * 2 (x,y) = 34
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        # Temporal modeling
        self.temporal_lstm = nn.LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )

        # Multi-modal fusion
        self.fusion_layer = nn.MultiheadAttention(
            embed_dim=256, num_heads=8, dropout=0.1
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_activities)
        )

        self.num_activities = num_activities

    def forward(self, pose_sequence: torch.Tensor,
               temporal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Recognize activity from pose sequence"""
        batch_size, seq_len, pose_dim = pose_sequence.shape

        # Process each time step
        pose_features = self.pose_branch(pose_sequence.view(-1, pose_dim))
        pose_features = pose_features.view(batch_size, seq_len, -1)

        # Apply temporal modeling
        temporal_features, _ = self.temporal_lstm(pose_features)

        # Apply attention for important frames
        attended_features, attention_weights = self.fusion_layer(
            temporal_features.transpose(0, 1),
            temporal_features.transpose(0, 1),
            temporal_features.transpose(0, 1)
        )
        attended_features = attended_features.transpose(0, 1)

        # Global average pooling
        pooled_features = attended_features.mean(dim=1)

        # Classify activity
        activity_logits = self.classifier(pooled_features)

        return activity_logits

class SensorFusionModule(nn.Module):
    """Fusion of multiple sensor modalities"""

    def __init__(self):
        super().__init__()

        # Modality-specific encoders
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 16, 256)
        )

        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 7, 2, 3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 16, 128)
        )

        self.audio_encoder = nn.Sequential(
            nn.Conv1d(1, 64, 15, 8, 7),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(64 * 16, 128)
        )

        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=256, num_heads=8, dropout=0.1
        )

        # Fusion classifier
        self.fusion_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128)
        )

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor,
               audio: torch.Tensor) -> torch.Tensor:
        """Fuse information from multiple sensors"""
        rgb_features = self.rgb_encoder(rgb)
        depth_features = self.depth_encoder(depth)
        audio_features = self.audio_encoder(audio)

        # Concatenate features
        combined_features = torch.cat([
            rgb_features, depth_features, audio_features
        ], dim=1)

        # Apply fusion
        fused_features = self.fusion_classifier(combined_features)

        return fused_features
```

#### Human Activity Recognition
Understanding human actions and behaviors with advanced techniques:

- **Multi-modal Data**: Combining visual, audio, and sensor data
- **Temporal Modeling**: Capturing temporal patterns in activities
- **Context Awareness**: Incorporating environmental context
- **Real-time Recognition**: Fast recognition for interaction
- **Social Context**: Understanding social interactions and group activities
- **Intent Prediction**: Predicting human intentions from observed actions
- **Cross-Modal Learning**: Learning from multiple sensory modalities

### Regression for Control

#### Inverse Kinematics
Learning mapping from desired end-effector pose to joint angles with advanced approaches:

- **Problem Formulation**: Mapping Cartesian space to joint space
- **Multiple Solutions**: Handling redundant robot configurations
- **Real-time Performance**: Fast computation for control
- **Accuracy Requirements**: Precision for manipulation tasks
- **Singularity Handling**: Managing kinematic singularities
- **Obstacle Avoidance**: Incorporating collision-free path planning
- **Redundancy Resolution**: Optimizing secondary objectives

Advanced inverse kinematics implementation:

```python
class AdvancedInverseKinematics(nn.Module):
    """Advanced learning-based inverse kinematics for humanoid robots"""

    def __init__(self, robot_config: Dict):
        super().__init__()

        self.robot_config = robot_config
        self.state_dim = robot_config['state_dim']  # Current joint positions
        self.action_dim = robot_config['action_dim']  # Desired end-effector pose
        self.joint_count = robot_config['joint_count']

        # Main IK network
        self.ik_network = nn.Sequential(
            nn.Linear(6 + self.state_dim, 512),  # 6D pose + current state
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.joint_count)
        )

        # Singularity detection network
        self.singularity_detector = nn.Sequential(
            nn.Linear(6, 128),  # 6D pose input
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Singularity probability
        )

        # Redundancy resolution network
        self.redundancy_resolver = nn.Sequential(
            nn.Linear(self.joint_count + 6, 256),  # Current joints + desired pose
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.joint_count),
            nn.Tanh()  # Adjustment to joint angles
        )

        # Jacobian approximation network
        self.jacobian_approximator = nn.Sequential(
            nn.Linear(self.joint_count, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 6 * self.joint_count)  # 6xN Jacobian matrix
        )

    def forward(self, desired_pose: torch.Tensor,
               current_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute inverse kinematics solution"""
        # Combine inputs
        combined_input = torch.cat([desired_pose, current_state], dim=-1)

        # Compute initial joint angles
        joint_angles = self.ik_network(combined_input)

        # Detect potential singularities
        singularity_prob = self.singularity_detector(desired_pose)

        # Apply redundancy resolution if needed
        if singularity_prob.mean() > 0.5:  # High singularity probability
            redundancy_adjustment = self.redundancy_resolver(
                torch.cat([joint_angles, desired_pose], dim=-1)
            )
            joint_angles = joint_angles + redundancy_adjustment * 0.1

        # Compute Jacobian for stability
        jacobian_flat = self.jacobian_approximator(current_state[:, :self.joint_count])
        jacobian = jacobian_flat.view(-1, 6, self.joint_count)

        return {
            'joint_angles': joint_angles,
            'singularity_probability': singularity_prob,
            'jacobian': jacobian,
            'success': singularity_prob < 0.8  # Consider successful if not in singularity
        }

    def iterative_refinement(self, desired_pose: torch.Tensor,
                           current_state: torch.Tensor,
                           max_iterations: int = 10) -> torch.Tensor:
        """Iteratively refine IK solution"""
        joint_angles = self.forward(desired_pose, current_state)['joint_angles']

        for i in range(max_iterations):
            # Compute forward kinematics approximation
            current_pose = self._approximate_forward_kinematics(joint_angles)

            # Compute error
            pose_error = desired_pose - current_pose

            # If error is small enough, return
            if torch.norm(pose_error, dim=-1).mean() < 0.001:
                break

            # Use Jacobian transpose for correction
            jacobian = self.jacobian_approximator(joint_angles).view(-1, 6, self.joint_count)
            jacobian_transpose = jacobian.transpose(-2, -1)

            # Compute joint angle correction
            joint_correction = torch.bmm(jacobian_transpose,
                                       pose_error.unsqueeze(-1)).squeeze(-1)

            # Apply correction
            joint_angles = joint_angles + joint_correction * 0.1

        return joint_angles

    def _approximate_forward_kinematics(self, joint_angles: torch.Tensor) -> torch.Tensor:
        """Approximate forward kinematics (in practice, this would use actual FK)"""
        # This is a placeholder - in real implementation, you'd use actual forward kinematics
        # or learn the forward model
        return torch.cat([
            joint_angles[:, :3] * 0.1,  # Position approximation
            joint_angles[:, 3:6] * 0.05  # Orientation approximation
        ], dim=-1)

class LearningBasedIKTrainer:
    """Trainer for learning-based inverse kinematics"""

    def __init__(self, ik_model: AdvancedInverseKinematics):
        self.ik_model = ik_model
        self.optimizer = torch.optim.Adam(ik_model.parameters(), lr=1e-4)
        self.mse_loss = nn.MSELoss()
        self.singularity_loss_weight = 0.1

    def train_step(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        desired_poses = batch_data['desired_poses']
        current_states = batch_data['current_states']
        target_joints = batch_data['target_joints']

        # Forward pass
        ik_result = self.ik_model(desired_poses, current_states)
        predicted_joints = ik_result['joint_angles']

        # Main loss: joint angle prediction
        main_loss = self.mse_loss(predicted_joints, target_joints)

        # Singularity avoidance loss
        singularity_loss = ik_result['singularity_probability'].mean()

        # Total loss
        total_loss = main_loss + self.singularity_loss_weight * singularity_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'main_loss': main_loss.item(),
            'singularity_loss': singularity_loss.item(),
            'avg_singularity_prob': ik_result['singularity_probability'].mean().item()
        }
```

#### Force Control
Learning to apply appropriate forces for interaction with sophisticated approaches:

- **Contact Modeling**: Understanding environment properties
- **Adaptive Control**: Adjusting to changing conditions
- **Safety Requirements**: Ensuring safe interaction forces
- **Compliance Control**: Learning compliant behaviors
- **Haptic Feedback**: Incorporating tactile sensing
- **Impedance Control**: Learning impedance parameters
- **Force-Position Control**: Hybrid force-position control strategies

Advanced force control implementation:

```python
class AdvancedForceController(nn.Module):
    """Advanced learning-based force controller for humanoid robots"""

    def __init__(self, robot_config: Dict):
        super().__init__()

        self.robot_config = robot_config
        self.state_dim = robot_config['state_dim']
        self.action_dim = robot_config['action_dim']
        self.force_dim = robot_config.get('force_dim', 6)  # 6D force/torque

        # Force prediction network
        self.force_predictor = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.force_dim)
        )

        # Compliance learning network
        self.compliance_learner = nn.Sequential(
            nn.Linear(self.state_dim + self.force_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # Compliance parameters (Kx, Ky, Kz, Krx, Kry, Krz)
        )

        # Force error correction network
        self.error_corrector = nn.Sequential(
            nn.Linear(2 * self.force_dim, 256),  # Current + desired forces
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim)
        )

        # Safety constraint network
        self.safety_network = nn.Sequential(
            nn.Linear(self.state_dim + self.force_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Safety probability
        )

        # Impedance parameter learner
        self.impedance_learner = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 12)  # 6 for stiffness, 6 for damping
        )

    def forward(self, state: torch.Tensor,
               desired_force: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute force control action"""
        # Predict resulting forces
        predicted_forces = self.force_predictor(
            torch.cat([state, desired_force], dim=-1)
        )

        # Learn compliance parameters
        compliance_params = torch.exp(self.compliance_learner(
            torch.cat([state, desired_force], dim=-1)
        ))  # Exponential to ensure positive values

        # Compute force error correction
        force_error = desired_force - predicted_forces
        error_correction = self.error_corrector(
            torch.cat([predicted_forces, desired_force], dim=-1)
        )

        # Compute safety probability
        safety_prob = self.safety_network(
            torch.cat([state, desired_force], dim=-1)
        )

        # Learn impedance parameters
        impedance_params = self.impedance_learner(state)
        stiffness_params = impedance_params[:, :6]
        damping_params = impedance_params[:, 6:]

        return {
            'predicted_forces': predicted_forces,
            'compliance_params': compliance_params,
            'force_correction': error_correction,
            'safety_probability': safety_prob,
            'stiffness_params': stiffness_params,
            'damping_params': damping_params,
            'safe_action': safety_prob > 0.3  # Action is safe if probability > 0.3
        }

    def compute_impedance_control(self, state: torch.Tensor,
                                 desired_pose: torch.Tensor,
                                 external_force: torch.Tensor) -> torch.Tensor:
        """Compute impedance control action"""
        # Get impedance parameters
        impedance_result = self.forward(state, external_force)

        # Compute position error
        current_pose = state[:, :6]  # Assuming first 6 dims are pose
        position_error = desired_pose - current_pose

        # Apply impedance control law: F = K * (x_d - x) - D * v
        # where K is stiffness, D is damping
        stiffness = impedance_result['stiffness_params']
        damping = impedance_result['damping_params']

        # Assuming velocity is in state (after position)
        current_velocity = state[:, 6:12] if state.shape[1] > 6 else torch.zeros_like(position_error)

        impedance_force = (stiffness * position_error -
                          damping * current_velocity +
                          external_force)

        return impedance_force

class ForceControlSafetySystem:
    """Safety system for force control in humanoid robots"""

    def __init__(self, max_force_limits: torch.Tensor,
                 max_rate_limits: torch.Tensor):
        self.max_force_limits = max_force_limits
        self.max_rate_limits = max_rate_limits
        self.force_history = []

    def check_safety(self, commanded_force: torch.Tensor,
                    current_force: torch.Tensor) -> Dict[str, any]:
        """Check if force command is safe"""
        # Check force magnitude limits
        force_magnitude_violation = torch.any(
            torch.abs(commanded_force) > self.max_force_limits
        )

        # Check force rate limits
        if len(self.force_history) > 0:
            last_force = self.force_history[-1]
            force_rate = torch.abs(commanded_force - last_force)
            rate_violation = torch.any(force_rate > self.max_rate_limits)
        else:
            rate_violation = False

        # Compute safety score
        force_norm = torch.norm(commanded_force)
        safety_score = torch.sigmoid(5.0 - force_norm / self.max_force_limits.mean())

        safety_result = {
            'is_safe': not (force_magnitude_violation or rate_violation),
            'force_violation': force_magnitude_violation.item(),
            'rate_violation': rate_violation,
            'safety_score': safety_score.item(),
            'force_magnitude': force_norm.item()
        }

        # Update history
        self.force_history.append(commanded_force.clone())
        if len(self.force_history) > 10:  # Keep last 10 values
            self.force_history.pop(0)

        return safety_result

    def modify_for_safety(self, commanded_force: torch.Tensor) -> torch.Tensor:
        """Modify force command to ensure safety"""
        # Clamp to force limits
        safe_force = torch.clamp(commanded_force,
                               -self.max_force_limits,
                               self.max_force_limits)

        # Limit rate of change
        if len(self.force_history) > 0:
            last_force = self.force_history[-1]
            force_diff = safe_force - last_force
            rate_limited_diff = torch.clamp(force_diff,
                                         -self.max_rate_limits,
                                         self.max_rate_limits)
            safe_force = last_force + rate_limited_diff

        return safe_force
```

## Unsupervised Learning Applications

### Clustering for Behavior Discovery

#### Motion Primitives
Discovering fundamental movement patterns with advanced techniques:

- **Kinematic Clustering**: Grouping similar movement patterns
- **Temporal Clustering**: Identifying recurring temporal patterns
- **Hierarchical Clustering**: Discovering patterns at multiple scales
- **Validation**: Ensuring discovered patterns are meaningful
- **Dynamic Time Warping**: Clustering sequences of different lengths
- **Topological Clustering**: Preserving spatial relationships in motion space
- **Multi-modal Clustering**: Clustering based on multiple sensor modalities

Here's an implementation of advanced motion primitive discovery:

```python
import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

class AdvancedMotionPrimitiveDiscovery(nn.Module):
    """Advanced system for discovering motion primitives in humanoid robots"""

    def __init__(self, robot_config: Dict):
        super().__init__()

        self.robot_config = robot_config
        self.state_dim = robot_config['state_dim']
        self.action_dim = robot_config['action_dim']
        self.max_primitive_length = robot_config.get('max_primitive_length', 100)

        # Motion encoding network
        self.motion_encoder = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)  # Compressed motion representation
        )

        # Temporal modeling for variable-length sequences
        self.temporal_encoder = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        # Primitive classifier
        self.primitive_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, robot_config.get('num_primitives', 10))
        )

        # Reconstruction network for motion generation
        self.motion_decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.state_dim + self.action_dim)
        )

        # Clustering components
        self.kmeans = None
        self.dbscan = None
        self.pca = PCA(n_components=10)  # For dimensionality reduction

        self.motion_primitives = []
        self.primitive_embeddings = []

    def encode_motion_sequence(self, motion_sequence: torch.Tensor) -> torch.Tensor:
        """Encode a sequence of motion states into a fixed-size representation"""
        batch_size, seq_len, feature_dim = motion_sequence.shape

        # Reshape to apply encoder to each time step
        motion_flat = motion_sequence.view(-1, feature_dim)

        # Encode each time step
        encoded_flat = self.motion_encoder(motion_flat)

        # Reshape back to sequence format
        encoded_seq = encoded_flat.view(batch_size, seq_len, -1)

        # Apply temporal encoding
        temporal_encoded, (hidden, _) = self.temporal_encoder(encoded_seq)

        # Use final hidden state as sequence representation
        return hidden[-1]  # Use last layer's hidden state

    def cluster_motion_sequences(self, motion_sequences: List[torch.Tensor],
                                method: str = 'kmeans', n_clusters: int = 5) -> Dict[str, any]:
        """Cluster motion sequences to discover primitives"""

        # Encode all motion sequences
        encoded_sequences = []
        for seq in motion_sequences:
            encoded = self.encode_motion_sequence(seq.unsqueeze(0))
            encoded_sequences.append(encoded.squeeze(0).detach().cpu().numpy())

        encoded_array = np.array(encoded_sequences)

        # Apply dimensionality reduction
        reduced_features = self.pca.fit_transform(encoded_array)

        # Apply clustering
        if method == 'kmeans':
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = self.kmeans.fit_predict(reduced_features)
        elif method == 'dbscan':
            self.dbscan = DBSCAN(eps=0.5, min_samples=2)
            cluster_labels = self.dbscan.fit_predict(reduced_features)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        # Organize sequences by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append({
                'sequence': motion_sequences[i],
                'encoding': encoded_sequences[i],
                'index': i
            })

        # Store discovered primitives
        self.motion_primitives = clusters
        self.primitive_embeddings = self.kmeans.cluster_centers_ if self.kmeans else None

        return {
            'clusters': clusters,
            'labels': cluster_labels,
            'centers': self.primitive_embeddings,
            'silhouette_score': self._compute_silhouette_score(reduced_features, cluster_labels)
        }

    def _compute_silhouette_score(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Compute silhouette score for clustering quality"""
        from sklearn.metrics import silhouette_score
        if len(set(labels)) > 1:  # Need at least 2 clusters
            return silhouette_score(features, labels)
        return -1.0

    def generate_motion_primitive(self, primitive_id: int,
                                 length: int = 50) -> torch.Tensor:
        """Generate a motion primitive from its encoded representation"""
        if primitive_id not in self.motion_primitives:
            raise ValueError(f"Primitive {primitive_id} not found")

        # Use the cluster center to generate a new sequence
        if self.primitive_embeddings is not None:
            center = torch.tensor(self.primitive_embeddings[primitive_id],
                                dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            # Repeat the center for the desired length
            repeated_center = center.repeat(1, length, 1)

            # Decode to motion sequence
            decoded_motion = self.motion_decoder(repeated_center)

            return decoded_motion
        else:
            # Fallback: return a random sequence from the cluster
            cluster_sequences = self.motion_primitives[primitive_id]
            import random
            selected_seq = random.choice(cluster_sequences)['sequence']
            return selected_seq[:length] if len(selected_seq) > length else selected_seq

    def detect_new_behavior(self, new_sequence: torch.Tensor,
                           threshold: float = 0.8) -> Dict[str, any]:
        """Detect if a new sequence represents a new behavior pattern"""
        encoded_new = self.encode_motion_sequence(new_sequence.unsqueeze(0))
        encoded_new_np = encoded_new.squeeze(0).detach().cpu().numpy()

        # Transform using existing PCA
        try:
            reduced_new = self.pca.transform([encoded_new_np])
        except:
            # If PCA wasn't fitted, return that it's a new behavior
            return {'is_new': True, 'similarity': 0.0, 'closest_cluster': -1}

        # Find closest cluster center
        if self.kmeans is not None:
            distances = np.linalg.norm(
                self.kmeans.cluster_centers_ - reduced_new, axis=1
            )
            closest_cluster = np.argmin(distances)
            min_distance = np.min(distances)

            # Convert distance to similarity (inverse relationship)
            similarity = 1.0 / (1.0 + min_distance)

            is_new = similarity < threshold

            return {
                'is_new': is_new,
                'similarity': similarity,
                'closest_cluster': closest_cluster if not is_new else -1,
                'distance': min_distance
            }
        else:
            return {'is_new': True, 'similarity': 0.0, 'closest_cluster': -1}

class MotionPrimitiveOptimizer:
    """Optimize discovered motion primitives for execution"""

    def __init__(self, primitive_discoverer: AdvancedMotionPrimitiveDiscovery):
        self.discoverer = primitive_discoverer

        # Smoothness optimization network
        self.smoothness_optimizer = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def optimize_primitive(self, primitive_id: int,
                          smoothness_weight: float = 0.1) -> torch.Tensor:
        """Optimize a motion primitive for smooth execution"""
        # Get the primitive
        primitive = self.discoverer.generate_motion_primitive(primitive_id)

        # Apply optimization
        optimized = self.smoothness_optimizer(primitive)

        return optimized

    def blend_primitives(self, primitive_ids: List[int],
                        weights: List[float]) -> torch.Tensor:
        """Blend multiple primitives to create new behaviors"""
        if len(primitive_ids) != len(weights):
            raise ValueError("Number of primitives must match number of weights")

        # Get all primitives
        primitives = []
        for pid in primitive_ids:
            prim = self.discoverer.generate_motion_primitive(pid)
            primitives.append(prim)

        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()

        # Blend primitives
        blended = torch.zeros_like(primitives[0])
        for i, prim in enumerate(primitives):
            # Ensure same length for blending
            min_len = min(blended.shape[1], prim.shape[1])
            blended[:, :min_len, :] += prim[:, :min_len, :] * weights[i]

        return blended

# Example usage
def example_motion_primitive_discovery():
    """Example of using motion primitive discovery"""

    robot_config = {
        'state_dim': 20,
        'action_dim': 14,
        'max_primitive_length': 100,
        'num_primitives': 8
    }

    discoverer = AdvancedMotionPrimitiveDiscovery(robot_config)

    # Simulate some motion sequences
    motion_sequences = []
    for i in range(20):  # 20 motion sequences
        # Each sequence has random length between 30-80
        seq_len = np.random.randint(30, 80)
        seq = torch.randn(1, seq_len, 34)  # 20 state + 14 action dims
        motion_sequences.append(seq.squeeze(0))

    # Discover motion primitives
    results = discoverer.cluster_motion_sequences(
        motion_sequences, method='kmeans', n_clusters=8
    )

    print(f"Discovered {len(results['clusters'])} motion clusters")
    print(f"Clustering quality (silhouette score): {results['silhouette_score']:.3f}")

    # Generate a motion primitive
    if 0 in results['clusters']:
        generated_primitive = discoverer.generate_motion_primitive(0)
        print(f"Generated primitive shape: {generated_primitive.shape}")

    return discoverer, results
```

#### Environmental Modeling
Understanding environmental structures with sophisticated techniques:

- **Scene Segmentation**: Grouping similar environmental regions
- **Object Discovery**: Finding objects without prior knowledge
- **Activity Patterns**: Discovering recurring environmental activities
- **Anomaly Detection**: Identifying unusual environmental events
- **Dynamic Environment Modeling**: Modeling changing environments
- **Multi-sensor Fusion**: Combining information from multiple sensors
- **Topological Mapping**: Learning environmental topology

Advanced environmental modeling implementation:

```python
class AdvancedEnvironmentalModeler(nn.Module):
    """Advanced system for environmental modeling and understanding"""

    def __init__(self, robot_config: Dict):
        super().__init__()

        self.robot_config = robot_config
        self.state_dim = robot_config['state_dim']
        self.sensor_dims = robot_config.get('sensor_dims', {'rgb': 3, 'depth': 1, 'lidar': 1024})

        # Multi-modal feature extractor
        self.rgb_feature_extractor = nn.Sequential(
            nn.Conv2d(self.sensor_dims['rgb'], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 16, 256)
        )

        self.depth_feature_extractor = nn.Sequential(
            nn.Conv2d(self.sensor_dims['depth'], 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 16, 128)
        )

        self.lidar_feature_extractor = nn.Sequential(
            nn.Linear(self.sensor_dims['lidar'], 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        # Environmental state encoder
        self.env_encoder = nn.Sequential(
            nn.Linear(256 + 128 + 128, 512),  # Combined features
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)  # Compressed environmental representation
        )

        # Object discovery network
        self.object_discoverer = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, robot_config.get('max_objects', 20))  # Object existence probabilities
        )

        # Anomaly detection network
        self.anomaly_detector = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Anomaly probability
        )

        # Scene segmentation network
        self.segmentation_network = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, robot_config.get('num_regions', 10)),
            nn.Softmax(dim=-1)  # Region assignment probabilities
        )

        # Memory for temporal consistency
        self.environment_memory = []
        self.memory_size = 100

    def forward(self, sensor_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process sensor data to understand environment"""

        # Extract features from different sensors
        rgb_features = self.rgb_feature_extractor(sensor_data['rgb'])
        depth_features = self.depth_feature_extractor(sensor_data['depth'])
        lidar_features = self.lidar_feature_extractor(sensor_data['lidar'])

        # Combine features
        combined_features = torch.cat([rgb_features, depth_features, lidar_features], dim=1)

        # Encode environmental state
        env_encoding = self.env_encoder(combined_features)

        # Discover objects
        object_probabilities = self.object_discoverer(env_encoding)

        # Detect anomalies
        anomaly_probability = self.anomaly_detector(env_encoding)

        # Segment scene
        region_probabilities = self.segmentation_network(env_encoding)

        # Update memory
        self._update_memory(env_encoding)

        return {
            'environment_encoding': env_encoding,
            'object_probabilities': object_probabilities,
            'anomaly_probability': anomaly_probability,
            'region_probabilities': region_probabilities,
            'is_anomaly': anomaly_probability > 0.5
        }

    def _update_memory(self, env_encoding: torch.Tensor):
        """Update environmental memory for temporal consistency"""
        self.environment_memory.append(env_encoding.detach().clone())
        if len(self.environment_memory) > self.memory_size:
            self.environment_memory.pop(0)

    def detect_environment_change(self, current_encoding: torch.Tensor,
                                 threshold: float = 0.3) -> bool:
        """Detect if environment has significantly changed"""
        if not self.environment_memory:
            return False

        # Compute average of recent encodings
        recent_encodings = torch.stack(self.environment_memory[-10:])  # Last 10
        avg_encoding = recent_encodings.mean(dim=0)

        # Compute distance to current encoding
        distance = torch.norm(current_encoding - avg_encoding, dim=-1)

        return distance.mean().item() > threshold

    def build_topological_map(self, trajectory_data: List[Dict]) -> Dict[str, any]:
        """Build topological map from trajectory and sensor data"""

        nodes = []
        edges = []

        for i, data_point in enumerate(trajectory_data):
            # Process sensor data for this location
            env_info = self(data_point['sensors'])

            node = {
                'position': data_point['position'],
                'orientation': data_point['orientation'],
                'environment_encoding': env_info['environment_encoding'],
                'objects': torch.nonzero(env_info['object_probabilities'] > 0.5).squeeze(-1).tolist(),
                'regions': torch.argmax(env_info['region_probabilities']).item()
            }

            nodes.append(node)

            # Create edges to nearby nodes
            for j in range(max(0, i-5), i):  # Connect to 5 previous nodes max
                distance = torch.norm(
                    torch.tensor(data_point['position']) -
                    torch.tensor(trajectory_data[j]['position'])
                )

                if distance < 2.0:  # Within 2 meters
                    edges.append({
                        'from': j,
                        'to': i,
                        'distance': distance.item()
                    })

        return {
            'nodes': nodes,
            'edges': edges,
            'num_nodes': len(nodes),
            'num_edges': len(edges)
        }

class AnomalyDetectionSystem:
    """Advanced anomaly detection for robotic environments"""

    def __init__(self, model: AdvancedEnvironmentalModeler,
                 anomaly_threshold: float = 0.7):
        self.model = model
        self.anomaly_threshold = anomaly_threshold

        # Statistical models for different types of anomalies
        self.normal_distribution_params = {}  # Store mean and std for each feature
        self.temporal_patterns = {}  # Store temporal regularities

    def update_normal_model(self, sensor_data: Dict[str, torch.Tensor]):
        """Update model of normal environmental conditions"""
        with torch.no_grad():
            env_info = self.model(sensor_data)
            encoding = env_info['environment_encoding']

            if 'mean' not in self.normal_distribution_params:
                self.normal_distribution_params['mean'] = encoding.clone()
                self.normal_distribution_params['std'] = torch.zeros_like(encoding)
                self.normal_distribution_params['count'] = 1
            else:
                # Update running statistics
                old_mean = self.normal_distribution_params['mean']
                old_count = self.normal_distribution_params['count']

                new_count = old_count + 1
                new_mean = (old_mean * old_count + encoding) / new_count

                # Update standard deviation
                if old_count > 1:
                    old_var = self.normal_distribution_params['std'] ** 2
                    new_var = ((old_count - 1) * old_var +
                              old_count * (old_mean - new_mean) ** 2 +
                              (encoding - new_mean) ** 2) / new_count
                    self.normal_distribution_params['std'] = torch.sqrt(new_var)

                self.normal_distribution_params['mean'] = new_mean
                self.normal_distribution_params['count'] = new_count

    def detect_anomaly(self, sensor_data: Dict[str, torch.Tensor]) -> Dict[str, any]:
        """Detect anomalies in the environment"""
        with torch.no_grad():
            env_info = self.model(sensor_data)
            encoding = env_info['environment_encoding']

            # Check statistical anomaly
            if 'mean' in self.normal_distribution_params:
                mean = self.normal_distribution_params['mean']
                std = self.normal_distribution_params['std'] + 1e-8  # Avoid division by zero

                z_scores = torch.abs(encoding - mean) / std
                statistical_anomaly_score = torch.mean(z_scores).item()
            else:
                statistical_anomaly_score = 0.0

            # Check model-based anomaly
            model_anomaly_score = env_info['anomaly_probability'].item()

            # Combine scores
            combined_anomaly_score = 0.6 * model_anomaly_score + 0.4 * (statistical_anomaly_score / 3.0)

            is_anomaly = combined_anomaly_score > self.anomaly_threshold

            return {
                'is_anomaly': is_anomaly,
                'combined_score': combined_anomaly_score,
                'model_score': model_anomaly_score,
                'statistical_score': statistical_anomaly_score,
                'anomaly_type': self._classify_anomaly_type(env_info)
            }

    def _classify_anomaly_type(self, env_info: Dict[str, torch.Tensor]) -> str:
        """Classify the type of anomaly detected"""
        object_probs = env_info['object_probabilities']
        max_obj_prob, max_obj_idx = torch.max(object_probs, dim=1)

        if max_obj_prob.item() > 0.8:
            return f"new_object_{max_obj_idx.item()}"
        elif env_info['anomaly_probability'].item() > 0.8:
            return "structural_anomaly"
        else:
            return "minor_variation"

# Example usage
def example_environmental_modeling():
    """Example of environmental modeling"""

    robot_config = {
        'state_dim': 20,
        'sensor_dims': {'rgb': 3, 'depth': 1, 'lidar': 1024},
        'max_objects': 15,
        'num_regions': 8
    }

    modeler = AdvancedEnvironmentalModeler(robot_config)
    anomaly_detector = AnomalyDetectionSystem(modeler)

    # Simulate sensor data
    batch_size = 1
    sensor_data = {
        'rgb': torch.randn(batch_size, 3, 224, 224),
        'depth': torch.randn(batch_size, 1, 224, 224),
        'lidar': torch.randn(batch_size, 1024)
    }

    # Process environment
    env_info = modeler(sensor_data)
    print(f"Objects detected: {torch.sum(env_info['object_probabilities'] > 0.5).item()}")
    print(f"Anomaly probability: {env_info['anomaly_probability'].item():.3f}")

    # Update normal model and detect anomalies
    anomaly_detector.update_normal_model(sensor_data)
    anomaly_result = anomaly_detector.detect_anomaly(sensor_data)
    print(f"Anomaly detected: {anomaly_result['is_anomaly']}")
    print(f"Anomaly score: {anomaly_result['combined_score']:.3f}")

    return modeler, anomaly_detector
```

### Dimensionality Reduction

#### Feature Learning
Reducing data complexity while preserving important information with advanced techniques:

- **Principal Component Analysis (PCA)**: Linear dimensionality reduction
- **Autoencoders**: Non-linear dimensionality reduction with neural networks
- **Manifold Learning**: Preserving local geometric structure
- **Applications**: Sensor data compression, visualization
- **Variational Autoencoders**: Learning probabilistic representations
- **Sparse Coding**: Learning sparse representations
- **Non-negative Matrix Factorization**: Learning non-negative factors

Advanced dimensionality reduction implementation:

```python
class AdvancedDimensionalityReducer(nn.Module):
    """Advanced system for dimensionality reduction in robotics"""

    def __init__(self, input_dim: int, latent_dim: int = 64):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Variational Autoencoder components
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Separate mu and logvar for VAE
        self.mu_head = nn.Linear(128, latent_dim)
        self.logvar_head = nn.Linear(128, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

        # Sparse coding component
        self.sparse_coding = SparseCodingModule(latent_dim, input_dim)

        # Manifold learning components
        self.manifold_projector = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode input to latent space (VAE style)"""
        h = self.encoder(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space to input space"""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with multiple reduction methods"""
        # VAE encoding
        z, mu, logvar = self.encode(x)
        recon_x = self.decode(z)

        # Sparse coding
        sparse_z = self.sparse_coding.encode(x)
        sparse_recon = self.sparse_coding.decode(sparse_z)

        # Manifold projection
        manifold_z = self.manifold_projector(x)

        return {
            'vae_latent': z,
            'mu': mu,
            'logvar': logvar,
            'reconstruction': recon_x,
            'sparse_latent': sparse_z,
            'sparse_reconstruction': sparse_recon,
            'manifold_latent': manifold_z,
            'vae_recon_error': torch.mean((x - recon_x) ** 2),
            'sparse_recon_error': torch.mean((x - sparse_recon) ** 2)
        }

    def compute_kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence for VAE loss"""
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return torch.mean(kl_loss)

class SparseCodingModule(nn.Module):
    """Module for sparse coding and dictionary learning"""

    def __init__(self, latent_dim: int, input_dim: int, sparsity_reg: float = 0.01):
        super().__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.sparsity_reg = sparsity_reg

        # Dictionary matrix (learnable)
        self.dictionary = nn.Parameter(torch.randn(latent_dim, input_dim) * 0.1)

        # Sparsity-promoting encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to sparse representation"""
        # Initial encoding
        z_init = self.encoder(x)

        # Apply soft thresholding for sparsity
        z_sparse = self._soft_threshold(z_init, threshold=0.1)

        return z_sparse

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode using dictionary"""
        # Reconstruct using dictionary
        recon = torch.matmul(z, self.dictionary)
        return recon

    def _soft_threshold(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        """Apply soft thresholding for sparsity"""
        return torch.sign(x) * torch.relu(torch.abs(x) - threshold)

class FeatureExtractorWithMemory(nn.Module):
    """Feature extractor with temporal memory for consistency"""

    def __init__(self, reducer: AdvancedDimensionalityReducer,
                 memory_size: int = 100):
        super().__init__()

        self.reducer = reducer
        self.memory_size = memory_size

        # Memory for temporal consistency
        self.feature_memory = []
        self.latent_memory = []

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features with temporal consistency"""
        results = self.reducer(x)

        # Store in memory
        self._update_memory(results)

        # Apply temporal smoothing if memory is available
        if len(self.latent_memory) > 1:
            smoothed_latent = self._temporal_smoothing(results['vae_latent'])
            results['smoothed_latent'] = smoothed_latent

        return results

    def _update_memory(self, results: Dict[str, torch.Tensor]):
        """Update feature memories"""
        self.feature_memory.append(results['vae_latent'].detach().clone())
        if len(self.feature_memory) > self.memory_size:
            self.feature_memory.pop(0)

    def _temporal_smoothing(self, current_latent: torch.Tensor) -> torch.Tensor:
        """Apply temporal smoothing to latent features"""
        if len(self.feature_memory) < 2:
            return current_latent

        # Use running average of recent features
        recent_features = torch.stack(self.feature_memory[-5:])  # Last 5
        avg_features = torch.mean(recent_features, dim=0)

        # Blend current with historical average
        alpha = 0.7  # Weight for current observation
        smoothed = alpha * current_latent + (1 - alpha) * avg_features

        return smoothed

class MultiScaleFeatureExtractor(nn.Module):
    """Extract features at multiple scales for comprehensive representation"""

    def __init__(self, input_dim: int, scales: List[int] = [1, 2, 4]):
        super().__init__()

        self.scales = scales
        self.extractors = nn.ModuleDict()

        for scale in scales:
            # Each scale has its own encoder
            self.extractors[f'scale_{scale}'] = nn.Sequential(
                nn.Linear(input_dim, 256 // scale),
                nn.ReLU(),
                nn.Linear(256 // scale, 128 // scale),
                nn.ReLU(),
                nn.Linear(128 // scale, 64 // scale)
            )

        # Fusion network to combine multi-scale features
        total_latent_dim = sum(64 // scale for scale in scales)
        self.fusion_network = nn.Sequential(
            nn.Linear(total_latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract and fuse multi-scale features"""
        scale_features = []

        for scale in self.scales:
            # Process at different scale
            scale_feat = self.extractors[f'scale_{scale}'](x)
            scale_features.append(scale_feat)

        # Concatenate all scale features
        concatenated = torch.cat(scale_features, dim=1)

        # Fuse to final representation
        fused_features = self.fusion_network(concatenated)

        return {
            'multi_scale_features': scale_features,
            'fused_features': fused_features,
            'individual_scale_features': dict(zip(
                [f'scale_{s}' for s in self.scales], scale_features
            ))
        }

# Example usage
def example_dimensionality_reduction():
    """Example of advanced dimensionality reduction"""

    input_dim = 1024  # High-dimensional sensor data
    latent_dim = 64   # Compressed representation

    reducer = AdvancedDimensionalityReducer(input_dim, latent_dim)
    memory_reducer = FeatureExtractorWithMemory(reducer)
    multi_scale_extractor = MultiScaleFeatureExtractor(input_dim)

    # Simulate high-dimensional input
    batch_size = 32
    x = torch.randn(batch_size, input_dim)

    # Apply dimensionality reduction
    results = reducer(x)
    print(f"Original dim: {input_dim}, Latent dim: {results['vae_latent'].shape[1]}")
    print(f"Reconstruction error: {results['vae_recon_error'].item():.4f}")

    # With memory
    memory_results = memory_reducer(x)
    print(f"Memory-enhanced features shape: {memory_results['smoothed_latent'].shape}")

    # Multi-scale extraction
    multi_scale_results = multi_scale_extractor(x)
    print(f"Multi-scale fused features shape: {multi_scale_results['fused_features'].shape}")

    return reducer, memory_reducer, multi_scale_extractor
```

## Reinforcement Learning for Robotics

### Markov Decision Processes (MDPs)

#### MDP Formulation
Formal framework for sequential decision making with advanced considerations:

- **States (S)**: Complete description of environment state
- **Actions (A)**: Set of possible actions
- **Transition Model (P)**: Probability of state transitions
- **Reward Function (R)**: Reward for state-action pairs
- **Discount Factor (γ)**: Trade-off between immediate and future rewards
- **Terminal States**: States where episodes end
- **Stochastic Transitions**: Probabilistic state transitions
- **Partially Observable MDPs (POMDPs)**: Dealing with uncertain observations

Here's an implementation of advanced MDP formulation for robotics:

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import gym
from gym import spaces

class AdvancedMDPEnvironment:
    """Advanced MDP environment for humanoid robotics with realistic dynamics"""

    def __init__(self, robot_config: Dict):
        self.config = robot_config
        self.state_dim = robot_config['state_dim']
        self.action_dim = robot_config['action_dim']
        self.max_episode_length = robot_config.get('max_episode_length', 1000)

        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

        # Robot dynamics parameters
        self.mass_matrix = torch.diag(torch.tensor(robot_config.get('mass', [1.0] * self.action_dim)))
        self.damping_matrix = torch.diag(torch.tensor(robot_config.get('damping', [0.1] * self.action_dim)))
        self.gravity_compensation = torch.tensor(robot_config.get('gravity_compensation', [0.0] * self.action_dim))

        # Safety constraints
        self.position_limits = torch.tensor(robot_config.get('position_limits', [[-np.pi, np.pi]] * self.action_dim))
        self.velocity_limits = torch.tensor(robot_config.get('velocity_limits', [[-5.0, 5.0]] * self.action_dim))

        # Current state
        self.current_state = torch.zeros(self.state_dim)
        self.current_action = torch.zeros(self.action_dim)
        self.step_count = 0
        self.episode_reward = 0.0

    def reset(self) -> torch.Tensor:
        """Reset environment to initial state"""
        # Initialize with random state within reasonable bounds
        self.current_state = torch.randn(self.state_dim) * 0.1
        self.current_action = torch.zeros(self.action_dim)
        self.step_count = 0
        self.episode_reward = 0.0

        return self.current_state

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, Dict]:
        """Execute action and return transition tuple"""
        # Validate action
        action = torch.clamp(action, -1.0, 1.0)

        # Apply safety constraints
        action = self._apply_safety_constraints(action)

        # Update robot dynamics
        next_state = self._update_dynamics(self.current_state, action)

        # Compute reward
        reward = self._compute_reward(self.current_state, action, next_state)

        # Check termination
        done = self._check_termination(next_state)

        # Update counters
        self.current_state = next_state
        self.current_action = action
        self.step_count += 1
        self.episode_reward += reward

        # Additional info
        info = {
            'step_count': self.step_count,
            'episode_reward': self.episode_reward,
            'action_magnitude': torch.norm(action).item(),
            'safety_violations': self._check_safety_violations(next_state)
        }

        return next_state, reward, done, info

    def _update_dynamics(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Update robot dynamics with realistic physics"""
        # Extract state components (assuming state contains position, velocity)
        current_pos = state[:self.action_dim]
        current_vel = state[self.action_dim:2*self.action_dim] if len(state) >= 2*self.action_dim else torch.zeros_like(current_pos)

        # Apply control action (torque-based control)
        torque = action * self.config.get('max_torque', 10.0)

        # Compute acceleration using inverse dynamics
        # τ = M(q)q̈ + C(q,q̇)q̇ + G(q)  =>  q̈ = M⁻¹(τ - C*q̇ - G)
        # Simplified: q̈ = M⁻¹(τ - D*q̇ - G_gravity)
        inv_mass = torch.inverse(self.mass_matrix)
        damping_force = torch.matmul(self.damping_matrix, current_vel.unsqueeze(1)).squeeze(1)
        gravity_compensation = self.gravity_compensation

        acceleration = torch.matmul(inv_mass, (torque - damping_force - gravity_compensation).unsqueeze(1)).squeeze(1)

        # Update state with numerical integration (Euler method)
        new_vel = current_vel + acceleration * self.config.get('dt', 0.01)
        new_pos = current_pos + new_vel * self.config.get('dt', 0.01)

        # Construct new state
        new_state = torch.cat([new_pos, new_vel] + ([state[2*self.action_dim:]] if len(state) > 2*self.action_dim else []))

        return new_state

    def _compute_reward(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> float:
        """Compute reward based on task objectives and constraints"""
        reward = 0.0

        # Task-specific reward (example: reaching a target)
        target_pos = torch.tensor(self.config.get('target_position', [0.0] * self.action_dim)[:self.action_dim])
        pos_error = torch.norm(target_pos - next_state[:self.action_dim])
        reward -= pos_error.item()  # Negative reward for distance from target

        # Penalty for large actions
        action_penalty = torch.norm(action).item() * 0.01
        reward -= action_penalty

        # Bonus for progress toward target
        prev_pos_error = torch.norm(target_pos - state[:self.action_dim])
        if pos_error < prev_pos_error:
            reward += 0.1  # Small bonus for making progress

        # Safety penalty
        safety_violations = self._check_safety_violations(next_state)
        reward -= safety_violations * 10.0

        return reward

    def _check_termination(self, state: torch.Tensor) -> bool:
        """Check if episode should terminate"""
        # Check episode length
        if self.step_count >= self.max_episode_length:
            return True

        # Check for safety violations
        if self._check_safety_violations(state) > 0:
            return True

        # Check for invalid state values
        if torch.any(torch.isnan(state)) or torch.any(torch.isinf(state)):
            return True

        return False

    def _apply_safety_constraints(self, action: torch.Tensor) -> torch.Tensor:
        """Apply safety constraints to actions"""
        # Limit action magnitude
        action = torch.clamp(action, -1.0, 1.0)

        return action

    def _check_safety_violations(self, state: torch.Tensor) -> int:
        """Check for safety violations"""
        violations = 0

        # Check position limits
        pos = state[:self.action_dim]
        for i in range(len(pos)):
            if pos[i] < self.position_limits[i, 0] or pos[i] > self.position_limits[i, 1]:
                violations += 1

        # Check velocity limits
        if len(state) >= 2 * self.action_dim:
            vel = state[self.action_dim:2*self.action_dim]
            for i in range(len(vel)):
                if vel[i] < self.velocity_limits[i, 0] or vel[i] > self.velocity_limits[i, 1]:
                    violations += 1

        return violations

class POMDPWrapper:
    """Wrapper for handling Partially Observable MDPs"""

    def __init__(self, mdp_env: AdvancedMDPEnvironment, observation_noise: float = 0.1):
        self.mdp_env = mdp_env
        self.observation_noise = observation_noise

        # Memory for belief state tracking
        self.belief_state = None
        self.observation_history = []

    def reset(self) -> torch.Tensor:
        """Reset environment and belief state"""
        true_state = self.mdp_env.reset()
        self.belief_state = true_state.clone()
        self.observation_history = []

        # Add observation noise
        noisy_observation = self._add_observation_noise(true_state)
        self.observation_history.append(noisy_observation)

        return noisy_observation

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, Dict]:
        """Step with partial observability"""
        true_next_state, reward, done, info = self.mdp_env.step(action)

        # Update belief state based on action and observation
        self._update_belief_state(action, true_next_state)

        # Get noisy observation
        noisy_observation = self._add_observation_noise(true_next_state)
        self.observation_history.append(noisy_observation)

        # Add belief state information to info
        info['belief_state'] = self.belief_state
        info['observation_uncertainty'] = torch.var(torch.stack(self.observation_history), dim=0).mean().item()

        return noisy_observation, reward, done, info

    def _add_observation_noise(self, true_state: torch.Tensor) -> torch.Tensor:
        """Add noise to observations"""
        noise = torch.randn_like(true_state) * self.observation_noise
        return true_state + noise

    def _update_belief_state(self, action: torch.Tensor, observation: torch.Tensor):
        """Update belief state using Bayesian filtering"""
        # Simplified Kalman filter update
        if self.belief_state is None:
            self.belief_state = observation.clone()
        else:
            # Simple averaging approach
            alpha = 0.1  # Learning rate
            self.belief_state = (1 - alpha) * self.belief_state + alpha * observation

class RobotMDPSolver:
    """Solver for robot MDP problems with advanced techniques"""

    def __init__(self, mdp_env: AdvancedMDPEnvironment, gamma: float = 0.99):
        self.env = mdp_env
        self.gamma = gamma

        # Value function approximators
        self.value_network = nn.Sequential(
            nn.Linear(self.env.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.policy_network = nn.Sequential(
            nn.Linear(self.env.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.env.action_dim),
            nn.Tanh()  # Actions in [-1, 1]
        )

        # Optimizers
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=1e-3)
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=1e-4)

        # Experience replay buffer
        self.replay_buffer = []
        self.buffer_size = 100000

    def compute_bellman_backup(self, state: torch.Tensor, reward: float,
                              next_state: torch.Tensor, done: bool) -> torch.Tensor:
        """Compute Bellman backup for value function learning"""
        with torch.no_grad():
            if done:
                target_value = torch.tensor([[reward]], dtype=torch.float32)
            else:
                next_value = self.value_network(next_state.unsqueeze(0))
                target_value = torch.tensor([[reward + self.gamma * next_value.item()]], dtype=torch.float32)

        return target_value

    def policy_gradient_update(self, state: torch.Tensor, action: torch.Tensor,
                              advantage: torch.Tensor):
        """Update policy using policy gradient"""
        action_probs = self.policy_network(state.unsqueeze(0))

        # Compute log probability (assuming Gaussian policy for continuous actions)
        log_prob = -0.5 * torch.sum((action - action_probs) ** 2)

        # Policy gradient loss
        policy_loss = -(log_prob * advantage.detach()).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

    def value_function_update(self, state: torch.Tensor, target_value: torch.Tensor):
        """Update value function"""
        predicted_value = self.value_network(state.unsqueeze(0))
        value_loss = nn.MSELoss()(predicted_value, target_value)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

# Example usage
def example_mdp_formulation():
    """Example of using advanced MDP formulation"""

    robot_config = {
        'state_dim': 20,  # 7 joint positions + 7 velocities + 6 additional states
        'action_dim': 14,  # 7 joint torques + 7 additional controls
        'max_episode_length': 500,
        'mass': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] + [0.5] * 7,  # Joint masses
        'damping': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1] + [0.05] * 7,  # Damping
        'position_limits': [[-2.0, 2.0]] * 14,  # Joint limits
        'velocity_limits': [[-5.0, 5.0]] * 14,  # Velocity limits
        'max_torque': 10.0,
        'dt': 0.01,
        'target_position': [0.0] * 14
    }

    # Create environment
    env = AdvancedMDPEnvironment(robot_config)

    # Create POMDP wrapper for partially observable scenarios
    pomdp_env = POMDPWrapper(env, observation_noise=0.05)

    # Create solver
    solver = RobotMDPSolver(env)

    print(f"Environment created with state dim: {env.state_dim}, action dim: {env.action_dim}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    return env, pomdp_env, solver
```

#### Policy Optimization
Finding optimal action-selection strategies with advanced techniques:

- **Policy**: Mapping from states to actions
- **Value Function**: Expected future reward from state
- **Optimality**: Bellman equations for optimal value functions
- **Convergence**: Conditions for algorithm convergence
- **Actor-Critic Methods**: Combining policy and value learning
- **Trust Region Methods**: Ensuring stable policy updates
- **Function Approximation**: Learning policies for continuous spaces

Advanced policy optimization implementation:

```python
class AdvancedPolicyOptimizer:
    """Advanced policy optimization for robotic control"""

    def __init__(self, state_dim: int, action_dim: int, gamma: float = 0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        # Actor network (policy)
        self.actor = GaussianPolicyNetwork(state_dim, action_dim)

        # Critic network (value function)
        self.critic = ValueNetwork(state_dim)

        # Target networks for stability
        self.actor_target = GaussianPolicyNetwork(state_dim, action_dim)
        self.critic_target = ValueNetwork(state_dim)

        # Copy parameters to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        # Hyperparameters for advanced techniques
        self.alpha = 0.2  # Entropy coefficient for soft actor-critic
        self.automatic_entropy_tuning = True
        self.target_entropy = -action_dim

        if self.automatic_entropy_tuning:
            self.log_alpha = torch.log(torch.tensor(alpha))
            self.log_alpha.requires_grad_(True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-4)

        # Experience replay
        self.replay_buffer = []
        self.buffer_size = 1000000

        # Normalization parameters
        self.state_mean = torch.zeros(state_dim)
        self.state_std = torch.ones(state_dim)
        self.update_count = 0

    def select_action(self, state: torch.Tensor, evaluate: bool = False) -> torch.Tensor:
        """Select action using current policy"""
        state = self._normalize_state(state)

        if evaluate:
            with torch.no_grad():
                action, _, _ = self.actor(state.unsqueeze(0))
                return action.squeeze(0).detach()
        else:
            with torch.no_grad():
                action, _, _ = self.actor(state.unsqueeze(0))
                return action.squeeze(0).detach()

    def update(self, batch_size: int = 256) -> Dict[str, float]:
        """Update policy and value networks"""
        if len(self.replay_buffer) < batch_size:
            return {}

        # Sample batch from replay buffer
        batch_indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in batch_indices]

        # Extract batch components
        states = torch.stack([torch.tensor(transition['state'], dtype=torch.float32) for transition in batch])
        actions = torch.stack([torch.tensor(transition['action'], dtype=torch.float32) for transition in batch])
        rewards = torch.tensor([transition['reward'] for transition in batch], dtype=torch.float32).unsqueeze(1)
        next_states = torch.stack([torch.tensor(transition['next_state'], dtype=torch.float32) for transition in batch])
        dones = torch.tensor([transition['done'] for transition in batch], dtype=torch.float32).unsqueeze(1)

        # Normalize states
        states = self._normalize_state_batch(states)
        next_states = self._normalize_state_batch(next_states)

        # Compute target Q-values for critic update
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor_target(next_states)

            # Compute target Q-value using soft actor-critic approach
            next_q_values = self.critic_target(next_states)
            next_v_values = next_q_values - self.alpha * next_log_probs

            target_q_values = rewards + (1 - dones) * self.gamma * next_v_values

        # Critic update
        current_q_values = self.critic(states)
        critic_loss = nn.MSELoss()(current_q_values, target_q_values.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        current_actions, current_log_probs, _ = self.actor(states)
        current_q_values = self.critic(states)

        # Compute actor loss (entropy-regularized)
        actor_loss = (self.alpha * current_log_probs - current_q_values).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update alpha if using automatic entropy tuning
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (current_log_probs + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()

        # Soft update target networks
        self._soft_update_targets()

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'alpha_loss': alpha_loss.item() if self.automatic_entropy_tuning else 0.0,
            'alpha': self.alpha.item() if self.automatic_entropy_tuning else self.alpha,
            'batch_size': batch_size
        }

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Normalize state using running statistics"""
        return (state - self.state_mean) / (self.state_std + 1e-8)

    def _normalize_state_batch(self, states: torch.Tensor) -> torch.Tensor:
        """Normalize batch of states"""
        return (states - self.state_mean.unsqueeze(0)) / (self.state_std.unsqueeze(0) + 1e-8)

    def _soft_update_targets(self, tau: float = 0.005):
        """Soft update target networks"""
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def add_experience(self, state: np.ndarray, action: np.ndarray,
                      reward: float, next_state: np.ndarray, done: bool):
        """Add experience to replay buffer"""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }

        self.replay_buffer.append(experience)
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)

        # Update normalization statistics
        self._update_normalization_stats(state)

    def _update_normalization_stats(self, state: np.ndarray):
        """Update running statistics for state normalization"""
        state_tensor = torch.tensor(state, dtype=torch.float32)
        self.update_count += 1

        if self.update_count == 1:
            self.state_mean = state_tensor
        else:
            # Incremental update of mean and std
            delta = state_tensor - self.state_mean
            self.state_mean += delta / self.update_count
            self.state_std = torch.sqrt(
                ((self.update_count - 1) * self.state_std.pow(2) +
                 self.update_count * (self.update_count - 1) * (delta / self.update_count).pow(2)) /
                self.update_count
            )

class GaussianPolicyNetwork(nn.Module):
    """Gaussian policy network for continuous action spaces"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Mean and log std head
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        # Initialize log std to reasonable range
        self.log_std_head.weight.data.uniform_(-1e-3, 1e-3)
        self.log_std_head.bias.data.uniform_(-1e-3, 1e-3)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning action, log probability, and entropy"""
        h = self.net(state)

        mean = torch.tanh(self.mean_head(h))  # Actions in [-1, 1]
        log_std = torch.clamp(self.log_std_head(h), -20, 2)
        std = torch.exp(log_std)

        # Reparameterization trick for sampling
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # For backpropagation
        action = torch.tanh(x_t)  # Squash to [-1, 1]

        # Compute log probability with correction for tanh squashing
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)  # Correction term
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # Compute entropy
        entropy = 0.5 + 0.5 * torch.log(2 * torch.tensor(np.pi)) + log_std
        entropy = entropy.sum(dim=-1, keepdim=True)

        return action, log_prob, entropy

class ValueNetwork(nn.Module):
    """Value network for estimating state values"""

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass to estimate state value"""
        return self.net(state)

class TrustRegionPolicyOptimizer(AdvancedPolicyOptimizer):
    """Policy optimizer using trust region methods for stable updates"""

    def __init__(self, state_dim: int, action_dim: int, gamma: float = 0.99):
        super().__init__(state_dim, action_dim, gamma)

        # TRPO/PPO specific parameters
        self.clip_epsilon = 0.2
        self.kl_beta = 1.0  # Coefficient for KL divergence penalty
        self.max_kl_div = 0.01  # Maximum KL divergence allowed

    def ppo_update(self, states: torch.Tensor, actions: torch.Tensor,
                   old_log_probs: torch.Tensor, advantages: torch.Tensor) -> Dict[str, float]:
        """Update policy using PPO (Proximal Policy Optimization)"""

        # Current policy evaluation
        _, current_log_probs, _ = self.actor(states)

        # Compute ratio
        ratio = torch.exp(current_log_probs - old_log_probs.detach())

        # Compute surrogate objectives
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        # Add KL divergence penalty
        with torch.no_grad():
            _, old_log_probs_detached, _ = self.actor(states)

        kl_div = torch.mean(old_log_probs_detached - current_log_probs)
        actor_loss += self.kl_beta * kl_div

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40.0)  # Gradient clipping
        self.actor_optimizer.step()

        return {
            'ppo_actor_loss': actor_loss.item(),
            'kl_divergence': kl_div.item(),
            'clip_fraction': (torch.abs(ratio - 1.0) > self.clip_epsilon).float().mean().item()
        }

    def trpo_update(self, states: torch.Tensor, actions: torch.Tensor,
                    old_log_probs: torch.Tensor, advantages: torch.Tensor) -> Dict[str, float]:
        """Update policy using TRPO (Trust Region Policy Optimization)"""
        # TRPO implementation would involve conjugate gradient and line search
        # This is a simplified version

        # Compute policy gradient
        _, current_log_probs, _ = self.actor(states)
        ratio = torch.exp(current_log_probs - old_log_probs.detach())
        surrogate_loss = -(ratio * advantages).mean()

        # Compute KL divergence
        with torch.no_grad():
            _, old_log_probs_detached, _ = self.actor(states)

        kl_div = torch.mean(old_log_probs_detached - current_log_probs)

        # TRPO constraint: KL divergence should be less than max_kl_div
        if kl_div.item() > self.max_kl_div:
            # Adjust learning rate or reject update
            return {
                'trpo_actor_loss': 0.0,
                'kl_divergence': kl_div.item(),
                'update_accepted': False
            }

        # Update actor with constraint
        self.actor_optimizer.zero_grad()
        surrogate_loss.backward()
        self.actor_optimizer.step()

        return {
            'trpo_actor_loss': surrogate_loss.item(),
            'kl_divergence': kl_div.item(),
            'update_accepted': True
        }
```

### Deep Reinforcement Learning

#### Deep Q-Networks (DQN)
Combining Q-learning with deep neural networks for continuous control with advanced techniques:

- **Experience Replay**: Storing and replaying past experiences
- **Target Network**: Stable target for learning
- **Epsilon-Greedy**: Exploration-exploitation trade-off
- **Applications**: Discrete action control tasks
- **Double DQN**: Reducing overestimation bias
- **Dueling DQN**: Separating value and advantage estimation
- **Prioritized Experience Replay**: Focusing on important experiences
- **Multi-step Learning**: Incorporating longer-term consequences

Advanced DQN implementation for robotics:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from typing import Dict, List, Tuple, Optional

class DuelingDQN(nn.Module):
    """Dueling Deep Q-Network for advanced value estimation"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512):
        super().__init__()

        # Common feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # State value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # State value V(s)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)  # Advantages A(s,a)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(state)

        # Compute value and advantages
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Combine to get Q-values: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values

class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer with proportional prioritization"""

    def __init__(self, capacity: int, alpha: float = 0.6, beta_start: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization is used
        self.beta_start = beta_start  # Importance sampling weight
        self.beta = beta_start

        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.pos = 0

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool, priority: float = None):
        """Add experience to buffer with priority"""
        if priority is None:
            priority = 1.0  # Default priority for new experiences

        # Convert to numpy arrays if they're tensors
        if torch.is_tensor(state):
            state = state.cpu().numpy()
        if torch.is_tensor(next_state):
            next_state = next_state.cpu().numpy()

        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(priority)

    def sample(self, batch_size: int, beta: float = None) -> Tuple[
        List[np.ndarray], List[int], List[float], List[np.ndarray], List[bool], List[float], List[int]
    ]:
        """Sample batch with prioritization"""
        if beta is None:
            beta = self.beta

        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)

        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for sampled experiences"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self) -> int:
        return len(self.buffer)

class AdvancedDQN:
    """Advanced DQN implementation with multiple improvements"""

    def __init__(self, state_dim: int, action_dim: int,
                 lr: float = 1e-4, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995, target_update_freq: int = 1000):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.q_network = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_network = DuelingDQN(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

        # Update target network
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Parameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.action_dim = action_dim
        self.update_count = 0

        # Replay buffer with prioritization
        self.replay_buffer = PrioritizedReplayBuffer(100000)

        # Double DQN flag
        self.double_dqn = True

    def act(self, state: torch.Tensor, training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            state = state.unsqueeze(0).to(self.device)
            q_values = self.q_network(state)
            return q_values.argmax().item()

    def update_epsilon(self):
        """Decay epsilon"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update(self, batch_size: int = 32) -> Dict[str, float]:
        """Update network parameters"""
        if len(self.replay_buffer) < batch_size:
            return {}

        # Sample from prioritized replay buffer
        states, actions, rewards, next_states, dones, weights, indices = \
            self.replay_buffer.sample(batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        # Current Q values
        current_q_values = self.q_network(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Next Q values using target network
        next_q_values = self.target_network(next_states)

        if self.double_dqn:
            # Double DQN: use main network to select actions, target network to evaluate
            next_actions = self.q_network(next_states).argmax(dim=1)
            next_q_values = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        else:
            # Regular DQN: use target network for both selection and evaluation
            next_q_values = next_q_values.max(dim=1)[0]

        # Compute target Q values
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # Compute loss with importance sampling weights
        td_errors = (current_q_values - target_q_values.detach()).abs()
        loss = (weights * (td_errors ** 2)).mean()

        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        self.optimizer.step()

        # Update priorities in replay buffer
        priorities = td_errors.cpu().numpy()
        self.replay_buffer.update_priorities(indices, priorities)

        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Update epsilon
        self.update_epsilon()

        return {
            'loss': loss.item(),
            'epsilon': self.epsilon,
            'td_error_mean': td_errors.mean().item()
        }

class MultiStepDQN(AdvancedDQN):
    """Multi-step DQN for longer-term credit assignment"""

    def __init__(self, state_dim: int, action_dim: int, n_step: int = 3,
                 **kwargs):
        super().__init__(state_dim, action_dim, **kwargs)
        self.n_step = n_step
        self.n_step_buffer = []

    def push_to_nstep_buffer(self, state: torch.Tensor, action: int,
                           reward: float, next_state: torch.Tensor, done: bool):
        """Add to n-step buffer"""
        self.n_step_buffer.append((state, action, reward, next_state, done))

        if len(self.n_step_buffer) >= self.n_step:
            # Compute n-step return
            n_step_return = 0.0
            for i, (_, _, r, _, _) in enumerate(self.n_step_buffer):
                n_step_return += (self.gamma ** i) * r

            # Get the first state and action, and the nth state
            first_state, first_action, _, _, _ = self.n_step_buffer[0]
            _, _, _, last_next_state, last_done = self.n_step_buffer[-1]

            # Add to replay buffer with n-step return
            self.replay_buffer.push(
                first_state.numpy(), first_action, n_step_return,
                last_next_state.numpy(), last_done
            )

            # Remove the first transition
            self.n_step_buffer.pop(0)

# Example usage
def example_advanced_dqn():
    """Example of using advanced DQN techniques"""

    state_dim = 20
    action_dim = 8  # Discrete actions for robotics

    dqn = AdvancedDQN(state_dim, action_dim)
    multi_step_dqn = MultiStepDQN(state_dim, action_dim, n_step=5)

    print(f"DQN initialized with state dim: {state_dim}, action dim: {action_dim}")
    print(f"Using device: {dqn.device}")

    return dqn, multi_step_dqn
```

#### Policy Gradient Methods
Directly optimizing the policy function with advanced techniques:

- **REINFORCE**: Basic policy gradient algorithm
- **Actor-Critic**: Combining policy and value learning
- **Trust Region Methods**: Ensuring stable updates
- **Continuous Actions**: Handling continuous action spaces
- **Generalized Advantage Estimation (GAE)**: Better advantage estimation
- **Natural Policy Gradients**: Incorporating curvature information
- **Maximum Entropy RL**: Balancing exploration and exploitation

Advanced policy gradient implementation:

```python
class GeneralizedAdvantageEstimation:
    """Generalized Advantage Estimation for better advantage calculation"""

    def __init__(self, gamma: float = 0.99, lam: float = 0.95):
        self.gamma = gamma
        self.lam = lam

    def compute_advantages(self, rewards: List[float], values: List[float],
                          dones: List[bool]) -> List[float]:
        """Compute advantages using GAE"""
        advantages = []
        gae = 0.0

        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0.0 if dones[i] else values[i]
            else:
                next_value = values[i + 1]

            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lam * (1 - dones[i]) * gae
            advantages.insert(0, gae)

        return advantages

class ContinuousActorCritic(nn.Module):
    """Actor-Critic for continuous action spaces"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor (policy) network
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))

        # Critic (value) network
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning action mean, logstd, and state value"""
        features = self.feature_extractor(state)

        # Actor: mean and std for Gaussian policy
        action_mean = torch.tanh(self.actor_mean(features))  # Actions in [-1, 1]
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)

        # Critic: state value
        state_value = self.critic(features)

        return action_mean, action_std, state_value

    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy"""
        action_mean, action_std, _ = self.forward(state)

        # Create distribution and sample
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.rsample()  # Reparameterization trick
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

        return action, log_prob

class AdvancedPolicyGradient:
    """Advanced policy gradient with multiple improvements"""

    def __init__(self, state_dim: int, action_dim: int, lr_actor: float = 3e-4,
                 lr_critic: float = 1e-3, gamma: float = 0.99, gae_lambda: float = 0.95):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.actor_critic = ContinuousActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer_actor = torch.optim.Adam(
            self.actor_critic.feature_extractor.parameters(), lr=lr_actor
        )
        self.optimizer_actor.add_param_group({
            'params': self.actor_critic.actor_mean.parameters(), 'lr': lr_actor
        })
        self.optimizer_actor.add_param_group({
            'params': [self.actor_critic.actor_logstd], 'lr': lr_actor
        })

        self.optimizer_critic = torch.optim.Adam(
            list(self.actor_critic.feature_extractor.parameters()) +
            [self.actor_critic.critic.weight, self.actor_critic.critic.bias],
            lr=lr_critic
        )

        # Parameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.gae_calculator = GeneralizedAdvantageEstimation(gamma, gae_lambda)

        # Storage for rollout
        self.rollout_storage = []

    def act(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from current policy"""
        state = state.to(self.device)
        action, log_prob = self.actor_critic.get_action(state.unsqueeze(0))
        return action.squeeze(0).detach(), log_prob.squeeze(0).detach()

    def put_experience(self, state: torch.Tensor, action: torch.Tensor,
                      reward: float, next_state: torch.Tensor, done: bool,
                      log_prob: torch.Tensor, value: torch.Tensor):
        """Store experience in rollout buffer"""
        self.rollout_storage.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob,
            'value': value
        })

    def update(self) -> Dict[str, float]:
        """Update policy and value networks"""
        if not self.rollout_storage:
            return {}

        # Extract data
        states = torch.stack([exp['state'] for exp in self.rollout_storage])
        actions = torch.stack([exp['action'] for exp in self.rollout_storage])
        rewards = [exp['reward'] for exp in self.rollout_storage]
        dones = [exp['done'] for exp in self.rollout_storage]
        old_log_probs = torch.stack([exp['log_prob'] for exp in self.rollout_storage])
        values = torch.stack([exp['value'] for exp in self.rollout_storage]).squeeze(-1)

        # Compute advantages using GAE
        advantages = self.gae_calculator.compute_advantages(rewards, values.tolist(), dones)
        advantages = torch.FloatTensor(advantages).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute returns
        returns = []
        R = 0
        for reward, done in zip(rewards[::-1], dones[::-1]):
            R = reward + self.gamma * R * (1 - done)
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).to(self.device)

        # Update networks
        states = states.to(self.device)
        actions = actions.to(self.device)

        # Actor update
        _, _, new_values = self.actor_critic(states)
        new_values = new_values.squeeze(-1)

        # Get new action probabilities
        with torch.no_grad():
            action_means, action_stds, _ = self.actor_critic(states)
            dist = torch.distributions.Normal(action_means, action_stds)
            new_log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)

        # Ratio and surrogate loss
        ratios = torch.exp(new_log_probs - old_log_probs.detach())
        surr1 = ratios * advantages.unsqueeze(-1)
        surr2 = torch.clamp(ratios, 0.8, 1.2) * advantages.unsqueeze(-1)
        actor_loss = -torch.min(surr1, surr2).mean()

        # Critic update
        critic_loss = F.mse_loss(new_values, returns)

        # Update networks
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 40)
        self.optimizer_actor.step()

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # Clear rollout storage
        self.rollout_storage = []

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'advantage_mean': advantages.mean().item(),
            'return_mean': returns.mean().item()
        }

class NaturalPolicyGradient(AdvancedPolicyGradient):
    """Natural Policy Gradient with Fisher Information Matrix"""

    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-3,
                 damping: float = 0.01, **kwargs):
        super().__init__(state_dim, action_dim, **kwargs)
        self.damping = damping
        self.fisher_matrix = None
        self.fisher_inv = None

    def compute_fisher_information_matrix(self, states: torch.Tensor,
                                        old_actions: torch.Tensor,
                                        old_log_probs: torch.Tensor) -> torch.Tensor:
        """Compute Fisher Information Matrix"""
        # Compute policy gradient
        _, _, values = self.actor_critic(states)
        advantages = self.gae_calculator.compute_advantages(
            [1.0] * len(states), values.squeeze(-1).tolist(), [False] * len(states)
        )
        advantages = torch.FloatTensor(advantages).to(self.device)

        # Compute log probabilities for current policy
        action_means, action_stds, _ = self.actor_critic(states)
        dist = torch.distributions.Normal(action_means, action_stds)
        new_log_probs = dist.log_prob(old_actions).sum(dim=-1, keepdim=True)

        # Compute likelihood ratio
        ratios = torch.exp(new_log_probs - old_log_probs.detach())

        # Compute Fisher matrix using empirical approximation
        # This is a simplified version - in practice, you'd use conjugate gradient
        policy_params = list(self.actor_critic.actor_mean.parameters()) + [self.actor_critic.actor_logstd]
        flat_grads = []

        for i in range(len(states)):
            self.optimizer_actor.zero_grad()
            loss = -new_log_probs[i]
            loss.backward(retain_graph=True)

            # Flatten gradients
            grads = []
            for param in policy_params:
                if param.grad is not None:
                    grads.append(param.grad.view(-1))
            flat_grad = torch.cat(grads)
            flat_grads.append(flat_grad)

        flat_grads = torch.stack(flat_grads)

        # Compute Fisher matrix: E[grad log π * grad log π^T]
        fisher_matrix = torch.mean(
            flat_grads.unsqueeze(-1) * flat_grads.unsqueeze(-2), dim=0
        )

        return fisher_matrix

    def natural_gradient_update(self, states: torch.Tensor, actions: torch.Tensor,
                              advantages: torch.Tensor) -> Dict[str, float]:
        """Update using natural gradient"""
        # Compute regular gradient
        action_means, action_stds, _ = self.actor_critic(states)
        dist = torch.distributions.Normal(action_means, action_stds)
        new_log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)

        # Policy loss
        policy_loss = -(new_log_probs * advantages.unsqueeze(-1)).mean()

        # Compute gradients
        self.optimizer_actor.zero_grad()
        policy_loss.backward()
        regular_grad = torch.cat([
            param.grad.view(-1) if param.grad is not None else torch.zeros(param.numel())
            for param in list(self.actor_critic.actor_mean.parameters()) + [self.actor_critic.actor_logstd]
        ])

        # Compute Fisher matrix and natural gradient
        fisher_matrix = self.compute_fisher_information_matrix(states, actions, new_log_probs)

        # Add damping and compute natural gradient: F^{-1} * grad
        identity = torch.eye(fisher_matrix.shape[0]).to(self.device)
        fisher_reg = fisher_matrix + self.damping * identity

        # Solve for natural gradient: F^{-1} * grad
        try:
            natural_grad = torch.solve(regular_grad.unsqueeze(-1), fisher_reg)[0].squeeze(-1)
        except:
            # If matrix is singular, fall back to regular gradient
            natural_grad = regular_grad

        # Update parameters using natural gradient
        idx = 0
        for param in list(self.actor_critic.actor_mean.parameters()) + [self.actor_critic.actor_logstd]:
            param_count = param.numel()
            param_update = natural_grad[idx:idx + param_count].view(param.shape)
            param.data -= 0.01 * param_update  # Learning rate for natural gradient
            idx += param_count

        return {
            'policy_loss': policy_loss.item(),
            'natural_gradient_norm': natural_grad.norm().item()
        }
```

#### Advanced RL Algorithms

##### Proximal Policy Optimization (PPO)
- **Trust Region**: Constrained policy updates
- **Advantage Estimation**: Generalized advantage estimation
- **Clipping**: Preventing large policy updates
- **Stability**: More stable than other policy gradient methods
- **Sample Efficiency**: Better sample efficiency than TRPO
- **Multiple Epochs**: Multiple updates per sample batch

Advanced PPO implementation:

```python
class PPOAgent:
    """Proximal Policy Optimization agent with advanced features"""

    def __init__(self, state_dim: int, action_dim: int,
                 lr_actor: float = 3e-4, lr_critic: float = 1e-3,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2, ppo_epochs: int = 10,
                 mini_batch_size: int = 64, entropy_coef: float = 0.01):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.actor_critic = ContinuousActorCritic(state_dim, action_dim).to(self.device)

        # Separate optimizers for actor and critic
        self.optimizer_actor = torch.optim.Adam(
            list(self.actor_critic.feature_extractor.parameters()) +
            list(self.actor_critic.actor_mean.parameters()) +
            [self.actor_critic.actor_logstd],
            lr=lr_actor
        )

        self.optimizer_critic = torch.optim.Adam(
            list(self.actor_critic.feature_extractor.parameters()) +
            [self.actor_critic.critic.weight, self.actor_critic.critic.bias],
            lr=lr_critic
        )

        # Parameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.entropy_coef = entropy_coef

        # Storage
        self.storage = []
        self.gae_calculator = GeneralizedAdvantageEstimation(gamma, gae_lambda)

    def act(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action and return action, log probability, and value"""
        state = state.to(self.device).unsqueeze(0)
        action_mean, action_std, state_value = self.actor_critic(state)

        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

        return action.squeeze(0).detach(), log_prob.squeeze(0).detach(), state_value.squeeze(0).detach()

    def put_experience(self, state: torch.Tensor, action: torch.Tensor,
                      reward: float, next_state: torch.Tensor, done: bool,
                      log_prob: torch.Tensor, value: torch.Tensor):
        """Store experience in storage"""
        self.storage.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob,
            'value': value
        })

    def update(self) -> Dict[str, float]:
        """Update using PPO algorithm"""
        if len(self.storage) == 0:
            return {}

        # Extract all data
        states = torch.stack([exp['state'] for exp in self.storage])
        actions = torch.stack([exp['action'] for exp in self.storage])
        rewards = [exp['reward'] for exp in self.storage]
        dones = [exp['done'] for exp in self.storage]
        old_log_probs = torch.stack([exp['log_prob'] for exp in self.storage])
        old_values = torch.stack([exp['value'] for exp in self.storage]).squeeze(-1)

        # Compute advantages and returns
        advantages = self.gae_calculator.compute_advantages(rewards, old_values.tolist(), dones)
        advantages = torch.FloatTensor(advantages).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute returns
        returns = []
        R = 0
        for reward, done in zip(rewards[::-1], dones[::-1]):
            R = reward + self.gamma * R * (1 - done)
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).to(self.device)

        # Convert to device
        states = states.to(self.device)
        actions = actions.to(self.device)

        # PPO update with multiple epochs
        actor_losses = []
        critic_losses = []

        for _ in range(self.ppo_epochs):
            # Create mini batches
            batch_size = len(states)
            indices = torch.randperm(batch_size)

            for start_idx in range(0, batch_size, self.mini_batch_size):
                end_idx = min(start_idx + self.mini_batch_size, batch_size)
                batch_indices = indices[start_idx:end_idx]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Actor update
                action_means, action_stds, batch_values = self.actor_critic(batch_states)
                dist = torch.distributions.Normal(action_means, action_stds)

                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1, keepdim=True)
                entropy = dist.entropy().sum(dim=-1, keepdim=True)

                # Compute ratio
                ratios = torch.exp(new_log_probs - batch_old_log_probs.detach())

                # Compute surrogate losses
                surr1 = ratios * batch_advantages.unsqueeze(-1)
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages.unsqueeze(-1)
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()

                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 40)
                self.optimizer_actor.step()

                # Critic update
                critic_loss = F.mse_loss(batch_values.squeeze(-1), batch_returns)

                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                self.optimizer_critic.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())

        # Clear storage
        self.storage = []

        return {
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses),
            'ppo_epochs_run': self.ppo_epochs,
            'mini_batches_processed': (len(states) + self.mini_batch_size - 1) // self.mini_batch_size * self.ppo_epochs
        }

class AdaptivePPO(PPOAgent):
    """PPO with adaptive hyperparameters"""

    def __init__(self, state_dim: int, action_dim: int, **kwargs):
        super().__init__(state_dim, action_dim, **kwargs)

        # Adaptive parameters
        self.clip_epsilon_base = kwargs.get('clip_epsilon', 0.2)
        self.entropy_coef_base = kwargs.get('entropy_coef', 0.01)
        self.kl_target = 0.01
        self.kl_alpha = 2.0  # Scaling factor for KL divergence adaptation

        # KL divergence tracking
        self.kl_history = deque(maxlen=100)

    def update(self) -> Dict[str, float]:
        """Update with adaptive hyperparameters based on KL divergence"""
        if len(self.storage) == 0:
            return {}

        # Calculate KL divergence before update
        with torch.no_grad():
            states = torch.stack([exp['state'] for exp in self.storage]).to(self.device)
            old_actions = torch.stack([exp['action'] for exp in self.storage]).to(self.device)

            old_means, old_stds, _ = self.actor_critic(states)
            old_dist = torch.distributions.Normal(old_means, old_stds)

            new_means, new_stds, _ = self.actor_critic(states)
            new_dist = torch.distributions.Normal(new_means, new_stds)

            kl_div = torch.distributions.kl_divergence(old_dist, new_dist).mean().item()
            self.kl_history.append(kl_div)

        # Adapt hyperparameters based on average KL divergence
        if len(self.kl_history) > 10:
            avg_kl = np.mean(list(self.kl_history)[-10:])

            # Adjust clip epsilon
            if avg_kl > self.kl_target * self.kl_alpha:
                self.clip_epsilon = max(0.05, self.clip_epsilon_base / 1.5)
            elif avg_kl < self.kl_target / self.kl_alpha:
                self.clip_epsilon = min(0.5, self.clip_epsilon_base * 1.5)

            # Adjust entropy coefficient
            if avg_kl > self.kl_target * 2:
                self.entropy_coef = min(0.1, self.entropy_coef_base * 1.2)
            elif avg_kl < self.kl_target / 2:
                self.entropy_coef = max(1e-4, self.entropy_coef_base / 1.2)

        # Perform regular PPO update
        result = super().update()

        # Add adaptive parameter info
        result.update({
            'clip_epsilon': self.clip_epsilon,
            'entropy_coef': self.entropy_coef,
            'avg_kl_divergence': np.mean(list(self.kl_history)[-10:]) if self.kl_history else 0.0
        })

        return result
```

##### Soft Actor-Critic (SAC)
- **Maximum Entropy**: Balancing reward and exploration
- **Off-policy**: Learning from past experiences
- **Continuous Control**: Excellent for continuous action spaces
- **Sample Efficiency**: More efficient than many alternatives
- **Temperature Control**: Automatic entropy temperature adjustment
- **Twin Critics**: Reducing overestimation bias
- **Target Entropy**: Balancing exploration and exploitation

Advanced SAC implementation:

```python
class SoftActorCritic(nn.Module):
    """Soft Actor-Critic with advanced features"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 alpha: float = 0.2, automatic_entropy_tuning: bool = True):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.automatic_entropy_tuning = automatic_entropy_tuning

        # Actor network (policy)
        self.actor = GaussianPolicyNetwork(state_dim, action_dim, hidden_dim)

        # Twin critics (Q-functions)
        self.critic_1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.critic_2 = QNetwork(state_dim, action_dim, hidden_dim)

        # Target networks
        self.critic_1_target = QNetwork(state_dim, action_dim, hidden_dim)
        self.critic_2_target = QNetwork(state_dim, action_dim, hidden_dim)

        # Copy weights to target networks
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        # Entropy temperature
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(action_dim)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=3e-4)
        else:
            self.alpha = alpha
            self.target_entropy = None

    def select_action(self, state: torch.Tensor, evaluate: bool = False) -> torch.Tensor:
        """Select action using current policy"""
        if evaluate:
            with torch.no_grad():
                action, _, _ = self.actor(state.unsqueeze(0))
                return action.squeeze(0).detach()
        else:
            action, _, _ = self.actor(state.unsqueeze(0))
            return action.squeeze(0).detach()

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for policy evaluation"""
        return self.actor(state)

class QNetwork(nn.Module):
    """Q-network for SAC"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Concatenate state and action
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute Q-value"""
        sa = torch.cat([state, action], dim=1)
        return self.net(sa)

class AdvancedSAC:
    """Advanced Soft Actor-Critic with multiple improvements"""

    def __init__(self, state_dim: int, action_dim: int,
                 lr_actor: float = 3e-4, lr_critic: float = 3e-4, lr_alpha: float = 3e-4,
                 gamma: float = 0.99, tau: float = 0.005, alpha: float = 0.2,
                 automatic_entropy_tuning: bool = True, target_update_interval: int = 1):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # SAC agent
        self.agent = SoftActorCritic(
            state_dim, action_dim, alpha=alpha,
            automatic_entropy_tuning=automatic_entropy_tuning
        ).to(self.device)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.agent.actor.parameters(), lr=lr_actor)
        self.critic_1_optimizer = torch.optim.Adam(self.agent.critic_1.parameters(), lr=lr_critic)
        self.critic_2_optimizer = torch.optim.Adam(self.agent.critic_2.parameters(), lr=lr_critic)

        # Parameters
        self.gamma = gamma
        self.tau = tau
        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = automatic_entropy_tuning

        # Update counter
        self.update_step = 0

        # Replay buffer
        self.replay_buffer = []
        self.buffer_size = 1000000

    def act(self, state: torch.Tensor, evaluate: bool = False) -> torch.Tensor:
        """Select action using current policy"""
        state = state.to(self.device)
        return self.agent.select_action(state, evaluate)

    def update(self, batch_size: int = 256) -> Dict[str, float]:
        """Update SAC networks"""
        if len(self.replay_buffer) < batch_size:
            return {}

        # Sample batch
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]

        # Extract batch
        states = torch.stack([torch.tensor(exp['state'], dtype=torch.float32) for exp in batch]).to(self.device)
        actions = torch.stack([torch.tensor(exp['action'], dtype=torch.float32) for exp in batch]).to(self.device)
        rewards = torch.tensor([exp['reward'] for exp in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.stack([torch.tensor(exp['next_state'], dtype=torch.float32) for exp in batch]).to(self.device)
        dones = torch.tensor([exp['done'] for exp in batch], dtype=torch.float32, device=self.device).unsqueeze(1)

        # Compute next actions and entropies
        next_actions, next_log_probs, _ = self.agent.actor(next_states)

        # Compute target Q values
        with torch.no_grad():
            next_q1 = self.agent.critic_1_target(next_states, next_actions)
            next_q2 = self.agent.critic_2_target(next_states, next_actions)
            min_next_q = torch.min(next_q1, next_q2)

            # Temperature-adjusted target
            if self.automatic_entropy_tuning:
                alpha = self.agent.log_alpha.exp()
            else:
                alpha = self.agent.alpha

            target_q = rewards + (1 - dones) * self.gamma * (min_next_q - alpha * next_log_probs)

        # Critic loss
        current_q1 = self.agent.critic_1(states, actions)
        current_q2 = self.agent.critic_2(states, actions)

        critic_1_loss = F.mse_loss(current_q1, target_q)
        critic_2_loss = F.mse_loss(current_q2, target_q)

        # Update critics
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # Actor loss
        current_actions, current_log_probs, _ = self.agent.actor(states)
        current_q1 = self.agent.critic_1(states, current_actions)
        current_q2 = self.agent.critic_2(states, current_actions)
        min_current_q = torch.min(current_q1, current_q2)

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.agent.log_alpha * (current_log_probs + self.agent.target_entropy).detach()).mean()

            self.agent.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.agent.alpha_optim.step()

            alpha = self.agent.log_alpha.exp()
        else:
            alpha_loss = torch.tensor(0.0)
            alpha = self.agent.alpha

        # Actor loss with entropy regularization
        actor_loss = (alpha * current_log_probs - min_current_q).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        if self.update_step % self.target_update_interval == 0:
            for target_param, param in zip(self.agent.critic_1_target.parameters(), self.agent.critic_1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for target_param, param in zip(self.agent.critic_2_target.parameters(), self.agent.critic_2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.update_step += 1

        return {
            'actor_loss': actor_loss.item(),
            'critic_1_loss': critic_1_loss.item(),
            'critic_2_loss': critic_2_loss.item(),
            'alpha_loss': alpha_loss.item() if self.automatic_entropy_tuning else 0.0,
            'alpha': alpha.item() if self.automatic_entropy_tuning else alpha,
            'batch_size': batch_size
        }

# Example usage
def example_advanced_rl():
    """Example of using advanced RL algorithms"""

    state_dim = 20
    action_dim = 14  # Continuous action space for robotics

    # Create agents
    ppo_agent = PPOAgent(state_dim, action_dim)
    sac_agent = AdvancedSAC(state_dim, action_dim)
    adaptive_ppo = AdaptivePPO(state_dim, action_dim)

    print(f"PPO Agent created with state dim: {state_dim}, action dim: {action_dim}")
    print(f"SAC Agent created with automatic entropy tuning: {sac_agent.automatic_entropy_tuning}")

    return ppo_agent, sac_agent, adaptive_ppo
```

### Challenges in Robotic RL

#### Sample Efficiency
Robots need to learn efficiently from limited experience:

- **Simulation**: Learning in simulation before real-world deployment
- **Transfer Learning**: Applying learned skills to new tasks
- **Curriculum Learning**: Structured learning progression
- **Meta-Learning**: Learning to learn quickly

#### Safety During Learning
Ensuring safe exploration and learning:

- **Safe Exploration**: Constrained exploration to safe regions
- **Shielding**: Formal safety guarantees during learning
- **Human-in-the-Loop**: Human oversight during learning
- **Constraint Learning**: Learning safety constraints from demonstrations

#### Reality Gap
Difference between simulation and real-world performance:

- **Domain Randomization**: Varying simulation parameters
- **System Identification**: Understanding real-world dynamics
- **Adaptive Control**: Adjusting to real-world conditions
- **Sim-to-Real Transfer**: Techniques for bridging the gap

## Imitation Learning

### Learning from Demonstrations

#### Behavior Cloning
Directly learning to map observations to actions:

- **Supervised Learning**: Treating demonstrations as training data
- **Neural Networks**: Deep neural networks for complex mappings
- **Limitations**: Covariate shift problem, error accumulation
- **Applications**: Simple manipulation and locomotion tasks

#### Inverse Reinforcement Learning (IRL)
Learning the reward function from expert demonstrations:

- **Principle**: Inferring what expert is optimizing
- **Algorithms**: Maximum entropy IRL, generative adversarial IRL
- **Advantages**: Learning complex behaviors not captured by simple metrics
- **Challenges**: Reward function ambiguity, computational complexity

#### Generative Adversarial Imitation Learning (GAIL)
Combining GANs with imitation learning:

- **Discriminator**: Distinguishes expert from learner behavior
- **Generator**: Learner policy trying to fool discriminator
- **Advantages**: No need to explicitly define reward function
- **Applications**: Complex motor skill learning

### Learning Complex Skills

#### Bimanual Coordination
Learning to coordinate two arms for complex tasks:

- **Multi-agent Approach**: Treating arms as separate agents
- **Centralized Control**: Coordinating arms with central controller
- **Hierarchical Learning**: Learning coordination at multiple levels
- **Applications**: Complex manipulation tasks

#### Tool Use
Learning to use tools effectively:

- **Affordance Learning**: Understanding object affordances
- **Tool Dynamics**: Learning tool-environment interactions
- **Skill Transfer**: Applying learned skills to new tools
- **Safety**: Ensuring safe tool use

## Multi-Task and Transfer Learning

### Multi-Task Learning

#### Shared Representations
Learning multiple tasks simultaneously:

- **Parameter Sharing**: Sharing neural network parameters
- **Hard Parameter Sharing**: Shared lower layers, task-specific outputs
- **Soft Parameter Sharing**: Similar but not identical parameters
- **Benefits**: Improved generalization, reduced overfitting

#### Learning Order Optimization
Determining optimal order for learning multiple tasks:

- **Curriculum Learning**: Structured learning progression
- **Dependency Analysis**: Learning prerequisite skills first
- **Transfer Analysis**: Identifying beneficial skill transfers
- **Optimization**: Optimizing learning sequence for efficiency

### Transfer Learning

#### Domain Transfer
Transferring knowledge between different environments:

- **Domain Adaptation**: Adapting to new environmental conditions
- **Domain Generalization**: Learning robust representations
- **Adversarial Adaptation**: Using adversarial training for adaptation
- **Applications**: Indoor to outdoor, simulation to reality

#### Task Transfer
Transferring skills between different tasks:

- **Behavior Transfer**: Applying learned behaviors to new tasks
- **Feature Transfer**: Transferring learned representations
- **Model Transfer**: Transferring learned models
- **Meta-Learning**: Learning to transfer quickly

## Learning from Human Interaction

### Social Learning
Learning through interaction with humans:

#### Active Learning
Robot queries human for information:

- **Uncertainty Sampling**: Querying for most uncertain examples
- **Query by Committee**: Using multiple models to identify queries
- **Expected Model Change**: Querying to maximize learning
- **Efficiency**: Reducing human effort in training

#### Interactive Learning
Learning through human-robot interaction:

- **Reinforcement Learning from Human Feedback**: Learning from human rewards
- **Cooperative Learning**: Humans and robots learning together
- **Natural Interaction**: Learning through natural human-robot interaction
- **Adaptation**: Adapting to individual human preferences

### Learning Social Behaviors
Acquiring appropriate social behaviors:

- **Social Norms**: Learning appropriate social responses
- **Personalization**: Adapting to individual humans
- **Cultural Adaptation**: Adapting to cultural contexts
- **Emotional Intelligence**: Learning to respond to emotions

## Learning in Continuous Environments

### Online Learning
Learning from data that arrives sequentially:

#### Incremental Learning
Updating models with new data without forgetting old knowledge:

- **Catastrophic Forgetting**: Problem of forgetting old tasks
- **Elastic Weight Consolidation**: Protecting important weights
- **Progressive Networks**: Adding new networks for new tasks
- **Rehearsal Methods**: Retraining on old data

#### Adaptive Learning
Adjusting to changing environments and conditions:

- **Concept Drift**: Adapting to changing data distributions
- **Online Algorithms**: Algorithms that adapt continuously
- **Change Detection**: Detecting when adaptation is needed
- **Stability-Plasticity**: Balancing adaptation and stability

### Lifelong Learning
Maintaining performance across many tasks over time:

#### Continual Learning
Learning new tasks without forgetting old ones:

- **Architectural Methods**: Growing network architecture
- **Regularization Methods**: Constraining changes to protect old knowledge
- **Rehearsal Methods**: Combining old and new experiences
- **Dynamic Networks**: Adapting network structure over time

#### Curriculum Learning
Structured learning progression:

- **Self-Paced Learning**: Automatically determining learning pace
- **Automatic Curriculum**: Learning optimal task sequences
- **Adaptive Difficulty**: Adjusting task difficulty based on performance
- **Transfer Optimization**: Optimizing for maximum positive transfer

## Implementation Considerations

### Real-time Learning
Learning algorithms that operate in real-time:

#### Computational Requirements
- **Latency**: Fast processing for real-time interaction
- **Throughput**: Processing data streams efficiently
- **Memory**: Managing memory usage for continuous operation
- **Power**: Efficient computation for battery-powered robots

#### Online vs. Batch Learning
- **Online Learning**: Updating models with each new example
- **Batch Learning**: Updating models with accumulated data
- **Mini-batch Learning**: Compromise between online and batch
- **Trade-offs**: Accuracy vs. speed vs. stability

### Safety and Robustness

#### Safe Learning
Ensuring learning algorithms operate safely:

- **Safety Constraints**: Incorporating safety requirements into learning
- **Robustness**: Maintaining performance under uncertainty
- **Validation**: Verifying learned policies before deployment
- **Monitoring**: Continuously monitoring learned behavior

#### Failure Handling
Managing learning algorithm failures:

- **Graceful Degradation**: Maintaining basic functionality
- **Fallback Policies**: Safe behaviors when learning fails
- **Error Detection**: Identifying when learning is not working
- **Recovery**: Returning to safe operation after failures

## Evaluation and Validation

### Performance Metrics

#### Learning Efficiency
- **Sample Efficiency**: Performance improvement per sample
- **Convergence Speed**: Time to reach desired performance
- **Asymptotic Performance**: Final performance level
- **Stability**: Consistency of learning progress

#### Generalization Performance
- **Cross-Validation**: Performance on held-out data
- **Domain Generalization**: Performance on new domains
- **Transfer Performance**: Performance on related tasks
- **Robustness**: Performance under environmental variations

### Benchmarking
Standardized evaluation protocols:

- **Simulation Environments**: Standardized testing platforms
- **Real-world Tasks**: Consistent evaluation protocols
- **Comparison Metrics**: Standard metrics for algorithm comparison
- **Reproducibility**: Ensuring results can be reproduced

## Future Directions

### Advanced Learning Paradigms

#### Neuromorphic Learning
Learning algorithms inspired by brain mechanisms:

- **Spiking Neural Networks**: Event-based neural computation
- **Synaptic Plasticity**: Biologically-inspired learning rules
- **Neuromorphic Hardware**: Specialized hardware for brain-like computation
- **Energy Efficiency**: Dramatically reduced power consumption

#### Quantum Machine Learning
Leveraging quantum computing for learning:

- **Quantum Advantage**: Potential exponential speedups
- **Quantum Algorithms**: Quantum versions of classical algorithms
- **Hybrid Systems**: Combining classical and quantum computation
- **Current Limitations**: Limited quantum hardware availability

### Integration Trends

#### End-to-End Learning
Learning complete robot behaviors from raw sensory input:

- **Deep Perception-Action**: Direct mapping from sensors to actions
- **Hierarchical Learning**: Learning at multiple levels of abstraction
- **Multi-Modal Integration**: Learning to integrate different sensory modalities
- **Challenges**: Sample efficiency and interpretability

#### Human-Robot Collaborative Learning
Humans and robots learning together:

- **Mutual Learning**: Both human and robot learning from interaction
- **Social Learning**: Learning through social interaction
- **Cultural Learning**: Acquiring cultural knowledge and norms
- **Trust Building**: Developing trust through learning interaction

Learning algorithms represent the key to creating truly adaptive and capable humanoid robots. As these algorithms continue to advance, humanoid robots will become increasingly capable of learning complex skills, adapting to new situations, and providing valuable assistance in human environments.