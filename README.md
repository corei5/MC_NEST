# MC-NEST-GPT

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![OpenAI API](https://img.shields.io/badge/OpenAI-GPT--4-blue)](https://platform.openai.com/)

MC_NEST is a **Monte Carlo Tree Search** framework enhanced with **GPT-4** designed to tackle complex mathematical problems. Inspired by the principles of Monte Carlo Tree Self-Refinement [1], MC_NEST represents a powerful fusion of AI-driven reasoning and algorithmic precision. The framework leverages the **GPT-4** API to refine, evaluate, and improve answers iteratively using **Nash Equilibrium-based exploration strategies** and **importance sampling techniques**.

## Features

- **GPT-4 Integration**: Harness the reasoning power of GPT-4 to solve and refine mathematical problems.
- **Monte Carlo Tree Search (MCTS)**: An advanced implementation of MCTS with configurable selection policies and initialization strategies.
- **Multi-Agent Critique & Refinement**: GPT-based agents for answer refinement and critique through structured JSON responses.
- **Customizable Selection Policies**:
  - **GREEDY**
  - **IMPORTANCE SAMPLING**
  - **PAIRWISE IMPORTANCE SAMPLING**
- **Reward System**: Uses strict GPT-based evaluation with penalties for overestimation.
- **Visualization**: Built-in tree-printing utility to debug and analyze the MCTS process.

## Installation

To get started, clone the repository and install the dependencies:

```bash
git clone https://github.com/corei5/MC_NEST_GPT.git
cd MC_NEST_GPT
pip install -r requirements.txt
```

## Usage

Import the Package

```bash
from MC_NEST_GPT.MC_NEST_GPT.MC_NEST_ import MC_NEST_gpt4o
import openai
```

## Set Up OpenAI API Key

Provide your OpenAI API key either directly or through environment variables:

```bash
openai.api_key = 'your_openai_api_key'
```

## Initialize and Run MC_NEST

Define your problem, configure MCTS settings, and execute:

```bash
GREEDY = 1
IMPORTANCE_SAMPLING = 2
PAIRWISE_IMPORTANCE_SAMPLING = 3
ZERO_SHOT = 1
DUMMY_ANSWER = 2

# Initialize MC_NEST with a problem, number of rollouts, and selection policy
problem = "Let $S$ be a list of positive integers not necessarily distinct in which the number $68$ appears. The average (arithmetic mean) of the numbers in $S$ is $56$. However, if $68$ is removed, the average of the remaining numbers drops to $55$. What is the largest number that can appear in $S$?"
MC_NEST = MC_NEST_gpt4o(problem=problem, max_rollouts=4, selection_policy = IMPORTANCE_SAMPLING, initialize_strategy = ZERO_SHOT)

# Run MCTS to get the best answer
best_answer = MC_NEST.run()
print(best_answer)
```
## Customize Selection Policies

Choose from multiple selection policies:

```bash
GREEDY = 1
IMPORTANCE_SAMPLING = 2
PAIRWISE_IMPORTANCE_SAMPLING = 3
```

## References

[1] Zhang, D., Li, J., Huang, X., Zhou, D., Li, Y., & Ouyang, W. (2024). Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B. arXiv preprint arXiv:2406.07394. 

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for major changes.

## License

## 📄 Citation

```bash

@article{YourCitationKey,
  author    = {Your Name and Co-author(s)},
  title     = {Title of Your Paper},
  journal   = {Journal Name or arXiv},
  year      = {Year},
  volume    = {Volume (if applicable)},
  number    = {Number (if applicable)},
  pages     = {Pages (if applicable)},
  doi       = {DOI or URL},
}

```

This project is licensed under the MIT License.

Happy coding! 🚀
