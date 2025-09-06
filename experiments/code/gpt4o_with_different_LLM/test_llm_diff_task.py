from __future__ import annotations

import pandas as pd 
import random
import math
from collections import deque
import tqdm
import numpy as np
from typing import ClassVar
from openai.types.chat import ChatCompletionMessageParam, ChatCompletion
from pydantic import BaseModel, Field
from openai import OpenAI


def openai_chat_completion(messages: list[ChatCompletionMessageParam], model: str, temperature: float, **kwargs) -> ChatCompletion:
    client = OpenAI(api_key='')
    response = client.chat.completions.create(model=model, messages=messages, temperature=temperature, **kwargs)
    return response


class RefineResponse(BaseModel):
    thought: str = Field(..., description="The thought process behind the answer.")
    answer: float = Field(..., description="The answer to the problem.")


def gpt_prompt_config():
    critic_system_prompt = "Provide a detailed and constructive critique to improve the answer. Highlight specific areas that need refinement or correction."
    refine_system_prompt = """# Instruction
                            Refine the answer based on the critique. Your refined answer should be a direct and concise solution to the problem.

                            ## Additional guidelines
                            - Your response should not refer to or discuss the criticisms.
                            - Do not repeat the problem statement.
                            - Respond with a detailed solution and answer.

                            # JSON Response format
                            {
                                "thought": "The thought process behind the answer.",
                                "answer": "A float representing the answer to the problem."
                            }
                            """
    evaluate_system_prompt = "Provide a reward score between -100 and 100 for the answer quality, using very strict standards. " \
                             "Do not give a full score above 95. Make sure the reward score is an integer. " \
                             "Return *ONLY* the score."
    return critic_system_prompt, refine_system_prompt, evaluate_system_prompt


class Node(BaseModel):
    answer: str
    parent: Node | None = None
    children: list[Node] = []
    visit: int = 0
    Q: float = 0.0
    reward_samples: list[int] = []

    def __hash__(self):
        return hash(self.answer)

    def __eq__(self, other):
        return isinstance(other, Node) and self.answer == other.answer

    def add_child(self, child_node: Node):
        self.children.append(child_node)

    def add_reward(self, reward: int):
        self.reward_samples.append(reward)
        avg_reward = np.mean(self.reward_samples)
        min_reward = np.min(self.reward_samples)
        self.Q = (min_reward + avg_reward) / 2

    def __repr__(self):
        return f"Node(answer='{self.answer}', Q={self.Q}, visit={self.visit})"


GREEDY = 1
IMPORTANCE_SAMPLING = 2
PAIRWISE_IMPORTANCE_SAMPLING = 3
ZERO_SHOT = 1
DUMMY_ANSWER = 2


class MCTSr(BaseModel):
    problem: str
    max_rollouts: int
    exploration_constant: float = 1.0
    max_children: int = 2
    epsilon: float = 1e-10
    reward_limit: int = 95
    excess_reward_penalty: int = 5
    selection_policy: int = PAIRWISE_IMPORTANCE_SAMPLING
    initialize_strategy: int = ZERO_SHOT

    root: Node = Node(answer="I don't know.")

    def self_refine(self, node: Node) -> Node:
        raise NotImplementedError()

    def _evaluate_answer(self, node: Node) -> int:
        raise NotImplementedError()

    def self_evaluate(self, node: Node):
        reward = self._evaluate_answer(node)
        if reward > self.reward_limit:
            reward = reward - self.excess_reward_penalty
        node.add_reward(reward=reward)
        node.visit += 1

    def backpropagate(self, node: Node):
        parent = node.parent
        while parent:
            best_child_Q = max(child.Q for child in parent.children)
            parent.Q = (parent.Q + best_child_Q) / 2
            parent.visit += 1
            parent = parent.parent

    def uct(self, node: Node):
        if not node.parent:
            return 10_000
        return node.Q + self.exploration_constant * math.sqrt(math.log(node.parent.visit + 1) / (node.visit + self.epsilon))

    def is_fully_expanded(self, node: Node):
        return len(node.children) >= self.max_children or any(child.Q > node.Q for child in node.children)

    def calculate_nash_equilibrium_strategy(self) -> dict[Node, float]:
        nash_equilibrium_strategy = {}
        for node in self.root.children:
            nash_equilibrium_strategy[node] = 1 / len(self.root.children)
        return nash_equilibrium_strategy

    def select_node(self):
        candidates: list[Node] = []
        to_consider = deque([self.root])
        while to_consider:
            current_node = to_consider.popleft()
            if not self.is_fully_expanded(current_node):
                candidates.append(current_node)
            to_consider.extend(current_node.children)
        if not candidates:
            return self.root

        nash_equilibrium_strategy = self.calculate_nash_equilibrium_strategy()

        if self.selection_policy == GREEDY:
            candidate_scores = [(self.uct(node), nash_equilibrium_strategy.get(node, 0)) for node in candidates]
            chosen_node = max(candidates, key=lambda node: candidate_scores[candidates.index(node)][0] + candidate_scores[candidates.index(node)][1])
            return chosen_node

        elif self.selection_policy == IMPORTANCE_SAMPLING:
            uct_scores = [self.uct(node) for node in candidates]
            weights = [uct_scores[i] * nash_equilibrium_strategy.get(node, 0) for i, node in enumerate(candidates)]
            if sum(weights) <= 0:
                return random.choice(candidates)
            return random.choices(candidates, weights=weights, k=1)[0]

        elif self.selection_policy == PAIRWISE_IMPORTANCE_SAMPLING:
            uct_scores = [self.uct(node) for node in candidates]
            pairs = [(i, j) for i in range(len(candidates)) for j in range(i + 1, len(candidates))]
            pair_weights = [abs(uct_scores[i] - uct_scores[j]) * nash_equilibrium_strategy.get(candidates[i], 0) * nash_equilibrium_strategy.get(candidates[j], 0) for i, j in pairs]
            if sum(pair_weights) <= 0:
                return random.choice(candidates)
            selected_pair_idx = random.choices(range(len(pairs)), weights=pair_weights, k=1)[0]
            selected_candidate_idx = max(pairs[selected_pair_idx], key=lambda x: uct_scores[x])
            return candidates[selected_candidate_idx]
        else:
            raise ValueError(f"Invalid selection policy: {self.selection_policy}")

    def zero_shot(self) -> str:
        raise NotImplementedError()

    def initialize(self):
        if self.initialize_strategy == ZERO_SHOT:
            self.root = Node(answer=self.zero_shot())
        elif self.initialize_strategy == DUMMY_ANSWER:
            self.root = Node(answer="I don't know.")
        else:
            raise ValueError(f"Invalid initialize strategy: {self.initialize_strategy}")

    def run(self):
        self.initialize()
        for _ in tqdm.tqdm(range(self.max_rollouts)):
            node = self.select_node()
            self.self_evaluate(node)
            child = self.self_refine(node)
            node.add_child(child)
            self.self_evaluate(child)
            self.backpropagate(child)
        return self.get_best_answer()

    def get_best_answer(self):
        to_visit = deque([self.root])
        best_node = self.root
        while to_visit:
            current_node = to_visit.popleft()
            if current_node.Q > best_node.Q:
                best_node = current_node
            to_visit.extend(current_node.children)
        return best_node.answer


class MCTSrDualLLM(MCTSr):
    # GPT-5 for generation, GPT-4 for critique/refinement prompts
    model_gen: ClassVar[str] = "gpt-5"
    model_critic: ClassVar[str] = "gpt-4o"
    model_eval: ClassVar[str] = "gpt-5"
    critic_system_prompt: ClassVar[str]
    refine_system_prompt: ClassVar[str]
    evaluate_system_prompt: ClassVar[str]

    critic_system_prompt, refine_system_prompt, evaluate_system_prompt = gpt_prompt_config()

    def zero_shot(self) -> str:
        response = openai_chat_completion(
            messages=[
                {"role": "system", "content": "Solve the problem. Return only the output. Step by step."},
                {"role": "user", "content": f"<problem>\n{self.problem}\n</problem>"},
            ],
            model=self.model_gen,
            temperature=1.0,
            max_completion_tokens=4000,
        )
        return response.choices[0].message.content

    def self_refine(self, node: Node) -> Node:
        # GPT-4 critique
        critique_response = openai_chat_completion(
            messages=[
                {"role": "system", "content": self.critic_system_prompt},
                {"role": "user", "content": f"<problem>\n{self.problem}\n</problem>\n<current_answer>\n{node.answer}\n</current_answer>"},
            ],
            model=self.model_critic,
            temperature=1.0,
            max_completion_tokens=4000,
        )
        critique = critique_response.choices[0].message.content

        # GPT-5 refinement
        refined_response = openai_chat_completion(
            messages=[
                {"role": "system", "content": self.refine_system_prompt},
                {"role": "user", "content": f"<problem>\n{self.problem}\n</problem>\n<current_answer>\n{node.answer}\n</current_answer>\n<critique>\n{critique}\n</critique>"},
            ],
            model=self.model_gen,
            temperature=1.0,
            max_completion_tokens=4000,
            response_format={"type": "json_object"},
        )
        refined_answer = RefineResponse.model_validate_json(refined_response.choices[0].message.content)
        return Node(answer=f"{refined_answer.thought}\n\n# Answer\n{refined_answer.answer}", parent=node)

    def _evaluate_answer(self, node: Node) -> int:
        messages = [
            {"role": "system", "content": self.evaluate_system_prompt},
            {"role": "user", "content": f"<problem>\n{self.problem}\n</problem>\n<answer>\n{node.answer}\n</answer>"},
        ]
        for attempt in range(3):
            try:
                response = openai_chat_completion(
                    messages=messages,
                    model=self.model_eval,
                    temperature=1.0,
                    max_completion_tokens=4000,
                )
                return int(response.choices[0].message.content)
            except ValueError:
                messages.append({"role": "user", "content": "Failed to parse reward as integer."})
                if attempt == 2:
                    raise



def print_tree(node: Node | None, level: int = 0):
    if node is None:
        return
    indent = " " * level * 2
    node_str = repr(node)
    for line in node_str.split("\n"):
        print(indent + line)
    for child in node.children:
        print_tree(child, level + 1)


if __name__ == "__main__":
    df = pd.read_csv("/nfs/home/rabbyg/CAG/AIME_dataset_exp/dataset/AIME_2025/aime2025.csv")
    df = df[df['answer'].notna()]
    
    for rollout in range(4, 37, 4):
        df[f'rollout_{rollout}'] = np.nan

    def run_mcts_and_save(df, index, rollouts):
        for max_rollout in rollouts:
            try:
                mcts = MCTSrDualLLM(problem=df.at[index, 'question'], max_rollouts=max_rollout)
                best_answer = mcts.run()
                df.at[index, f'rollout_{max_rollout}'] = str(best_answer.split('# Answer'))
            except Exception as e:
                print(f"Error processing index {index} with {max_rollout} rollouts: {e}")
                df.at[index, f'rollout_{max_rollout}'] = np.nan

    rollouts = [4]  # You can extend as needed

    for index, row in df.iterrows():
        print(index)
        print(row['question'])
        run_mcts_and_save(df, index, rollouts)

        df.to_csv("/nfs/home/rabbyg/CAG/AIME_dataset_exp/output/AIME_2025/diff_llm/4_PAIRWISE_IMPORTANCE_SAMPLING_rollout_mctsr_NE_AIME_2025_gpt-5-thinking.csv", encoding='utf-8', index=False)
