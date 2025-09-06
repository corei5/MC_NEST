from __future__ import annotations

import openai
import random
import math
from collections import deque
from openai.types.chat import ChatCompletionMessageParam, ChatCompletion
import os
import openai
from pydantic import BaseModel, Field
import tqdm
import numpy as np
from typing import ClassVar
from openai import OpenAI
import pandas as pd
import json
import torch
import gc

import transformers
print(f"Transformers Version: {transformers.__version__}")

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoConfig,
    StoppingCriteria,
    set_seed
)
set_seed(42)

from LLM.llm import load_llm_Phi_3_mini
tokenizer, model, pipeline, MODEL_PATH = load_llm_Phi_3_mini()
model.dtype, model.hf_device_map

print(MODEL_PATH)

def LLM_chat_completion(messages: list[ChatCompletionMessageParam], model: str, temperature: float, **kwargs) -> str:
    response = pipeline(messages)
    #print(f"Pipeline response: {response}")
    
    # Ensure we are extracting the correct text from the response
    if isinstance(response, list):
        if len(response) > 0 and isinstance(response[0], dict) and "generated_text" in response[0]:
            return response[0]["generated_text"]
        else:
            raise ValueError(f"Unexpected list response format: {response}")
    elif isinstance(response, str):
        return response
    else:
        raise ValueError(f"Unexpected response format: {response}")

class RefineResponse(BaseModel):
    thought: str = Field(..., description="The thought process behind the answer.")
    answer: float = Field(..., description="The answer to the problem.")

def LLM_prompt_config(model: str):
    critic_system_prompt = "Provide a detailed and constructive critique to improve the answer. Highlight specific areas that need refinement or correction."
    refine_system_prompt = """# Instruction
                Refine the answer based on the critique. Your refined answer should be a direct and concise solution to the problem.

                ## Additional guidelines
                - Your response should not refer to or discuss the criticisms.
                - Do not repeat the problem statement.
                - Respond with a detailed solution and answer.
                """
    evaluate_system_prompt = "Provide a reward score between -100 and 100 for the answer quality, using very strict standards. Do not give a full score above 95. Make sure the reward score is an integer. Return *ONLY* the score."

    return model, critic_system_prompt, refine_system_prompt, evaluate_system_prompt

class Node(BaseModel):
    answer: str
    parent: Node | None = None
    children: list[Node] = []
    visit: int = 0
    Q: float = 0.0
    reward_samples: list[int] = []

    def add_child(self, child_node: Node):
        self.children.append(child_node)

    def add_reward(self, reward: int):
        self.reward_samples.append(reward)
        avg_reward = np.mean(self.reward_samples)
        min_reward = np.min(self.reward_samples)
        self.Q = (min_reward + avg_reward) / 2

    def __hash__(self):
        # You can base the hash on the answer attribute, for example
        return hash(self.answer)

    def __eq__(self, other):
        # Define equality based on the answer attribute
        if isinstance(other, Node):
            return self.answer == other.answer
        return False

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
    selection_policy: int = IMPORTANCE_SAMPLING #PAIRWISE_IMPORTANCE_SAMPLING #GREEDY #IMPORTANCE_SAMPLING 
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
        # Example: Uniform Nash Equilibrium strategy
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

        # Retrieve Nash Equilibrium strategy
        nash_equilibrium_strategy = self.calculate_nash_equilibrium_strategy()

        if self.selection_policy == GREEDY:

            print("................GREEDY............")            

            candidate_scores = [(self.uct(node), nash_equilibrium_strategy.get(node, 0)) for node in candidates]
            chosen_node = max(candidates, key=lambda node: candidate_scores[candidates.index(node)][0] + candidate_scores[candidates.index(node)][1])
            return chosen_node

        elif self.selection_policy == IMPORTANCE_SAMPLING:
            
            print("................IMPORTANCE_SAMPLING............")

            uct_scores = [self.uct(node) for node in candidates]

            #print(uct_scores)

            weights = [uct_scores[i] * nash_equilibrium_strategy.get(node, 0) for i, node in enumerate(candidates)]

            #print(weights)

            if sum(weights) <= 0:  # Handle case where all weights are zero or negative
                return random.choice(candidates)
            selected_node = random.choices(candidates, weights=weights, k=1)[0]
            return selected_node

        elif self.selection_policy == PAIRWISE_IMPORTANCE_SAMPLING:

            print("...............PAIRWISE_IMPORTANCE_SAMPLING...........")

            uct_scores = [self.uct(node) for node in candidates]
            pairs = [(i, j) for i in range(len(candidates)) for j in range(i + 1, len(candidates))]
            pair_weights = [abs(uct_scores[i] - uct_scores[j]) * nash_equilibrium_strategy.get(candidates[i], 0) * nash_equilibrium_strategy.get(candidates[j], 0) for i, j in pairs]
            if sum(pair_weights) <= 0:  # Handle case where all pair weights are zero or negative
                return random.choice(candidates)
            selected_pair_idx = random.choices(range(len(pairs)), weights=pair_weights, k=1)[0]
            selected_candidate_idx = max(pairs[selected_pair_idx], key=lambda x: uct_scores[x])
            return candidates[selected_candidate_idx]
        else:
            raise ValueError(f"Invalid selection policy: {self.selection_policy}")

    def zero_shot(self) -> str:
        response = LLM_chat_completion(
            messages=[{"role": "user", "content": f"The user will provide a problem. Solve the problem. Think step by step.\n<problem>\n{self.problem}\n</problem>"}],
            model=self.model,
            temperature=0.9,
            max_tokens=4000,
        )
        #print(f"Zero-shot response: {response}")
        assert response is not None
        return response

    def initialize(self):
        if self.initialize_strategy == ZERO_SHOT:
            response = self.zero_shot()

            #print(f"Zero-shot response: {response}")

            self.root = Node(answer= str(response))

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

class MCTSrLLM(MCTSr):
    model: ClassVar[str]
    critic_system_prompt: ClassVar[str]
    refine_system_prompt: ClassVar[str]
    evaluate_system_prompt: ClassVar[str]

    model, critic_system_prompt, refine_system_prompt, evaluate_system_prompt = LLM_prompt_config(model)

    def zero_shot(self) -> str:
        response = LLM_chat_completion(
            messages=[{"role": "user", "content": f"The user will provide a problem. Solve the problem. Think step by step.\n<problem>\n{self.problem}\n</problem>"}],
            model=self.model,
            temperature=0.9,
            max_tokens=4000,
        )
        #print(f"Zero-shot response: {response}")
        assert response is not None
        return response

    def self_refine(self, node: Node) -> Node:
        critique_response = LLM_chat_completion(
            messages=[
                {"role": "system", "content": self.critic_system_prompt},
                {"role": "user", "content": f"<problem>\n{self.problem}\n</problem>\n<current_answer>\n{node.answer}\n</current_answer>"},
            ],
            model=self.model,
            temperature=0.9,
            max_tokens=4000,
        )
        #print(f"Critique response: {critique_response}")
        
        refined_answer = LLM_chat_completion(
            # messages=[
            #     {"role": "system", "content": self.refine_system_prompt},
            #     {"role": "user", "content": f"<problem>\n{self.problem}\n</problem>\n<critique_of_current_answer>\n{critique_response}\n</critique_of_current_answer>"},
            # ],

            messages=str(self.refine_system_prompt)+f" <problem>\n{self.problem}\n</problem>\n<critique_of_current_answer>\n{critique_response}\n</critique_of_current_answer>",
            model=self.model,
            temperature=0.9,
            max_tokens=4000,
        )
        #print(f"Refined answer: {refined_answer}")

        assert refined_answer is not None
        return Node(answer=refined_answer, parent=node)

    def _evaluate_answer(self, node: Node) -> int:
        evaluation_response = LLM_chat_completion(
            # messages=[
            #     {"role": "system", "content": self.evaluate_system_prompt},
            #     {"role": "user", "content": f"<problem>\n{self.problem}\n</problem>\n<current_answer>\n{node.answer}\n</current_answer>"},
            # ],
            messages=str(self.evaluate_system_prompt)+f"<problem>\n{self.problem}\n</problem>\n<current_answer>\n{node.answer}\n</current_answer>",
            model=self.model,
            temperature=0.9,
            max_tokens=4000,
        )

        # import re

        # print(f"Evaluation response: {evaluation_response[0]}")
        # pattern = r"'content': '([^']*)'"

        # # Find all matches in the string
        # matches = re.findall(pattern, evaluation_response[0])

        # # Print the extracted content values
        # for match in matches:
        #     print(match)

        try:
            evaluation = int(evaluation_response)
            return evaluation
        except ValueError:
            print(f"Error in _evaluate_answer: {evaluation_response}")
            return -100



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
    df = pd.read_csv("/nfs/home/rabbyg/CAG/AIME_dataset_exp/dataset/AIME_Problems_1983_to_2024_with_columns - AIME_Problems_1983_to_2024_with_columns.csv")
    df = df[df['Exact_Answer'].notna()]
    
    df['rollout_4'] = np.nan
    df['rollout_8'] = np.nan
    df['rollout_12'] = np.nan
    df['rollout_16'] = np.nan
    df['rollout_20'] = np.nan
    df['rollout_24'] = np.nan
    df['rollout_28'] = np.nan
    df['rollout_32'] = np.nan
    df['rollout_36'] = np.nan

    def run_mcts_and_save(df, index, rollouts):
        for max_rollout in rollouts:
            try:
                mcts = MCTSrLLM(problem=df.at[index, 'Problem_Statement'], max_rollouts=max_rollout)
                best_answer = mcts.run()
                df.at[index, f'rollout_{max_rollout}'] = str(best_answer.split('# Answer'))
            except Exception as e:
                print(f"Error processing index {index} with {max_rollout} rollouts: {e}")
                df.at[index, f'rollout_{max_rollout}'] = np.nan

            # Clear GPU memory after each task and perform garbage collection
            gc.collect()
            torch.cuda.empty_cache()  # Explicitly clear the cache after each task

    rollouts = [4, 8, 12, 16, 20, 24, 28, 32, 36]

    # Start processing from index 15
    for index, row in df.iloc[115:].iterrows():  # Slice the DataFrame to start at index 15
        if index > 150:  # Stop processing beyond index 200
            break

        print(index)
        print(row['Problem_Statement'])

        run_mcts_and_save(df, index, rollouts)
        torch.cuda.empty_cache()
        gc.collect()

        df.to_csv("/nfs/home/rabbyg/CAG/AIME_dataset_exp/output/NE_output/Mistral_22B/IMPORTANCE_SAMPLING_rollout_mctsr_NE_Mistral_22B_AIME_from_115.csv", encoding='utf-8', index=False)

