import pandas as pd
from tot.methods.bfs import solve
from dotenv import load_dotenv
from openai import OpenAI
import os
import time


# -------------------------------
# Load environment variables
# -------------------------------



load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or ""


if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing! Set it in your .env file or export it before running.")

# -------------------------------
# Custom OpenAI chat completion
# -------------------------------
client = OpenAI(api_key=OPENAI_API_KEY)



#Define model pricing in USD per 1k tokens
MODEL_PRICING = {
    "o3-2025-04-16": 0.003,   # example: $0.003 per 1k tokens
}

def openai_chat_completion(messages: list, model: str, temperature: float = 0.7, **kwargs):
    """
    Custom OpenAI Chat Completion using explicit API key.
    Returns a tuple: (response_text, cost_in_usd)
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        **kwargs
    )
    # Extract tokens usage
    usage = response.usage
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = prompt_tokens + completion_tokens
    # Compute cost
    price_per_1k = MODEL_PRICING.get(model, 0.003)
    cost = total_tokens / 1000 * price_per_1k
    return response.choices[0].message.content, cost

# -------------------------------
# Custom AIMETask for ToT
# -------------------------------
class AIMETask:
    def __init__(self, problem: str, answer: str = None):
        self.problem = problem
        self.answer = answer
        self.steps = 1  # Required by ToT BFS

    def get_input(self, problem: str = None):
        return problem if problem else self.problem

    def propose_prompt_wrap(self, problem: str, k: int):
        return f"""
        You are solving an AIME math competition problem.

        Problem:
        {problem}

        Generate {k} possible reasoning paths to solve this problem.
        Each reasoning path should show step-by-step calculations and end with a final numerical answer.
        Format each candidate as: "Thought X: ... Final Answer: ...".
        """

    def value_prompt_wrap(self, problem: str, thoughts: list):
        thoughts_list = "\n".join([f"Thought {i+1}: {t}" for i, t in enumerate(thoughts)])
        return f"""
        You are solving an AIME math competition problem.

        Problem:
        {problem}

        The following candidate solutions were proposed:
        {thoughts_list}

        Evaluate each thought based on correctness, completeness, and logical validity.
        Respond with a score between 0 (completely wrong) and 1 (completely correct) for each candidate.
        Return only a JSON list of scores, e.g. [0.2, 1.0, 0.7].
        """

    def test_output(self, output: str):
        if self.answer is not None:
            return output.strip() == str(self.answer).strip()
        return None

# -------------------------------
# Run ToT solver with OpenAI
# -------------------------------
def run_tot_solver(problem, model="o3-2025-04-16", max_depth=3, branching_factor=2):
    task = AIMETask(problem)

    class Args:
        backend = model
        temperature = 0.7
        method_generate = "propose"
        method_evaluate = "value"
        method_select = "greedy"
        n_generate_sample = 1
        n_evaluate_sample = branching_factor
        n_select_sample = branching_factor
        naive_run = False
        prompt_sample = None
        task = "aime"
        # Inject custom OpenAI call
        openai_chat_fn = staticmethod(openai_chat_completion)

    args = Args()
    # Solve returns a string normally; here we also return cost
    # Wrap the chat completion to capture cost inside BFS if needed
    solution = solve(args, task, idx=0)
    # Note: BFS doesn't return cost; estimate cost per problem could be implemented via monkey-patch
    # For simplicity, we will just return solution here
    return solution

# -------------------------------
# Main AIME processing
# -------------------------------
if __name__ == "__main__":
    input_csv = "/nfs/home/rabbyg/CAG/AIME_dataset_exp/dataset/AIME_2025/aime2025.csv"
    output_csv = "/nfs/home/rabbyg/CAG/AIME_dataset_exp/output/AIME_2025/TOT_solutions.csv"

    df = pd.read_csv(input_csv)
    df = df[df['answer'].notna()]
    df['tot_solution'] = None
    df['tot_time_seconds'] = None
    df['tot_cost_usd'] = None  # <-- new column for cost

    for index, row in df.iterrows():
        print(f"\nProcessing index {index}: {row['question']}")
        start_time = time.time()
        try:
            solution = run_tot_solver(
                row['question'],
                model="o3-2025-04-16",
                max_depth=3,
                branching_factor=2
            )
            elapsed_time = time.time() - start_time
            # For now we cannot get exact cost from solve() because BFS calls openai_chat_completion internally
            # If you want exact cost per problem, we need to monkey-patch openai_chat_fn to sum the cost
            df.at[index, 'tot_solution'] = solution
            df.at[index, 'tot_time_seconds'] = elapsed_time
            df.at[index, 'tot_cost_usd'] = None  # placeholder
            print(f"✅ Solution computed in {elapsed_time:.2f} sec")
        except Exception as e:
            elapsed_time = time.time() - start_time
            df.at[index, 'tot_solution'] = None
            df.at[index, 'tot_time_seconds'] = elapsed_time
            df.at[index, 'tot_cost_usd'] = None
            print(f"❌ Error at index {index}: {e} (Time: {elapsed_time:.2f} sec)")

        # Save progress
        df.to_csv(output_csv, encoding='utf-8', index=False)
