from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import match
from inspect_ai.solver import generate, prompt_template, system_message
from datasets import load_dataset

import asyncio
from typing import Any, Dict, List
from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, GenerateConfig, Model, ModelAPI, ModelOutput, ModelUsage
from inspect_ai.tool import Tool, ToolChoice
import openai
from inspect_ai.model import ChatMessageUser, ChatMessageSystem, ChatMessage
from inspect_ai import eval
from inspect_ai.model._registry import modelapi



from inspect_ai.model._providers.model_router import VLLMRouter

from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import match
from inspect_ai.solver import (
    Solver,
    generate,
    prompt_template,
)

USER_PROMPT_TEMPLATE = """
Solve the following math problem step by step.
The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.
""".strip()


@task
def aime2024() -> Task:
    """Inspect Task implementation for the AIME 2024 benchmark."""
    dataset = hf_dataset(
        path="Maxwell-Jia/AIME_2024",
        split="train",
        trust=True,
        sample_fields=record_to_sample,
    )

    return Task(
        dataset=dataset,
        solver=aime2024_solver(),
        scorer=[
            match(),
        ],
    )


def aime2024_solver() -> list[Solver]:
    """Build solver for AIME 2024 task."""
    solver = [prompt_template(USER_PROMPT_TEMPLATE), generate()]
    return solver


def record_to_sample(record: dict[str, Any]) -> Sample:
    sample = Sample(
        id=record["ID"],
        input=record["Problem"],
        target=str(record["Answer"]),
        metadata={
            "solution": record["Solution"],
        },
    )
    return sample

if __name__ == "__main__":
    task = aime2024()
    eval(task, model=Model(api=VLLMRouter(model_name="vllm_router/math_router",
                                          models=["Qwen/Qwen2.5-1.5B-Instruct", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"],
                                          ports=[33534, 33535],
                                          routing_config={"weights_path": "/home/ubuntu/inspect_ai/router_weights.pt"},
                                          server_args=[{"max_model_len": 4096, "device": "0"}, {"max_model_len": 4096, "device": "1"}],
                                          server_launch_delay=30),
                           config=GenerateConfig(max_tokens=1000)))
    # eval(task, model=Model(api=MartialRouterModel(model_name="organizations/386aa70e-f2d9-44e6-a067-e04ff02cd125/routers/reasoning-router"), config=GenerateConfig(max_tokens=1000)))
    
    