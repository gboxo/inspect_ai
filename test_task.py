from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match
from inspect_ai.solver import generate

@task
def simple_test():
    return Task(
        dataset=[Sample(input="What is 2+2?", target="4")],
        solver=generate(),
        scorer=match(),
    ) 