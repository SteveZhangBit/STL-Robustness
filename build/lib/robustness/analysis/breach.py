from robustness.agents import Agent
from robustness.analysis import *
from robustness.envs import DeviatableEnv


class BreachSTL(TraceEvaluator):
    def __init__(self, formula: str):
        self.formula = formula

    def __str__(self) -> str:
        return self.formula
        