from abc import ABC, abstractmethod

from problem_instance import ProblemInstance

class SolverInterface(ABC):
    @abstractmethod
    def solve(self, problem_instance: ProblemInstance):
        """
        Return a solution for the given problem instance
        If the problem has no solution, return array of all zeros
        """
        raise NotImplementedError()