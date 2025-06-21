from concurrent.futures import ThreadPoolExecutor, TimeoutError
import time

from solver_interface import SolverInterface
from problem_instance import ProblemInstance
from dataset import CityDataset

class Benchmark:
  def timed_solve(self, solver: SolverInterface, problem_instance: ProblemInstance, timeout_secs: int):
    with ThreadPoolExecutor() as executor:
      future = executor.submit(solver.solve, problem_instance)
      try:
        return future.result(timeout=timeout_secs)
      except TimeoutError as e:
        raise TimeoutError(f'Solver unable to solve problem in {timeout_secs} secs, problem: {problem_instance}') from e
      
if __name__ == '__main__':
  class MockSolver(SolverInterface):
    def solve(self, problem_instance: ProblemInstance):
      time.sleep(0.5)
      return problem_instance.distance_matrix
  
  benchmark = Benchmark()
  benchmark.timed_solve(MockSolver(), CityDataset().generate_problem_instance(3, 1, 3, seed=5), 1)

  
    