from concurrent.futures import ThreadPoolExecutor, TimeoutError

from solver_interface import SolverInterface
from problem_instance import ProblemInstance

class Benchmark:
  def timed_solve(self, solver: SolverInterface, problem_instance: ProblemInstance, timeout_secs: int):
    with ThreadPoolExecutor() as executor:
      future = executor.submit(solver.solve(problem_instance))
      try:
        return future.result(timeout=timeout_secs)
      except TimeoutError:
        raise TimeoutError(f'Solver unable to solve problem in {timeout_secs} secs, problem: {problem_instance}')
      

  
    