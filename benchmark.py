from concurrent.futures import ThreadPoolExecutor, TimeoutError
import time
import numpy as np

from solver_interface import SolverInterface
from solver import Solver
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
      
  def generate_benchmark_dataset(self, city_dataset: CityDataset, seed: int) -> list[ProblemInstance]:
      """
      Generate bechmark dataset
      Use a fixed num_friends for all samples in the dataset
      Fix min_car_capacity at 0, max_car_capacity increments from 1 to num_friends as we generate new samples
      All samples will be valid problems
      """
      NUM_FRIENDS = 20
      MIN_CAR_CAPACITY = 0
      np.random.seed(seed)

      # increment max_car_capacity from 1 to 20 and generate 10 problems at each increment
      problems = []
      for max_capacity in range(1, 21):
        for _ in range(10):
          problem_instance = city_dataset.generate_problem_instance(
            num_friends=NUM_FRIENDS,
            min_car_capacity=MIN_CAR_CAPACITY,
            max_car_capacity=max_capacity,
            force_valid=True
          )
          problems.append(problem_instance)
      return problems

if __name__ == '__main__':
  class MockSolver(SolverInterface):
    def solve(self, problem_instance: ProblemInstance):
      time.sleep(0.5)
      return problem_instance.distance_matrix
  
  solver = Solver()

  city_dataset = CityDataset()
  benchmark = Benchmark()
  benchmark_dataset = benchmark.generate_benchmark_dataset(city_dataset=city_dataset, seed=3)
  
  total_cost = 0
  lower_bound_cost = 0
  for problem in benchmark_dataset:
    assert(problem.is_valid) 
    naive_solution = solver.naive_solution(problem) 
    min_spanning_tree_solution = solver.min_spanning_tree_solution(problem)
    assert(problem.validate_solution(naive_solution))
    total_cost += problem.get_solution_cost(naive_solution) 
    lower_bound_cost += problem.get_solution_cost(min_spanning_tree_solution)

  print(f'Avg cost: {total_cost / len(benchmark_dataset)}, avg lower bound: {lower_bound_cost / len(benchmark_dataset)}')




  
    