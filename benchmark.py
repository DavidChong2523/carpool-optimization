from concurrent.futures import ThreadPoolExecutor, TimeoutError
from collections.abc import Callable
import time
import numpy as np

from solver import Solver
from problem_instance import ProblemInstance
from dataset import CityDataset

class Benchmark:
  def timed_solve(self, solver: Callable[[ProblemInstance], np.array], problem_instance: ProblemInstance, timeout_secs: int):
    with ThreadPoolExecutor() as executor:
      future = executor.submit(solver, problem_instance)
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
  # test benchmark
  class MockSolver:
    def solve(self, problem_instance: ProblemInstance) -> np.array:
      time.sleep(0.5)
      return problem_instance.distance_matrix
  
  city_dataset = CityDataset()
  benchmark = Benchmark()
  
  mock_solver = MockSolver()
  solver = Solver()

  test_problem = city_dataset.generate_problem_instance(3, 0, 2, True)
  benchmark.timed_solve(mock_solver.solve, test_problem, 1)


  # run solvers on benchmark
  benchmark_dataset = benchmark.generate_benchmark_dataset(city_dataset=city_dataset, seed=3)
  
  naive_cost = 0
  greedy_cost = 0
  lower_bound_cost = 0
  for problem in benchmark_dataset:
    assert(problem.is_valid) 

    naive_solution = benchmark.timed_solve(solver.naive_solution, problem, 1) 
    min_spanning_tree_solution = benchmark.timed_solve(solver.min_spanning_tree_solution, problem, 1)
    greedy_solution = benchmark.timed_solve(solver.greedy_solution, problem, 1)
    assert(problem.validate_solution(naive_solution))
    assert(problem.validate_solution(greedy_solution))

    naive_cost += problem.get_solution_cost(naive_solution) 
    greedy_cost += problem.get_solution_cost(greedy_solution)
    lower_bound_cost += problem.get_solution_cost(min_spanning_tree_solution)

  num_samples = len(benchmark_dataset)
  print(f'avg naive cost: {naive_cost / num_samples}')
  print(f'avg greedy cost: {greedy_cost / num_samples}')
  print(f'avg lower bound: {lower_bound_cost / num_samples}')




  
    