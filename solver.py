import scipy
import numpy as np

from dataset import CityDataset
from problem_instance import ProblemInstance

class Solver:
  def min_spanning_tree_solution(self, problem_instance: ProblemInstance) -> np.array:
    mst = scipy.sparse.csgraph.minimum_spanning_tree(problem_instance.distance_matrix)
    undirected_solution = mst.toarray()
    undirected_solution[undirected_solution > 1] = 1

    # get directed solution by traversing the tree from root to leaves
    solution = np.zeros(undirected_solution.shape)
    nodes = [problem_instance.destination_index]
    while len(nodes) != 0:
      node = nodes.pop(0)
      solution[:, node] = np.logical_or(undirected_solution[:, node], undirected_solution[node, :]).astype(int)
      undirected_solution[node, :] = np.zeros(problem_instance.num_nodes)
      undirected_solution[:, node] = np.zeros(problem_instance.num_nodes)
      children_nodes = np.where(solution[:, node] > 0)[0].tolist()
      nodes.extend(children_nodes)
    return solution

  def naive_solution(self, problem_instance: ProblemInstance):
    solution = np.zeros(problem_instance.distance_matrix.shape)
    if sum(problem_instance.car_capacities) < problem_instance.num_nodes - 1:
      # problem unsolvable because not enough car capacity
      return solution

    node_to_car_capacity = {}
    for i in range(len(problem_instance.car_capacities)):
      node = i if i - problem_instance.destination_index < 0 else i + 1
      node_to_car_capacity[node] = problem_instance.car_capacities[i]
    no_car_nodes = [n for n in node_to_car_capacity if node_to_car_capacity[n] == 0]
    for n in no_car_nodes:
      # get picked up by the closest node with car capacity >= 2
      min_distance = float('inf')
      closest_valid_node = 0
      for car_node, car_capacity in node_to_car_capacity.items():
        if car_capacity < 2:
          continue
        distance = problem_instance.distance_matrix[car_node][n]
        if distance < min_distance:
          min_distance = distance
          closest_valid_node = car_node
      solution[closest_valid_node][n] = 1
      node_to_car_capacity[n] = node_to_car_capacity[closest_valid_node]-1
      node_to_car_capacity[closest_valid_node] = 0

    # now all nodes with car capacity > 0 go directly to destination
    for node, car_capacity in node_to_car_capacity.items():
      if car_capacity == 0:
        continue
      solution[node][problem_instance.destination_index] = 1
    return solution
  
  def greedy_solution(self, problem_instance: ProblemInstance):
    # order by car capacity
    # then go lowest to highest car capacity
    # 0 car capacity gets picked up by closest node
    # then iterate through nodes, for every node move to either the destination or to another node that saves the optimal amount of time
    pass

  def simulated_annealing(self, problem_instane: ProblemInstance):
    # swap two edges, swap n edges
    # add an edge
    pass

  def solve_instance(self, problem_instance: ProblemInstance):
    # return self.min_spanning_tree_solution(problem_instance)
    return self.naive_solution(problem_instance)

def test_solver():
  city_dataset = CityDataset()
  problem_instance = city_dataset.generate_problem_instance(num_friends=4, min_car_capacity=0, max_car_capacity=3, force_valid=True)

  solver = Solver()
  spanning_tree_solution = solver.min_spanning_tree_solution(problem_instance)
  naive_solution = solver.naive_solution(problem_instance)
  solution = solver.solve_instance(problem_instance)

  is_valid = problem_instance.validate_solution(solution)
  cost = problem_instance.get_solution_cost(solution)

  print(problem_instance)
  print()
  print(solution)
  print(f'is_valid: {is_valid}, cost: {cost}, min_cost: {problem_instance.get_solution_cost(spanning_tree_solution)}')
  
  assert(is_valid)

if __name__ == '__main__':
    test_solver()

