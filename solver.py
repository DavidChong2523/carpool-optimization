import scipy
import numpy as np

from dataset import CityDataset
from problem_instance import ProblemInstance

class State: 
  def __init__(self, problem_instance: ProblemInstance):
    self.node_to_capacity = {}
    self.node_to_num_people = {}
    self.destination_node = problem_instance.destination_index
    self.distance_matrix = problem_instance.distance_matrix
    self.problem_instance = problem_instance
    # for conveniently iterating across non-destination nodes
    self.starting_nodes = [i for i in range(problem_instance.num_nodes) if i != problem_instance.destination_index]
    self.solution = np.zeros(problem_instance.distance_matrix.shape)

    for i, c in enumerate(problem_instance.car_capacities):
      node = i if i - problem_instance.destination_index < 0 else i + 1
      self.node_to_capacity[node] = c
      self.node_to_num_people[node] = 1
    self.node_to_num_people[self.destination_node] = 0
    self.node_to_capacity[self.destination_node] = float('inf')

  @property
  def is_final(self):
    """True if no valid moves exist, False otherwise"""
    non_destination_capacity = [self.node_to_capacity[n] for n in self.starting_nodes]
    return sum(non_destination_capacity) == 0
      
  def move(self, start_node: int, end_node: int):
    """Move the people from the start node to the end node"""
    start_capacity, start_num_people = self.node_to_capacity[start_node], self.node_to_num_people[start_node] 
    end_capacity, end_num_people = self.node_to_capacity[end_node], self.node_to_num_people[end_node]
    if start_capacity < start_num_people:
      raise RuntimeError(f'start node capacity {start_capacity} is < start node num people {start_num_people}')
    
    final_capacity = max(start_capacity, end_capacity)
    final_num_people = start_num_people + end_num_people
    if final_capacity < final_num_people:
      raise RuntimeError(f'max capacity {final_capacity} is < total num people {final_num_people} between start and end nodes')

    self.node_to_capacity[end_node] = final_capacity
    self.node_to_num_people[end_node] = final_num_people
    self.node_to_capacity[start_node] = 0
    self.node_to_num_people[start_node] = 0

    self.solution[start_node][end_node] = 1

      


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

  def pick_up_carless_nodes_greedily(self, state: State) -> State: 
    """Each car-less node is picked up by the closest node with sufficient capacity"""
    no_car_nodes = []
    for node, num_people in state.node_to_num_people.items():
      if num_people == 0:
        continue 
      if state.node_to_capacity[node] == 0:
        no_car_nodes.append(node)
        
    for n in no_car_nodes:
      min_distance = float('inf')
      closest_valid_node = -1
      for node in state.starting_nodes:
        if state.node_to_capacity[node] - state.node_to_num_people[node] < 1:
          continue
        distance = state.distance_matrix[node][n]
        if distance < min_distance:
          min_distance = distance
          closest_valid_node = node
      if closest_valid_node < 0:
        raise RuntimeError(f'no valid node found')
      state.move(closest_valid_node, n)
    return state


  def naive_solution(self, problem_instance: ProblemInstance):
    if sum(problem_instance.car_capacities) < problem_instance.num_nodes - 1:
      # problem unsolvable because not enough car capacity
      return np.zeros(problem_instance.distance_matrix.shape)

    state = State(problem_instance)
    state = self.pick_up_carless_nodes_greedily(state)

    # now all nodes with car capacity > 0 go directly to destination
    for node in state.starting_nodes:
      if state.node_to_capacity[node] == 0:
        continue
      state.move(node, problem_instance.destination_index)
    return state.solution
  

  def compute_time_save(self, curr_node: int, next_node: int, destination_ind: int, distance_matrix: np.array) -> float:
    direct_travel_cost = distance_matrix[curr_node][destination_ind] + distance_matrix[next_node][destination_ind]
    carpool_cost = distance_matrix[curr_node][next_node] + distance_matrix[next_node][destination_ind]
    return direct_travel_cost - carpool_cost


  def greedy_solution(self, problem_instance: ProblemInstance):
    if sum(problem_instance.car_capacities) < problem_instance.num_nodes - 1:
      # problem unsolvable because not enough car capacity
      return np.zeros(problem_instance.distance_matrix.shape)
    
    state = State(problem_instance) 
    state = self.pick_up_carless_nodes_greedily(state)

    # then go highest to lowest car capacity, for each car pick up the optimal number of 
    # friends assuming every other car will go directly to the destination
    while not state.is_final:
      # find the node with max capacity
      curr_node = state.starting_nodes[0]
      for n in state.starting_nodes:
        if state.node_to_capacity[n] > state.node_to_capacity[curr_node]:
          curr_node = n 
          
      # move the car to the destination
      while curr_node != problem_instance.destination_index:
        target_node = problem_instance.destination_index
        best_time_save = 0
        for next_node in state.starting_nodes:
          if next_node == curr_node or state.node_to_num_people[next_node] == 0:
            continue 
          future_capacity = max(state.node_to_capacity[curr_node], state.node_to_capacity[next_node])
          future_num_people = state.node_to_num_people[curr_node] + state.node_to_num_people[next_node]
          if future_capacity < future_num_people:
            continue 
          time_save = self.compute_time_save(curr_node, next_node, problem_instance.destination_index, problem_instance.distance_matrix)
          if time_save < best_time_save:
            continue 
          target_node = next_node 
          best_time_save = time_save
        state.move(curr_node, target_node)
        curr_node = target_node
        
    return state.solution


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

