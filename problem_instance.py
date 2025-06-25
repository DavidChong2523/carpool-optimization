import numpy as np


class ProblemInstance:
  def __init__(self, distance_matrix: np.array, destination_index: int, car_capacities: np.array, node_labels: list[str] | None = None):
    """
    An instance of the carpooling optimization problem
    distance_matrix is an n x n matrix where distance_matrix[i][j] represents the distance between node i and node j
    destination_index is an index in [0, n-1] which is the index of the destination node, all other nodes are starting locations
    car_capacities is a list of n-1 integers, representing the car capacities of nodes 0...n-1, excluding the destination node
    node_labels is a list of n strings, where each string is a human readable label of the nodes from index 0 to n-1
    
    NOTE: 
    If there are i friends at location A and friend i+1 visits this location, they must pick up all i friends
    They cannot pick up j < i friends and let the remaining i - j friends carpool together to the destination
    Allowing this breaks the validation assumption that valid solutions will be a tree
    """
    self.distance_matrix = distance_matrix
    self.num_nodes = distance_matrix.shape[0]
    self.destination_index = destination_index
    self.car_capacities = car_capacities
    # if no node_labels, just use the index values
    self.node_labels = node_labels if node_labels else [i for i in range(self.num_nodes)]

  @property
  def is_valid(self) -> bool:
    """Whether or not there is enough car capacity to carry everyone to the destination"""
    return sum(self.car_capacities) >= (self.num_nodes-1)

  def get_solution_cost(self, solution_matrix: np.array):
    """
    Get the total car-hours the provided solution requires
    solution_matrix is an n x n binary matrix, where a 1 represents an edge from node i to node j and 0 represents no edge
    """
    return np.sum(np.multiply(solution_matrix, self.distance_matrix))

  def validate_solution(self, solution_matrix: np.array):
    """
    Validate the provided solution
    solution_matrix is an n x n binary matrix, where a 1 represents an edge from node i to node j and 0 represents no edge

    A valid solution is a spanning tree such that for every subtree (except the root tree)
    the number of nodes in the subtree <= the max capacity of any node in the subtree

    We validate the input by doing bfs from the destination node and:
    1. validate no cycles are encountered
    2. validate that all nodes are reached
    3. determine the parent-child relationships of all nodes

    Then we do a second pass starting at the leaves and iterating up through parents.
    For each node we track:
    1. num_passengers = 1 + sum of num_passengers of all child nodes
    2. max_car_capacity = max(current node car_capacity, child nodes max_car_capacity)
    For leaves, num_passengers = 1, max_car_capacity = car_capacity
    For each node, num_passengers must be <= max_car_capacity
    """
    # validate solution format
    if not solution_matrix.shape == self.distance_matrix.shape:
      print(f'Invalid solution: array shape {solution_matrix.shape} does not match problem instance shape {self.distance_matrix.shape}')
      return False
    if not np.all(np.isin(solution_matrix, [1, 0])):
      print('Invalid solution: array elements must be either 1 or 0')
      return False

    # visited[i] == 1 if node i has been visited
    visited = np.zeros(self.num_nodes)
    # num_passengers[i] = num_passengers of the ith node
    num_passengers = np.ones(self.num_nodes)
    # max_capacities[i] = max_capacity of the ith node, we set the capacity of the destination node to the max passengers
    max_capacities = np.concatenate((self.car_capacities[:self.destination_index], np.array([self.num_nodes]), self.car_capacities[self.destination_index:]))

    ordered_nodes = [self.destination_index]
    unvisited_ptr = 0
    while unvisited_ptr < len(ordered_nodes):
      next_node = ordered_nodes[unvisited_ptr]
      if visited[next_node] == 1:
        # solution contains cycles
        print('Invalid solution: contains cycles')
        return False
      visited[next_node] = 1
      for i in range(self.num_nodes):
        if solution_matrix[i][next_node] == 1:
          ordered_nodes.append(i)
      unvisited_ptr += 1

    if not np.all(visited == 1):
      # solution doesn't span all nodes
       print('Invalid solution: does not span all nodes')
       return False

    # validate all subtrees
    selection_matrix = solution_matrix.copy() + np.identity(self.num_nodes)
    for node in ordered_nodes[::-1]:
      num_passengers[node] = np.dot(selection_matrix[:, node], num_passengers)
      max_capacities[node] = np.max(np.multiply(selection_matrix[:, node], max_capacities))
      # car not large enough to drive all passengers
      if num_passengers[node] > max_capacities[node]:
        print('Invalid solution: node has more passengers than max car capacity')
        return False
    return True
  
  def __str__(self):
    return f'Distance Matrix:\n{self.distance_matrix}\nDest: {self.destination_index}, Car Capacities: {[int(c) for c in self.car_capacities]}'