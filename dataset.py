import math
import numpy as np

from problem_instance import ProblemInstance

# Dataset from: ATT48 https://people.sc.fsu.edu/~jburkardt/datasets/tsp/tsp.html
city_coordinates = """6734 1453
2233   10
5530 1424
 401  841
3082 1644
7608 4458
7573 3716
7265 1268
6898 1885
1112 2049
5468 2606
5989 2873
4706 2674
4612 2035
6347 2683
6107  669
7611 5184
7462 3590
7732 4723
5900 3561
4483 3369
6101 1110
5199 2182
1633 2809
4307 2322
 675 1006
7555 4819
7541 3981
3177  756
7352 4506
7545 2801
3245 3305
6426 3173
4608 1198
  23 2216
7248 3779
7762 4595
7392 2244
3484 2829
6271 2135
4985  140
1916 1569
7280 4899
7509 3239
  10 2676
6807 2993
5185 3258
3023 1942"""

class CityDataset:
  def __init__(self):
    # maps city_id to tuple representing coordinate
    self.city_to_coordinates = {}
    # 2d array where intercity_distances[i][j] is the distance from city i to city j
    self.intercity_distances = None
    self.num_cities = 0

    self.initialize_data(city_coordinates)

  def initialize_data(self, dataset_str):
    for i, line in enumerate(dataset_str.split('\n')):
      # looks like ['2233', '', '', '10']
      coords = line.strip().split(' ')
      self.city_to_coordinates[i] = (int(coords[0]), int(coords[-1]))

    self.num_cities = len(self.city_to_coordinates)
    self.intercity_distances = np.zeros((self.num_cities, self.num_cities))
    for i, city_a in enumerate(self.city_to_coordinates):
      for j, city_b in enumerate(self.city_to_coordinates):
        city_a_coords = self.city_to_coordinates[city_a]
        city_b_coords = self.city_to_coordinates[city_b]
        self.intercity_distances[i][j] = int(math.sqrt((city_a_coords[0] - city_b_coords[0])**2 + (city_a_coords[1] - city_b_coords[1])**2))

    # validate intercity coordinates is symmetric
    for i in range(self.num_cities):
      for j in range(self.num_cities):
        assert(self.intercity_distances[i][j] == self.intercity_distances[j][i])

  def generate_problem_instance(self, num_friends: int, min_car_capacity: int, max_car_capacity: int, force_valid: bool, seed: int | None = None):
    """
    Generate a problem instance with a random subset of the cities
    If force_valid, guarantee sum(car_capacities) > num_friends
    """
    if max_car_capacity < 1:
      raise RuntimeError(f'max_car_capacity {max_car_capacity} must be >= 1')
    if min_car_capacity < 0:
      raise RuntimeError(f'min_car_capacity {min_car_capacity} must be >= 0')
    
    if seed is not None:
      np.random.seed(seed)
    selected_nodes = np.random.randint(0, self.num_cities, size=num_friends+1)
    distance_matrix = self.intercity_distances[np.ix_(selected_nodes, selected_nodes)]

    if not force_valid:
      car_capacities = np.random.randint(min_car_capacity, max_car_capacity+1, size=num_friends)
    else:
      shuffled_nodes = np.random.permutation(np.arange(num_friends))
      car_capacities = np.zeros(num_friends)
      curr_car_capacity = 0
      for i, n in enumerate(shuffled_nodes):
        # max capacity we could get ignoring the current node
        max_future_car_capacity = (num_friends-i-1) * max_car_capacity
        needed_car_capacity = num_friends - curr_car_capacity
        min_valid_curr_capacity = max(min_car_capacity, needed_car_capacity - max_future_car_capacity)
        car_capacities[n] = np.random.randint(min_valid_curr_capacity, max_car_capacity + 1) 
        curr_car_capacity += car_capacities[n]
    return ProblemInstance(distance_matrix, num_friends, car_capacities, node_labels=[str(n) for n in selected_nodes])

def test_generate_problem_instance():
  city_dataset = CityDataset() 

  # test we can generate valid problem instances
  for i in range(1000):
    problem_instance = city_dataset.generate_problem_instance(10, 0, 3, True)
    assert(problem_instance.is_valid)

  # test that we generate invalid problem instances 
  found_invalid = False 
  for i in range(1000):
    problem_instance = city_dataset.generate_problem_instance(10, 0, 3, False)
    found_invalid = not problem_instance.is_valid or found_invalid 
  assert(found_invalid)

if __name__ == '__main__':
  test_city_dataset = CityDataset()
  test_instance = test_city_dataset.generate_problem_instance(5, 0, 3, False)
  print(test_instance)

  test_generate_problem_instance()