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

  def generate_problem_instance(self, num_friends: int, min_car_capacity: int, max_car_capacity: int, seed: int | None = None):
    """Choose a random subset of the cities"""
    if seed is not None:
      np.random.seed(seed)
    selected_nodes = np.random.randint(0, self.num_cities+1, size=num_friends+1)
    distance_matrix = self.intercity_distances[np.ix_(selected_nodes, selected_nodes)]
    # TODO: standardize on car capacities as list or np.array
    car_capacities = list(np.random.randint(min_car_capacity, max_car_capacity+1, size=num_friends))
    return ProblemInstance(distance_matrix, num_friends, car_capacities, node_labels=[str(n) for n in selected_nodes])

if __name__ == '__main__':
  test_city_dataset = CityDataset()
  print(test_city_dataset.city_to_coordinates)
  print(test_city_dataset.intercity_distances)
  test_instance = test_city_dataset.generate_problem_instance(5, 0, 3)
  print(test_instance.distance_matrix, test_instance.node_labels, test_instance.car_capacities)