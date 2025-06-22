from problem_instance import ProblemInstance
import unittest
import numpy as np

class TestProblemInstance(unittest.TestCase):
    def test_is_valid(self):
        distance_matrix = np.ones((5, 5))
        destination_index = 4
        car_capacities = np.array([0, 0, 0, 4])
        problem_instance = ProblemInstance(distance_matrix, destination_index, car_capacities)
        assert(problem_instance.is_valid)

        car_capacities = np.array([0, 0, 0, 3])
        problem_instance = ProblemInstance(distance_matrix, destination_index, car_capacities)
        assert(not problem_instance.is_valid)

    def test_validate_solution(self):
        distance_matrix = np.ones((5, 5))
        destination_index = 3
        car_capacities = np.array([1, 3, 1, 2])
        problem_instance = ProblemInstance(distance_matrix, destination_index, car_capacities)

        valid_solution = np.array(
            [[0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],]
        )
        is_valid = problem_instance.validate_solution(valid_solution)
        assert(is_valid)

        cyclic_solution = np.array(
            [[0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],]
        )
        is_valid = problem_instance.validate_solution(cyclic_solution)
        assert(not is_valid)

        incomplete_solution = np.array(
            [[0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],]
        )
        is_valid = problem_instance.validate_solution(incomplete_solution)
        assert(not is_valid)

        insufficient_car_capacity_solution = np.array(
            [[0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],]
        )
        is_valid = problem_instance.validate_solution(insufficient_car_capacity_solution)
        assert(not is_valid)

        invalid_shape_solution = np.array(
            [[1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],]
        )
        is_valid = problem_instance.validate_solution(invalid_shape_solution)
        assert(not is_valid)

        invalid_elements_solution = np.array(
            [[0, 1, 0, 0, 0],
            [0, 0, 0, 5, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],]
        )
        is_valid = problem_instance.validate_solution(invalid_elements_solution)
        assert(not is_valid)
        
if __name__ == '__main__':
    unittest.main()