from problem_instance import ProblemInstance
import unittest
import numpy as np

class TestProblemInstance(unittest.TestCase):
    def test_validate_solution(self):
        distance_matrix = np.ones((5, 5))
        destination_index = 3
        car_capacities = [1, 3, 1, 2]
        problem_instance = ProblemInstance(distance_matrix, destination_index, car_capacities)

        valid_solution = np.array(
            [[0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],]
        )
        is_valid = problem_instance.validate_solution(valid_solution)
        assert(is_valid == True)

        cyclic_solution = np.array(
            [[0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],]
        )
        is_valid = problem_instance.validate_solution(cyclic_solution)
        assert(is_valid == False)

        incomplete_solution = np.array(
            [[0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],]
        )
        is_valid = problem_instance.validate_solution(incomplete_solution)
        assert(is_valid == False)

        insufficient_car_capacity_solution = np.array(
            [[0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],]
        )
        is_valid = problem_instance.validate_solution(insufficient_car_capacity_solution)
        assert(is_valid == False)

        invalid_shape_solution = np.array(
            [[1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],]
        )
        is_valid = problem_instance.validate_solution(invalid_shape_solution)
        assert(is_valid == False)

        invalid_elements_solution = np.array(
            [[0, 1, 0, 0, 0],
            [0, 0, 0, 5, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],]
        )
        is_valid = problem_instance.validate_solution(invalid_elements_solution)
        assert(is_valid == False)
        
if __name__ == '__main__':
    unittest.main()