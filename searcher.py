import numpy as np 
import networkx as nx

from dataset import CityDataset
from solver import Solver
from problem_instance import ProblemInstance

NODE_CAPACITY_LABEL = 'node_capacity'
MAX_CAPACITY_LABEL = 'max_capacity'
NUM_PASSENGERS_LABEL = 'num_passengers'

# TODO: add unit tests
class Optimizer:
    def solution_to_tree(self, solution: np.array, problem_instance: ProblemInstance) -> nx.DiGraph:
        tree = nx.DiGraph()
        for i in range(solution.shape[0]):
            for j in range(solution.shape[1]):
                if solution[i][j] == 1:
                    tree.add_edge(i, j)
        
        # iterate from leaves to root and populate node labels
        ordered_nodes = [problem_instance.destination_index]
        node_ptr = 0
        while node_ptr < len(ordered_nodes):
            curr_node = ordered_nodes[node_ptr]
            for u, _ in tree.in_edges(nbunch=curr_node):
                ordered_nodes.append(u)
            node_ptr += 1 
        
        node_attributes = {}
        for node in ordered_nodes[::-1]:
            node_to_capacity_index = node if node < problem_instance.destination_index else node - 1
            curr_capacity = problem_instance.car_capacities[node_to_capacity_index] if node != problem_instance.destination_index else float('inf')
            max_capacity = curr_capacity 
            num_passengers = 1 if node != problem_instance.destination_index else 0
            for u, _ in tree.in_edges(nbunch=node):
                max_capacity = max(node_attributes[u][MAX_CAPACITY_LABEL], max_capacity)
                num_passengers += node_attributes[u][NUM_PASSENGERS_LABEL]
            node_attributes[node] = {
                NODE_CAPACITY_LABEL: curr_capacity,
                MAX_CAPACITY_LABEL: max_capacity,
                NUM_PASSENGERS_LABEL: num_passengers
            }
        nx.set_node_attributes(tree, node_attributes)
        return tree 

    # TODO: rename to swap_node_parent
    def move_node(self, tree: nx.DiGraph, remove_edge: tuple[int,int], add_edge: tuple[int,int]):
        """
        Return (new_tree: nx.DiGraph, is_valid: bool)
        """
        if remove_edge[0] != add_edge[0]:
            raise RuntimeError('source node must be the same for remove_edge and add_edge')
        new_tree: nx.DiGraph = tree.copy()
        node_attrs = new_tree.nodes(data=True)
        root_node = [node for node, out_degree in new_tree.out_degree() if out_degree == 0][0]
        move_node = remove_edge[0]
        is_node_invalid = np.zeros(len(node_attrs))

        # remove edge and update attributes
        curr_node = remove_edge[1]
        while curr_node != root_node:
            new_num_passengers = node_attrs[curr_node][NUM_PASSENGERS_LABEL] - node_attrs[move_node][NUM_PASSENGERS_LABEL]
            new_max_capacity = node_attrs[curr_node][NODE_CAPACITY_LABEL]
            for u, _ in new_tree.in_edges(nbunch=curr_node):
                if u != move_node:
                    new_max_capacity = max(new_max_capacity, node_attrs[u][MAX_CAPACITY_LABEL])
            node_attrs[curr_node][NUM_PASSENGERS_LABEL] = new_num_passengers
            node_attrs[curr_node][MAX_CAPACITY_LABEL] = new_max_capacity
            if new_num_passengers > new_max_capacity:
                is_node_invalid[curr_node] = 1
            # move to parent
            for _, v in new_tree.out_edges(nbunch=curr_node):
                curr_node = v
        new_tree.remove_edge(*remove_edge)

        # add edge and update attributes
        new_tree.add_edge(*add_edge)
        curr_node = add_edge[1]
        visited = np.zeros(len(node_attrs))
        while curr_node != root_node:
            if visited[curr_node] == 1:
                return new_tree, False
            visited[curr_node] = 1
            new_num_passengers = node_attrs[curr_node][NUM_PASSENGERS_LABEL] + node_attrs[move_node][NUM_PASSENGERS_LABEL]
            new_max_capacity = max(node_attrs[curr_node][MAX_CAPACITY_LABEL], node_attrs[move_node][MAX_CAPACITY_LABEL])
            node_attrs[curr_node][NUM_PASSENGERS_LABEL] = new_num_passengers
            node_attrs[curr_node][MAX_CAPACITY_LABEL] = new_max_capacity
            if new_num_passengers > new_max_capacity:
                return new_tree, False 
            is_node_invalid[curr_node] = 0
            # move to parent
            for _, v in new_tree.out_edges(nbunch=curr_node):
                curr_node = v

        return new_tree, sum(is_node_invalid) == 0
    
    def get_tree_cost(self, tree: nx.DiGraph, problem_instance: ProblemInstance):
        cost = 0
        for i, j in tree.in_edges():
            cost += problem_instance.distance_matrix[i][j] 
        return cost

    def tree_to_solution(self, tree: nx.DiGraph) -> np.array:
        num_nodes = len(tree.nodes())
        solution = np.zeros((num_nodes, num_nodes))
        for i, j in tree.in_edges():
            solution[i][j] = 1
        return solution
    
    def optimize_step(self, tree: nx.DiGraph, problem_instance: ProblemInstance) -> nx.DiGraph:
        """
        Find the tree wih the lowest cost after any single move swap
        Returns: (best_tree, is_best_tree_original)
        """
        num_nodes = len(tree.nodes())
        best_tree = tree 
        best_cost = self.get_tree_cost(tree, problem_instance)
        is_best_tree_original = True
        for i in range(num_nodes):
            for j in range(num_nodes):
                if tree.out_degree(nbunch=i) == 0 or i == j:
                    continue 
                parent = [p for _, p in tree.out_edges(nbunch=i)][0]
                new_tree, is_valid = self.move_node(tree, (i, parent), (i, j))
                if not is_valid:
                    continue
                new_cost = self.get_tree_cost(new_tree, problem_instance) 
                if new_cost >= best_cost:
                    continue 
                best_tree = new_tree 
                best_cost = new_cost 
                is_best_tree_original = False 
        return best_tree, is_best_tree_original
    
    def optimized_greedy_solution(self, problem_instance: ProblemInstance) -> np.array:
        solver = Solver()
        solution = solver.greedy_solution(problem_instance) 
        tree = self.solution_to_tree(solution, problem_instance) 
        while True:
            tree, is_best_tree_original = self.optimize_step(tree, problem_instance)
            if is_best_tree_original:
                break 
        optimized_solution = self.tree_to_solution(tree)
        return optimized_solution
    

def test_move_node():
    optimizer = Optimizer()

    distance_matrix = np.ones((5, 5))
    destination_index = 3
    car_capacities = np.array([3, 1, 2, 2])
    problem_instance = ProblemInstance(distance_matrix, destination_index, car_capacities)
    valid_solution = np.array(
        [[0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],]
    )

    tree = optimizer.solution_to_tree(valid_solution, problem_instance) 
    # validate tree
    assert(tree.nodes()[0][NUM_PASSENGERS_LABEL] == 1)
    assert(tree.nodes()[1][NUM_PASSENGERS_LABEL] == 3)
    assert(tree.nodes()[2][NUM_PASSENGERS_LABEL] == 1)
    assert(tree.nodes()[3][NUM_PASSENGERS_LABEL] == 4)
    assert(tree.nodes()[4][NUM_PASSENGERS_LABEL] == 1)

    assert(tree.nodes()[0][MAX_CAPACITY_LABEL] == 3)
    assert(tree.nodes()[1][MAX_CAPACITY_LABEL] == 3)
    assert(tree.nodes()[2][MAX_CAPACITY_LABEL] == 2)
    assert(tree.nodes()[3][MAX_CAPACITY_LABEL] == float('inf'))
    assert(tree.nodes()[4][MAX_CAPACITY_LABEL] == 2)

    assert(tree.nodes()[0][NODE_CAPACITY_LABEL] == car_capacities[0])
    assert(tree.nodes()[1][NODE_CAPACITY_LABEL] == car_capacities[1])
    assert(tree.nodes()[2][NODE_CAPACITY_LABEL] == car_capacities[2])
    assert(tree.nodes()[3][NODE_CAPACITY_LABEL] == float('inf'))
    assert(tree.nodes()[4][NODE_CAPACITY_LABEL] == car_capacities[3])

    # move creates cycle
    _, is_valid = optimizer.move_node(tree, (1, 3), (1, 2))
    assert(not is_valid)

    # move makes new parent over capacity
    _, is_valid = optimizer.move_node(tree, (1, 3), (1, 4))
    assert(not is_valid) 

    # valid move
    new_tree, is_valid = optimizer.move_node(tree, (0, 1), (0, 4))
    assert(is_valid)
    assert(new_tree.nodes()[0][NUM_PASSENGERS_LABEL] == 1)
    assert(new_tree.nodes()[1][NUM_PASSENGERS_LABEL] == 2)
    assert(new_tree.nodes()[2][NUM_PASSENGERS_LABEL] == 1)
    assert(new_tree.nodes()[3][NUM_PASSENGERS_LABEL] == 4)
    assert(new_tree.nodes()[4][NUM_PASSENGERS_LABEL] == 2)

    assert(new_tree.nodes()[0][MAX_CAPACITY_LABEL] == 3)
    assert(new_tree.nodes()[1][MAX_CAPACITY_LABEL] == 2)
    assert(new_tree.nodes()[2][MAX_CAPACITY_LABEL] == 2)
    assert(new_tree.nodes()[3][MAX_CAPACITY_LABEL] == float('inf'))
    assert(new_tree.nodes()[4][MAX_CAPACITY_LABEL] == 3)

    assert(new_tree.nodes()[0][NODE_CAPACITY_LABEL] == car_capacities[0])
    assert(new_tree.nodes()[1][NODE_CAPACITY_LABEL] == car_capacities[1])
    assert(new_tree.nodes()[2][NODE_CAPACITY_LABEL] == car_capacities[2])
    assert(new_tree.nodes()[3][NODE_CAPACITY_LABEL] == float('inf'))
    assert(new_tree.nodes()[4][NODE_CAPACITY_LABEL] == car_capacities[3])

if __name__ == '__main__':
    test_move_node()