import numpy as np 
import networkx as nx

from dataset import CityDataset
from solver import Solver
from problem_instance import ProblemInstance

NODE_CAPACITY_LABEL = 'node_capacity'
MAX_CAPACITY_LABEL = 'max_capacity'
NUM_PASSENGERS_LABEL = 'num_passengers'

def solution_to_tree(solution: np.array, problem_instance: ProblemInstance) -> nx.DiGraph:
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

def is_valid_move(tree: nx.DiGraph, edge_to_delete: tuple[int, int], edge_to_add: tuple[int, int]) -> bool:
    first_parent = tree.node

def test():
    problem_instance = CityDataset().generate_problem_instance(20, 0, 10, force_valid=True, seed=3)
    solution = Solver().greedy_solution(problem_instance)
    assert(problem_instance.validate_solution(solution))
    print(solution)

    tree = solution_to_tree(solution, problem_instance)
    print(solution)
    print([edge for edge in tree.in_edges()])
    print([edge for edge in tree.out_edges()])
    print(tree.nodes(data=True))

if __name__ == '__main__':
    test()