import random
import math

def generate_initial_solutions(num_cities, num_nests):
    # Generate random permutations of cities as initial solutions
    solutions = []
    for _ in range(num_nests):
        cities = list(range(num_cities))
        random.shuffle(cities)
        solutions.append(cities)
    return solutions

def calculate_distance(city1, city2, distances):
    # Retrieve distance from distance matrix
    return distances[city1][city2]

def calculate_total_distance(solution, distances):
    total_distance = 0
    for i in range(len(solution) - 1):
        city1 = solution[i]
        city2 = solution[i + 1]
        total_distance += calculate_distance(city1, city2, distances)
    # Add distance back to the starting city
    total_distance += calculate_distance(solution[-1], solution[0], distances)
    return total_distance

def generate_new_solution(solution):
    # Apply Levy flight to create a new solution
    new_solution = solution.copy()
    step_size = random.uniform(0, 1) * math.pow(random.random(), -1.5)
    start_index = random.randint(0, len(solution) - 2)
    end_index = random.randint(start_index + 1, len(solution) - 1)
    sublist = new_solution[start_index:end_index]
    sublist = sublist[::-1]  # Reverse the sublist
    new_solution[start_index:end_index] = sublist
    return new_solution

def cuckoo_search(distances, num_cities, num_nests, pa, max_iterations):
    best_solution = None
    best_distance = float('inf')

    solutions = generate_initial_solutions(num_cities, num_nests)

    for iteration in range(max_iterations):
        for i in range(num_nests):
            new_solution = generate_new_solution(solutions[i])
            new_distance = calculate_total_distance(new_solution, distances)

            if new_distance < calculate_total_distance(solutions[i], distances):
                solutions[i] = new_solution

            if new_distance < best_distance:
                best_solution = new_solution
                best_distance = new_distance

        # Replace some nests with new random solutions (pa = probability of discovery)
        for i in range(num_nests):
            if random.random() < pa:
                solutions[i] = generate_initial_solutions(num_cities, 1)[0]

    return best_solution, best_distance
distances = [[0, 10, 20, 30], [10, 0, 40, 50], [20, 40, 0, 60], [30, 50, 60, 0]]
best_solution, best_distance = cuckoo_search(distances, 4, 10, 0.25, 100)
print("Best solution:", best_solution)
print("Best distance:", best_distance)
