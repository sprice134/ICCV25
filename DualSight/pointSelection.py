import matplotlib.pyplot as plt
import numpy as np
import random
import time
from itertools import combinations
import pandas as pd
import numpy as np
import random
from skimage.measure import label, regionprops
from scipy.ndimage import binary_erosion

random.seed(42)
np.random.seed(42)

# Naive maximization function with ensured valid cell points
def naiveMaximization(white_cells, num_points):
    max_distance = 0
    furthest_set = None
    num_checks = 0

    # Calculate the center (centroid) of the shape based on available white cells
    centroid_x = np.mean([cell[0] for cell in white_cells])
    centroid_y = np.mean([cell[1] for cell in white_cells])
    centroid = (centroid_x, centroid_y)

    # Start timing
    start_time = time.time()

    # Special case for 1 point: select the closest valid point to the centroid
    if num_points == 1:
        closest_to_centroid = min(white_cells, key=lambda cell: np.sqrt((cell[0] - centroid_x) ** 2 + (cell[1] - centroid_y) ** 2))
        furthest_set = [closest_to_centroid]
        max_distance = 0
        num_checks = 1
    else:
        # Check all combinations of 'num_points' cells
        for point_set in combinations(white_cells, num_points):
            num_checks += 1
            # Calculate pairwise Euclidean distances between all points in the set
            aggregate_distance = 0
            for i in range(num_points):
                for j in range(i + 1, num_points):
                    dist_ij = np.sqrt((point_set[i][0] - point_set[j][0])**2 + 
                                      (point_set[i][1] - point_set[j][1])**2)
                    aggregate_distance += dist_ij

            # Update max distance if this set has a larger aggregate distance
            if aggregate_distance > max_distance:
                max_distance = aggregate_distance
                furthest_set = point_set

    # End timing
    end_time = time.time()
    run_time = end_time - start_time

    # Print results
    # print(f"Naive - Number of checks: {num_checks}")
    # print(f"Naive - Total run time: {run_time:.6f} seconds")

    return furthest_set#, max_distance

# Simulated Annealing with increased patience
def simulatedAnnealingMaximization(white_cells, num_points, initial_temp=1000, cooling_rate=0.995, max_iterations=1000, patience=300):
    def calculate_total_distance(points):
        """Calculate the aggregate pairwise distance for a given set of points."""
        total_distance = 0
        for i in range(num_points):
            for j in range(i + 1, num_points):
                dist = np.sqrt((points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2)
                total_distance += dist
        return total_distance

    # Start timing
    start_time = time.time()

    # Initial random selection of points
    current_points = random.sample(white_cells, num_points)
    current_distance = calculate_total_distance(current_points)
    best_points = current_points[:]
    best_distance = current_distance

    num_checks = 0
    temperature = initial_temp
    no_improvement_counter = 0  # For early stopping

    for iteration in range(max_iterations):
        # Randomly swap out one point from current_points with a new point from white_cells
        new_points = current_points[:]
        swap_index = random.randint(0, num_points - 1)
        new_points[swap_index] = random.choice(white_cells)
        new_distance = calculate_total_distance(new_points)
        
        num_checks += 1

        # Decide whether to accept the new points based on simulated annealing criteria
        if new_distance > current_distance or np.exp((new_distance - current_distance) / temperature) > random.random():
            current_points = new_points
            current_distance = new_distance
            # Update best points if this is the best configuration found
            if new_distance > best_distance:
                best_points = new_points[:]
                best_distance = new_distance
                no_improvement_counter = 0  # Reset the counter on improvement
            else:
                no_improvement_counter += 1
        else:
            no_improvement_counter += 1

        # Early stopping condition
        if no_improvement_counter >= patience:
            # print(f"Simulated Annealing - Early stopping at iteration {iteration}")
            break

        # Cool down
        temperature *= cooling_rate
        if temperature < 1e-10:  # Stop if temperature is effectively zero
            break

    # End timing
    end_time = time.time()
    run_time = end_time - start_time

    # Print results
    # print(f"Simulated Annealing - Number of checks: {num_checks}")
    # print(f"Simulated Annealing - Total run time: {run_time:.6f} seconds")

    return best_points#, best_distance

# Hill Climbing algorithm
def hillClimbingMaximization(white_cells, num_points, max_iterations=1000):
    def calculate_total_distance(points):
        """Calculate the aggregate pairwise distance for a given set of points."""
        total_distance = 0
        for i in range(num_points):
            for j in range(i + 1, num_points):
                dist = np.sqrt((points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2)
                total_distance += dist
        return total_distance

    # Start timing
    start_time = time.time()
    # Initial random selection of points
    current_points = random.sample(white_cells, num_points)
    current_distance = calculate_total_distance(current_points)
    best_points = current_points[:]
    best_distance = current_distance

    num_checks = 0

    for iteration in range(max_iterations):
        improved = False
        
        # Try swapping out each point in current_points with a different point from white_cells
        for swap_index in range(num_points):
            for new_point in white_cells:
                if new_point != current_points[swap_index]:  # Avoid replacing with the same point
                    new_points = current_points[:]
                    new_points[swap_index] = new_point
                    new_distance = calculate_total_distance(new_points)
                    num_checks += 1

                    # Accept the new configuration if it improves the aggregate distance
                    if new_distance > current_distance:
                        current_points = new_points
                        current_distance = new_distance
                        improved = True
                        break
            if improved:
                break

        # Update best points if this is the best configuration found
        if current_distance > best_distance:
            best_points = current_points[:]
            best_distance = current_distance

        # Stop if no improvement was made in the last round
        if not improved:
            # print(f"Hill Climbing - Converged after {iteration} iterations")
            break

    # End timing
    end_time = time.time()
    run_time = end_time - start_time

    # Print results
    # print(f"Hill Climbing - Number of checks: {num_checks}")
    # print(f"Hill Climbing - Total run time: {run_time:.6f} seconds")

    return best_points#, best_distance

def clusterInitialization(white_cells, num_points):
    # Start by selecting a random point
    selected_points = [random.choice(white_cells)]
    
    while len(selected_points) < num_points:
        # For each remaining point, select the one that maximizes distance from all chosen points
        max_distance = 0
        next_point = None
        for point in white_cells:
            # Calculate the minimum distance from this point to all previously selected points
            min_distance_to_selected = min(np.sqrt((point[0] - selected[0]) ** 2 + (point[1] - selected[1]) ** 2) 
                                           for selected in selected_points)
            # If this distance is the largest we've seen, consider this point
            if min_distance_to_selected > max_distance:
                max_distance = min_distance_to_selected
                next_point = point

        # Add the selected point with max distance to the list
        selected_points.append(next_point)
    
    # Calculate the total pairwise distance for the selected points
    aggregate_distance = 0
    for i in range(num_points):
        for j in range(i + 1, num_points):
            dist_ij = np.sqrt((selected_points[i][0] - selected_points[j][0]) ** 2 + 
                              (selected_points[i][1] - selected_points[j][1]) ** 2)
            aggregate_distance += dist_ij

    return selected_points#, aggregate_distance

# Function to plot grid with furthest points highlighted
def plot_grid_circle(num_points=3, selection_algorithm=naiveMaximization):
    rows, cols = 10, 10  # Fixed grid size
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')

    # Circle parameters
    circle_radius = 0.5
    circle_center = (0.5, 0.5)
    
    # Grid size and cell dimensions
    cell_width = 1.0 / cols
    cell_height = 1.0 / rows

    # Lists to store white cells
    white_cells = []

    # Plot grid cells
    for row in range(rows):
        for col in range(cols):
            cell_x = col * cell_width
            cell_y = row * cell_height
            # Check if the cell center is within the circle
            cell_center_x = cell_x + cell_width / 2
            cell_center_y = cell_y + cell_height / 2
            distance_from_center = np.sqrt((cell_center_x - circle_center[0])**2 + (cell_center_y - circle_center[1])**2)

            # Cells within the circle
            if distance_from_center <= circle_radius:
                ax.add_patch(plt.Rectangle((cell_x, cell_y), cell_width, cell_height, edgecolor="black", facecolor="white"))
                white_cells.append((cell_center_x, cell_center_y))  # Store the coordinates of the white cell
            else:
                ax.add_patch(plt.Rectangle((cell_x, cell_y), cell_width, cell_height, color="black"))

    # Find and highlight the furthest set of white cells using the specified selection algorithm
    if len(white_cells) >= num_points:
        furthest_set, max_distance = selection_algorithm(white_cells, num_points)
        for cell in furthest_set:
            # Highlight each furthest cell in red
            ax.add_patch(plt.Rectangle((cell[0] - cell_width / 2, cell[1] - cell_height / 2), cell_width, cell_height, color="red"))

        # Print the results for this plot
        print(f"Number of points: {num_points}, Total white cells: {len(white_cells)}, "
              f"Aggregate distance of furthest set ({num_points} points): {max_distance:.2f}")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

# Random selection algorithm
def randomSelection(white_cells, num_points):
    selected_points = random.sample(white_cells, num_points)
    aggregate_distance = 0
    for i in range(num_points):
        for j in range(i + 1, num_points):
            dist_ij = np.sqrt((selected_points[i][0] - selected_points[j][0]) ** 2 + 
                              (selected_points[i][1] - selected_points[j][1]) ** 2)
            aggregate_distance += dist_ij
    return selected_points#, aggregate_distance

# def voronoi_wrapper(mask, num_points, iterations=50):
#     """
#     Perform Voronoi-based optimization to select points.

#     Parameters:
#         mask (2D array): Binary mask where 1 represents valid cells, and 0 represents background.
#         num_points (int): Number of points to select.
#         iterations (int): Number of optimization iterations.

#     Returns:
#         List of optimized points (in pixel coordinates).
#     """
#     def voronoi_optimization(mask, num_points, iterations=50):
#         mask = np.array(mask)

#         def initialize_points(binary_mask, n_points):
#             mask_coords = np.column_stack(np.where(binary_mask))
#             indices = np.random.choice(len(mask_coords), n_points, replace=False)
#             return mask_coords[indices]

#         def voronoi_partition(binary_mask, points):
#             rows, cols = binary_mask.shape
#             region_map = np.zeros_like(binary_mask, dtype=int)

#             x, y = np.meshgrid(np.arange(cols), np.arange(rows))
#             coords = np.column_stack((y.ravel(), x.ravel()))

#             distances = np.linalg.norm(coords[:, None, :] - points[None, :, :], axis=2)
#             nearest_point = np.argmin(distances, axis=1)
#             region_map[binary_mask] = nearest_point[binary_mask.ravel()]
#             return region_map

#         def optimize_points(binary_mask, region_map, n_points):
#             new_points = []
#             for i in range(n_points):
#                 region_coords = np.column_stack(np.where(region_map == i))
#                 if len(region_coords) > 0:
#                     centroid = region_coords.mean(axis=0)
#                     new_points.append(centroid)
#                 else:
#                     mask_coords = np.column_stack(np.where(binary_mask))
#                     new_points.append(mask_coords[np.random.choice(len(mask_coords))])
#             return np.array(new_points)

#         binary_mask = mask > 0
#         print(binary_mask.shape)
#         points = initialize_points(binary_mask, num_points)
        
#         for _ in range(iterations):
#             region_map = voronoi_partition(binary_mask, points)
#             new_points = optimize_points(binary_mask, region_map, num_points)
#             if np.allclose(points, new_points, atol=1e-2):
#                 break
#             points = new_points
#         print(points.tolist())
#         return points.tolist()

#     return voronoi_optimization(mask, num_points, iterations)

def voronoi_optimization_from_coords(coords, num_points, iterations=50):
    """
    Perform Voronoi-based optimization to select points from a list of coordinates.

    Parameters:
        coords (list of tuples): List of possible (x, y) coordinates.
        num_points (int): Number of points to select.
        iterations (int): Number of optimization iterations.

    Returns:
        List of optimized points (x, y).
    """
    # Ensure coordinates are a NumPy array
    coords = np.array(coords)

    # Step 1: Initialize points by randomly sampling from the coordinate list
    initial_indices = np.random.choice(len(coords), num_points, replace=False)
    points = coords[initial_indices]

    # Step 2: Define Voronoi partitioning
    def voronoi_partition(coords, points):
        """Assign each coordinate to the nearest point."""
        distances = np.linalg.norm(coords[:, None] - points[None, :], axis=2)
        nearest_point = np.argmin(distances, axis=1)
        return nearest_point

    # Step 3: Optimize points to balance regions
    for _ in range(iterations):
        # Assign each coordinate to its nearest point
        region_assignment = voronoi_partition(coords, points)

        # Compute new centroids for each region
        new_points = []
        for i in range(num_points):
            region_coords = coords[region_assignment == i]
            if len(region_coords) > 0:
                centroid = region_coords.mean(axis=0)
                new_points.append(centroid)
            else:
                # If a region is empty, reinitialize the point randomly
                new_points.append(coords[np.random.choice(len(coords))])
        new_points = np.array(new_points)

        # Check for convergence
        if np.allclose(points, new_points, atol=1e-2):
            break

        points = new_points

    return points.tolist()



def select_furthest_points_from_mask(
    mask, num_points, dropout_percentage=0, ignore_border_percentage=0, algorithm="Naive", select_perimeter=False
):
    """
    Selects N furthest points from a binary mask with specified dropout, border ignore, and algorithm.

    Parameters:
        mask (2D array): Binary mask where 1 represents valid cells, and 0 represents background.
        num_points (int): Number of points to select.
        dropout_percentage (float): Percentage of cells to drop randomly (0-100).
        ignore_border_percentage (float): Percentage of border width to ignore (0-100).
        algorithm (str): Algorithm to use for point selection (e.g., "Naive", "Simulated Annealing", etc.).
        select_perimeter (bool): Whether to select only the perimeter of the object.

    Returns:
        List of selected points, aggregate distance, run time.
    """
    # Step 1: Extract white cells (coordinates) from the binary mask
    rows, cols = np.where(mask > 0)
    white_cells = list(zip(rows, cols))

    # Step 2: If multiple objects are present, keep only the largest connected component
    labeled_mask = label(mask)
    regions = regionprops(labeled_mask)
    if len(regions) > 1:
        largest_region = max(regions, key=lambda r: r.area)
        white_cells = [(row, col) for row, col in white_cells if labeled_mask[row, col] == largest_region.label]

    # Step 3: Select only the perimeter (optional)
    if select_perimeter:
        # Compute the perimeter by subtracting the eroded mask from the original mask
        eroded_mask = binary_erosion(mask)
        perimeter_mask = mask & ~eroded_mask  # Subtract eroded mask from original mask
        rows, cols = np.where(perimeter_mask > 0)
        white_cells = list(zip(rows, cols))

    # Step 4: Apply border filtering
    if ignore_border_percentage > 0:
        object_rows = [r for r, _ in white_cells]
        object_columns = [c for _, c in white_cells]
        min_row = min(object_rows)
        max_row = max(object_rows)
        min_col = min(object_columns)
        max_col = max(object_columns)
        actual_width = max_col - min_col + 1
        actual_height = max_row - min_row + 1
        ignore_width = int(ignore_border_percentage / 100 * actual_width)
        ignore_height = int(ignore_border_percentage / 100 * actual_height)
        inner_row_start = min_row + ignore_height
        inner_row_end = max_row - ignore_height
        inner_col_start = min_col + ignore_width
        inner_col_end = max_col - ignore_width
        white_cells = [
            (r, c) for r, c in white_cells
            if inner_row_start <= r < inner_row_end and inner_col_start <= c < inner_col_end
        ]

    # Step 5: Apply dropout
    num_cells_to_keep = int(len(white_cells) * (1 - dropout_percentage / 100))
    white_cells = random.sample(white_cells, num_cells_to_keep)

    
    # Step 6: Choose algorithm and select points
    algorithm_functions = {
        "Naive": naiveMaximization,
        "Simulated Annealing": simulatedAnnealingMaximization,
        "Hill Climbing": hillClimbingMaximization,
        "Cluster Initialization": clusterInitialization,
        "Random": randomSelection,
        "Voronoi": voronoi_optimization_from_coords,  # Now callable directly
    }

    # Convert pixel coordinates to normalized grid coordinates for algorithms
    white_cells_normalized = [(x / mask.shape[0], y / mask.shape[1]) for x, y in white_cells]
    start_time = time.time()
    # Select points and calculate aggregate distance
    selected_points = algorithm_functions[algorithm](white_cells_normalized, num_points)
    end_time = time.time()
    run_time = end_time - start_time


    # Convert normalized selected points back to original pixel coordinates with clipping
    selected_points_pixel = [
        (min(max(int(p[0] * mask.shape[0]), 0), mask.shape[0] - 1),  # Clip x coordinate
         min(max(int(p[1] * mask.shape[1]), 0), mask.shape[1] - 1))  # Clip y coordinate
        for p in selected_points
    ]
    return selected_points_pixel, 0, run_time

