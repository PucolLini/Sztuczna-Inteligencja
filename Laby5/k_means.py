import numpy as np


def initialize_centroids_forgy(data, k):
    # TODO implement random initialization X
    return data[np.random.choice(data.shape[0], k, replace=False)]


def initialize_centroids_kmeans_pp(data, k):
    # TODO implement kmeans++ initizalization X
    centroids = np.zeros((k, data.shape[1]))
    centroids[0] = data[np.random.choice(data.shape[0], 1, replace=False)]

    for i in range(1, k):
        distances = np.zeros(data.shape[0])
        for j in range(data.shape[0]):
            # obliczanie najblizszego centroidu do i-tego centroidu (do tych, co juz sa zainicjowane :i)
            distances[j] = min([np.linalg.norm(data[j] - centroid)**2 for centroid in centroids[:i]])
        # wybieranie danej, która jest najdalsza od najnowszego centroidu
        centroids[i] = data[np.argmax(distances)]
    return centroids


def assign_to_cluster(data, centroids):
    # TODO find the closest cluster for each data point
    assignments = np.zeros(data.shape[0], dtype=int)

    for i in range(data.shape[0]):
        distances = [np.linalg.norm(data[i] - c)**2 for c in centroids]
        assignments[i] = np.argmin(distances)

    return assignments


def update_centroids(data, assignments):
    # TODO find new centroids based on the assignments
    new_centroids = np.zeros((len(assignments), data.shape[1]))

    for j in range(len(assignments)):
        assigned_data_points = data[assignments == j] # wszystkie punkty przypisane do tego klastra
        if len(assigned_data_points) > 0:
            new_centroids[j] = np.mean(assigned_data_points, axis=0) # srednia punktow

    return new_centroids


def mean_intra_distance(data, assignments, centroids):
    return np.sqrt(np.sum((data - centroids[assignments, :])**2))


def k_means(data, num_centroids, kmeansplusplus=False):
    # centroids initizalization
    if kmeansplusplus:
        centroids = initialize_centroids_kmeans_pp(data, num_centroids)
    else:
        centroids = initialize_centroids_forgy(data, num_centroids)

    assignments = assign_to_cluster(data, centroids)
    for i in range(100): # max number of iteration = 100
        print(f"Intra distance after {i} iterations: {mean_intra_distance(data, assignments, centroids)}")
        centroids = update_centroids(data, assignments)
        new_assignments = assign_to_cluster(data, centroids)
        if np.all(new_assignments == assignments): # stop if nothing changed
            break
        else:
            assignments = new_assignments

    return new_assignments, centroids, mean_intra_distance(data, new_assignments, centroids)

