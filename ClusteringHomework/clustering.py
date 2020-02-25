import numpy as np
from math import inf
import random
from itertools import combinations
import matplotlib.pyplot as plt
import math

# print("hello, world")


def standardize_c2_data():
    # list that contains all the vector ids
    id_list = []
    # list that stores all the x coordinates of points
    x_coord_list = []
    # list that stores all y coordinates of points
    y_coord_list = []
    # list that contains all the points xi
    X = []

    # read in the data
    with open('C2.txt', 'r') as data:
        c2_data = data.readlines()

    # process the data where points are strings of characters separated by new lines and two tabs
    for point in c2_data:
        point = point.replace('\n', '')
        point = point.split('  ')

        # get components of the points and convert them to id, x coordinate, y coordinate
        point_id = int(point[0])
        x_coord = float(point[1])
        y_coord = float(point[2])

        # update the lists
        id_list.append(point_id)
        x_coord_list.append(x_coord)
        y_coord_list.append(y_coord)

    for i in range(len(id_list)):
        x_coord = x_coord_list[i]
        y_coord = y_coord_list[i]
        point = (x_coord, y_coord)
        point_id = id_list[i]
        data_point = (point_id, point)
        X.append(data_point)
    return X

def euclidean_distance(point1, point2):
    point_1 = np.array(point1)
    point_2 = np.array(point2)
    distance = np.linalg.norm(point_1 - point_2)
    return distance

def gonzalez_algorithm(data, k):
    centers = []
    # the first center as specified by Professor Phillips
    c1 = data[0]
    centers.append(c1)
    for i in range(1, k):
        new_center = calculate_farthest_point(data, centers)
        centers.append(new_center)
    return centers

def calculate_farthest_point(data, centers):
    distances = [inf for i in range(len(data))]
    for center_id, center_point in centers:
        for id, point in data:
            distance = euclidean_distance(center_point, point)
            if distance < distances[id - 1]:
                distances[id - 1] = distance
    return data[np.argmax(distances)]





def kmeans_plus_plus(data, k):
    centers = []
    c1 = data[0]
    centers.append(c1)
    while len(centers) != k:
        new_center = kmeans_plus_plus_neighbors(data, centers)
        centers.append(new_center)
    return centers

def kmeans_plus_plus_neighbors(data, centers):
    clusters = [0] * len(data)
    distances_list = [0] * len(data)
    for i in range(len(data)):
        min_distance = inf
        for j in range(len(centers)):
            point_id, point = data[i]
            center_id, center = centers[j]
            distance = euclidean_distance(center, point)
            if distance < min_distance:
                min_distance = distance
                clusters[point_id - 1] = centers[j]
    for h in range(len(data)):
        point_id, point = data[h]
        center_id, center = clusters[point_id - 1]
        distance = euclidean_distance(center, point)
        distance_squared = math.pow(distance, 2)
        distances_list[h] = distance_squared

    distances_sum = sum(distances_list)
    for w in range(len(distances_list)):
        distances_list[w] = distances_list[w] / distances_sum


    X = random.uniform(0, 1)
    print(f"X value: {X}")

    sum_counter = 0
    counter = None
    for g in range(len(distances_list)):
        if sum_counter >= X:
            counter = g
            break
        sum_counter += distances_list[g]
        print(f"sum counter", sum_counter)

    new_center = data[counter]
    return new_center

def lloyds_driver(data, centers):
    new_centers, clusters = lloyds_algorithm(data, centers)
    counter = 0
    while centers != new_centers:
        centers = new_centers
        new_centers, clusters = lloyds_algorithm(data, centers)
        counter += 1
        print(f"The counter value is {counter}")
    return centers, clusters

def lloyds_algorithm(data, centers):
    center1_id, center1 = centers[0]
    center2_id, center2 = centers[1]
    center3_id, center3 = centers[2]
    new_centers = []
    clusters = [0] * len(data)
    cluster1 = []
    cluster2 = []
    cluster3 = []
    cluster_map = {x: [] for x in centers}
    new_centers = []
    for i in range(len(data)):
        min_distance = inf
        for j in range(len(centers)):
            point_id, point = data[i]
            center_id, center_point = centers[j]
            distance = euclidean_distance(center_point, point)
            if distance < min_distance:
                min_distance = distance
                clusters[point_id - 1] = centers[j]
    for k in range(len(clusters)):
        center_id, center = clusters[k]
        if center_id == center1_id:
            cluster1.append(data[k])
        elif center_id == center2_id:
            cluster2.append(data[k])
        else:
            cluster3.append(data[k])

    for c in range(len(centers)):
        center_id, center = centers[c]
        if center_id == center1_id:
            center = centers[c]
            cluster_map[center] = cluster1
        elif center_id == center2_id:
            center = centers[c]
            cluster_map[center] = cluster2
        else:
            center = centers[c]
            cluster_map[center] = cluster3

    for c in centers:
        cluster = cluster_map[c]
        new_center = calculate_median_point(cluster)
        new_centers.append(new_center)

    return new_centers, clusters

def calculate_median_point(cluster):
    total_sum = np.zeros(2)
    for point_id, point in cluster:
        point = np.array(point)
        total_sum = np.add(total_sum, point)
    average = total_sum / len(cluster)
    return tuple(average)





def plot_data(centers, data, name):
    point_list = [point for id, point in data]
    for x,y in point_list:
        for center_id, center in centers:
            if center_id is not 1:
                g, h = center
                plt.scatter(g, h, s=300, c='red', marker='*')
            else:
                g, h = center
                plt.scatter(g, h, s=300, c='red', marker='*')
                plt.scatter(x, y, c='b')
    plt.title(name)
    plt.show()

def three_center_cost(data, centroids):
    cluster_distances = [inf for i in range(len(data))]
    for id, point in data:
        for center_id, center in centroids:
            if cluster_distances[id - 1] == inf:
                cluster_distances[id - 1] = euclidean_distance(center, point)
                continue
            else:
                distance = euclidean_distance(center, point)
                if distance < cluster_distances[id - 1]:
                    cluster_distances[id - 1] = distance
    return max(cluster_distances)

def three_means_cost(data, centroids):
    cluster_distances = [inf for i in range(len(data))]
    for id, point in data:
        for center_id, center in centroids:
            if cluster_distances[id - 1] == inf:
                cluster_distances[id - 1] = euclidean_distance(center, point)
                continue
            else:
                distance = euclidean_distance(center, point)
                if distance < cluster_distances[id - 1]:
                    cluster_distances[id - 1] = distance
    squared_distances = [data ** 2 for data in cluster_distances]
    cost = math.sqrt(sum(squared_distances) / len(data))
    return cost


def kmeanscdf(num_experiments, data, gonzalez_centroids):
    cost = [0] * num_experiments
    gonzalez_cost = three_means_cost(data, gonzalez_centroids)
    for g in range(0, num_experiments):
        centers = kmeans_plus_plus(data, 3)
        centers_cost = three_means_cost(data, centers)
        cost[g] = centers_cost

    counter = 0
    for i in range(num_experiments):
        if cost[g] == gonzalez_cost:
            counter += 1
    fraction_gonzalez = counter / num_experiments
    print(f"The fraction of gonzalez cost is: {fraction_gonzalez}")

    random_centroids, random_bin_value = np.histogram(cost, bins = num_experiments)
    cdf = np.cumsum(random_centroids)
    cdf = cdf / num_experiments

    plt.title("CDF GRAPH FOR KMEANS++ FOR 20 EXPERIMENTS")
    plt.xlabel("cost")
    plt.ylabel("probability")
    plt.plot(random_bin_value[:-1], cdf)
    plt.show()


def lloydscdf(num_experiments, data):
    cost = [0] * num_experiments
    for g in range(0, num_experiments):
        k_means_centers = kmeans_plus_plus(data, 3)
        lloyds_centers, clusters = lloyds_driver(data, k_means_centers)
        centers_cost = three_means_cost(data, lloyds_centers)
        cost[g] = centers_cost

    counter = 0
    for i in range(num_experiments):
        for center_id, k_means_center in k_means_centers:
            for lloyds_center in lloyds_centers:
                if lloyds_center == k_means_center:
                    counter += 1
    fraction_kmeans = counter / num_experiments
    print(f"The fraction of kmeans input is: {fraction_kmeans}")

    random_centroids, random_bin_value = np.histogram(cost, bins = num_experiments)
    cdf = np.cumsum(random_centroids)
    cdf = cdf / num_experiments

    plt.title("CDF GRAPH FOR Lloyds and KMEANS++ FOR 20 EXPERIMENTS")
    plt.xlabel("cost")
    plt.ylabel("probability")
    plt.plot(random_bin_value[:-1], cdf)
    plt.show()

data = standardize_c2_data()
centers = []
centers.append(data[0])
centers.append(data[1])
centers.append(data[2])
# lloyds_centers, lloyds_clusters = lloyds_driver(data, centers)
# print(lloyds_centers)
gonzalez_centers = gonzalez_algorithm(data, 3)
# lloyds_centers, lloyds_clusters = lloyds_driver(data, gonzalez_centers)
# print(lloyds_centers)
# kmeanscdf(20, data, gonzalez_centers)
# three_center_cost_gonzalez = three_center_cost(data, gonzalez_centers)
# three_means_cost_gonzalez = three_means_cost(data, gonzalez_centers)
# print(f"Gonzalez three center cost: {three_center_cost_gonzalez}")
# print(f"Gonzalez three means cost: {three_means_cost_gonzalez}")
plot_data(gonzalez_centers, data, "Gonzalez Algorithm")
kmeans_centers = kmeans_plus_plus(data, 3)
plot_data(kmeans_centers, data, "kmeans++ Algorithm")
# plot_data(lloyds_centers, data, "lloyds Algorithm")
# lloydscdf(20, data)


gonzalez_centroids = gonzalez_algorithm(data, 3)
print(f"The centroids for the gonzalez algorithm for three clusters are: {gonzalez_centroids}")
gonzalez_three_center_cost = three_center_cost(data, gonzalez_centroids)
print(f"The 3-center cost for gonzalez algorithm centroids is: {gonzalez_three_center_cost}")
gonzalez_three_means_cost = three_means_cost(data, gonzalez_centroids)
print(f"The 3-means cost for gonzalez algorithm centroids is: {gonzalez_three_means_cost}")
kmeans_plus_plus_centroids = kmeans_plus_plus(data, 3)
print(f"The centroids for the kmeans++ algorithm for three clusters are: {kmeans_plus_plus}")
kmeans_plus_plus_three_center_cost = three_center_cost(data,  kmeans_plus_plus_centroids)
print(f"The 3-center cost for kmeans++ algorithm is: {kmeans_plus_plus_three_center_cost}")
kmeans_plus_plus_three_means_cost = three_means_cost(data, kmeans_plus_plus_centroids)
print(f"The 3-means cost for kmeans++ algorithm is: {kmeans_plus_plus_three_means_cost}")
centers = []
centers.append(data[0])
centers.append(data[1])
centers.append(data[2])
lloyds_centers, clusters = lloyds_driver(data, centers)
lloyds_three_means_cost = three_means_cost(data, lloyds_centers)
print(f"The centers for lloyds algorithm are: {lloyds_centers}")
print(f"The 3-means cost for lloyds algorithm is: {lloyds_three_means_cost}")
lloyds_gonzalez_centers,clusters = lloyds_driver(data, gonzalez_centroids)
print(f"The centers for lloyds algorihm using gonzalez output are: {lloyds_gonzalez_centers}")
lloyds_gonzalez_three_means_cost = three_means_cost((data, lloyds_gonzalez_centers))
print(f"The 3-means cost for lloyds alogorithm with gonzalez centroids is: {lloyds_gonzalez_three_means_cost}")

kmeanscdf(20, data, gonzalez_centroids)

lloydscdf(20, data)