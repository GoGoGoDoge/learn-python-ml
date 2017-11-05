import numpy;
from math import log;
"""
In this version of implementation, Guangyu and I decide to move on the choice of Calinski-Harabasz index.
This index can decide the optimal number of clusters when the index reach maximum.

As the clustering proceeding we will keep trace the value of CH index and return the maximum index

The link to the paper introduce various cluster measurement:
http://datamining.rutgers.edu/publication/internalmeasures.pdf

"""
class HAC_CH:
    """
    A class created to do clustering
    """
    def __init__(self, distance_matrix, gram_matrix):
        self.__distance_matrix = distance_matrix;
        self.__gram_matrix = gram_matrix;
        self.__total_number_data =len(gram_matrix);
        self.__last_average_distance =[[0] for i in range(len(distance_matrix))];
        # All calculation is based on index of cluster
        self.__clusters = [[i] for i in range(len(distance_matrix))];

        # Result Clusters
        self.__result_cluster = []
        self.__CH_INDEX = -1;

    def process(self):
        # Initialize
        self.__clusters = [[i] for i in range(len(self.__distance_matrix))];
        (self.__result_cluster, self.__CH_INDEX) = self.__improve_parameters(self.__clusters, self.__last_average_distance)

    def get_clusters(self):
        return self.__result_cluster;

    def get_index(self):
        return self.__CH_INDEX;

    def __improve_parameters(self, last_clusters, last_clusters_averages):
        """
        The idea is to iterate all possible number of clustering and we find the maximum
        value in the process and return the cluster when the maximum value is reached
        """
        clusters = last_clusters;
        ch_index = 0.0

        while len(last_clusters) > 2:
            # Keep aggregate the cluster and keep trace of CH index
            average_distance = last_clusters_averages;
            # Pre calculate all distance pairs
            next_min_average = -1;
            candidate_cluster_index_1 = -1;
            candidate_cluster_index_2 = -1;
            for i in range(len(last_clusters)):
                for j in range(i + 1, len(last_clusters)):
                    (avg_dist_after_comb) = self.__get_average_distance(last_clusters[i], last_clusters[j]);
                    if next_min_average == -1 or avg_dist_after_comb < next_min_average:
                        next_min_average = avg_dist_after_comb;
                        candidate_cluster_index_1 = i;
                        candidate_cluster_index_2 = j;
            # Consolidate the result into last_cluster
            tmp_clusters = [];
            tmp_clusters.append(last_clusters[candidate_cluster_index_1] + last_clusters[candidate_cluster_index_2]);
            tmp_averages = [next_min_average];
            for i in range(len(last_clusters)):
                if i == candidate_cluster_index_1 or i == candidate_cluster_index_2:
                    continue; # Already in the clusters
                tmp_clusters.append(last_clusters[i]);
                tmp_averages.append(last_clusters_averages[i]);
            last_clusters = tmp_clusters;
            last_clusters_averages = tmp_averages;
            # Calculate the CH index score
            ch_index = self.__calculate_Calinski_Harabasz_index(last_clusters);
            print("number of clusters", len(last_clusters), "index:", ch_index, self.__CH_INDEX)
            if ch_index > self.__CH_INDEX:
                self.__CH_INDEX = ch_index;
                self.__result_cluster = list(last_clusters);
        # print("result len:", len(self.__result_cluster))
        return (self.__result_cluster, self.__CH_INDEX);

    def __get_average_distance(self, cluster_a, cluster_b):
        """
        Calculate the average distance after combination of two clusters
        """
        result_cluster = cluster_a + cluster_b;
        count = 0;
        distance = 0.0;
        for i in range(len(result_cluster)):
            for j in range(i + 1, len(result_cluster)):
                count += 1;
                distance += self.__distance_matrix[result_cluster[i]][result_cluster[j]];
        return distance / count;

    def __distance_sqrt(self, index_object, cluster):
        n = len(cluster)
        ans = 0.0
        for i in cluster:
            ans += self.__distance_matrix[index_object][i]
        return ans / n / n

    def __calculate_Calinski_Harabasz_index(self, clusters):
        # Calculate the CH index
        number_of_cluster = len(clusters);
        numerator = 0.0
        denominator = 0.0
        for i in range(number_of_cluster):
            # Build a set to contains current cluster element
            cur_cluster = set(clusters[i]);
            numerator += len(cur_cluster) * self.__cluster_centroid_to_dataset_centroid(cur_cluster)
        numerator = numerator / (number_of_cluster - 1)

        for i in range(number_of_cluster):
            # Build a set to contains current cluster element
            cur_cluster = set(clusters[i]);
            denominator += self.__cluster_avg_square_distance(cur_cluster);
        denominator = denominator / (len(self.__gram_matrix) - number_of_cluster);
        # print(clusters, numerator, denominator)
        return numerator / denominator;

    def __cluster_avg_square_distance(self, cur_cluster):
        res = 0.0
        center_i = 0.0
        for i in cur_cluster:
            for j in cur_cluster:
                center_i += self.__gram_matrix[i][j]
        center_i = center_i / len(cur_cluster) / len(cur_cluster)

        for x in cur_cluster:
            A = self.__gram_matrix[x][x];
            B = 0.0
            for i in cur_cluster:
                if i == x:
                    continue;
                B += self.__gram_matrix[x][i];
            res += (A - B * 2 / len(cur_cluster) + center_i)

        return res;

    def __cluster_centroid_to_dataset_centroid(self, cur_cluster):
        other_points = set(range(len(self.__gram_matrix))) - cur_cluster;
        n = len(cur_cluster)
        m = len(self.__gram_matrix)
        A = 0.0
        B = 0.0
        C = 0.0
        for i in cur_cluster:
            for j in cur_cluster:
                A += self.__gram_matrix[i][j];
        A = A * (m - n) * (m - n);

        for i in cur_cluster:
            for j in other_points:
                B += self.__gram_matrix[i][j];
        B = 2 * (m - n) * n * B;

        for i in other_points:
            for j in other_points:
                C += self.__gram_matrix[i][j];
        C = n * n * C;
        return (A - B + C) / n / n / m / m;

if __name__=='__main__':
    distance_matrix = [[0,1,2,3,4],[1,0,4,5,2],[2,4,0,2,3],[3,5,2,0,4],[1,2,3,4,0]];
    gram_matrix = [[1,2,3,4,5],[2,3,4,5,6],[4,5,6,7,8],[9,8,7,6,5],[8,7,6,5,4]];
    a = [1.0,2.0]
    b = [6.0,7.0]
    print(max(a))
    # Test small thredshold
    # hac = HAC(distance_matrix, 3, 0.1);
    # Test Large Thredshold
    hac = HAC_CH(distance_matrix, gram_matrix);
    hac.process();
    print(hac.get_clusters());
