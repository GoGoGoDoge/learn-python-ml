import numpy;

class HAC:
    """
    A class created to do clustering
    """
    def __init__(self, distance_matrix, k = 3, cut_off_distance = 100, iterations_time = 20):
        self.__distance_matrix = distance_matrix;

        # All calculation is based on index of cluster
        self.__clusters = [[i] for i in range(len(distance_matrix))];
        self.__average_distance = [[0] for i in range(len(distance_matrix))];
        self.__iterations_time = iterations_time;
        # Number of cluster
        self.__k = k;
        self.__cut_off_distance = min(max(max(distance_matrix)), cut_off_distance);

    def process(self):
        # Initialize
        self.__clusters = [[i] for i in range(len(distance_matrix))];
        self.__average_distance = [0 for i in range(len(distance_matrix))];
        (self.__clusters, self.__average_distance) = self.__improve_parameters(self.__clusters, self.__average_distance)

    def get_clusters(self):
        return self.__clusters;

    def __improve_parameters(self, last_clusters, last_clusters_averages):
        """
        The idea is to keeps building tree until maximum of average distance exceed
        cut off distance
        """
        clusters = last_clusters;
        average_distance = last_clusters_averages;
        # print("debug avg:", last_clusters_averages)
        # print("debug cutoff:", self.__cut_off_distance)
        while (max(last_clusters_averages) < self.__cut_off_distance) and len(last_clusters) > 1:
            # Now we can safely update cluster
            clusters = last_clusters;
            average_distance = last_clusters_averages;
            # Pre calculate all distance pairs
            next_min_average = -1;
            candidate_cluster_index_1 = -1;
            candidate_cluster_index_2 = -1;
            for i in range(len(last_clusters)):
                for j in range(i + 1, len(last_clusters)):
                    (avg_dist_after_comb) = self.__get_average_distance(last_clusters[i], last_clusters[j]);
                    # print("debug", avg_dist_after_comb, i, j)
                    if next_min_average == -1 or avg_dist_after_comb < next_min_average:
                        next_min_average = avg_dist_after_comb;
                        candidate_cluster_index_1 = i;
                        candidate_cluster_index_2 = j;
            # Consolidate the result in to last_cluster
            tmp_clusters = [];
            tmp_clusters.append(last_clusters[candidate_cluster_index_1] + last_clusters[candidate_cluster_index_2]);
            tmp_averages = [next_min_average];
            for i in range(len(last_clusters)):
                if i == candidate_cluster_index_1 or i == candidate_cluster_index_2:
                    continue;
                tmp_clusters.append(last_clusters[i]);
                tmp_averages.append(last_clusters_averages[i]);
            last_clusters = tmp_clusters;
            last_clusters_averages = tmp_averages;
            # print(last_clusters_averages)
            # print("cluster:", last_clusters)
            # input("enter to continue...")


        return (clusters, average_distance);

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

if __name__=='__main__':
    distance_matrix = [[0,1,2,3,4],[1,0,4,5,2],[2,4,0,2,3],[3,5,2,0,4],[1,2,3,4,0]];
    a = [1.0,2.0]
    b = [6.0,7.0]
    print(max(a))
    # Test small thredshold
    # hac = HAC(distance_matrix, 3, 0.1);
    # Test Large Thredshold
    hac = HAC(distance_matrix, 3, 10);
    hac.process();
    print(hac.get_clusters());
