"""!

@brief Cluster analysis algorithm: X-Means
@details Based on article description:
         - D.Pelleg, A.Moore. X-means: Extending K-means with Efficient Estimation of the Number of Clusters. 2000.

@authors Andrei Novikov (pyclustering@yandex.ru)
@date 2014-2017
@copyright GNU Public License

@cond GNU_PUBLIC_LICENSE
    PyClustering is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PyClustering is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
@endcond

"""


import numpy;
import random;

from enum import IntEnum;

from math import log;

from pyclustering.cluster.encoder import type_encoding;

import pyclustering.core.xmeans_wrapper as wrapper;

# from pyclustering.utils import euclidean_distance_sqrt, euclidean_distance;
# [Marco Revise] The target is to replace euclidean_distance and euclidean_distance_sqrt with gram_matrix

from pyclustering.utils import list_math_addition_number, list_math_addition, list_math_division_number;

"""
As suggested by Professor Shin, we can use kernel based version K-means.
principles.pdf

For every cluster(not necessary the final cluster), every point in the cluster shall be used to calculate the distance to judge whether current point is belonging to that cluster or not.
"""



class kmeans:
    """!
    @brief Class represents clustering algorithm X-Means.
    @details X-means clustering method starts with the assumption of having a minimum number of clusters,
             and then dynamically increases them. X-means uses specified splitting criterion to control
             the process of splitting clusters.

    Example:
    @code
        # sample for cluster analysis (represented by list)
        sample = read_sample(path_to_sample);

        # create object of X-Means algorithm that uses CCORE for processing
        # initial centers - optional parameter, if it is None, then random center will be used by the algorithm
        initial_centers = [ [0.0, 0.5] ];
        xmeans_instance = xmeans(sample, initial_centers, ccore = True);

        # run cluster analysis
        xmeans_instance.process();

        # obtain results of clustering
        clusters = xmeans_instance.get_clusters();

        # display allocated clusters
        draw_clusters(sample, clusters);
    @endcode

    """

    def __init__(self, data, initial_centers = None, k = 3, tolerance = 0.025, iterations_time = 20):
        """!
        @brief Constructor of clustering algorithm X-Means.

        @param[in] data (list): Input data that is presented as list of points (objects), each point should be represented by list or tuple.
        @param[in] initial_centers (list): Initial coordinates of centers of clusters that are represented by list: [center1, center2, ...],
                    if it is not specified then X-Means starts from the random center.
        @param[in] kmax (uint): Maximum number of clusters that can be allocated.
        @param[in] tolerance (double): Stop condition for each iteration: if maximum value of change of centers of clusters is less than tolerance than algorithm will stop processing.
        @param[in] criterion (splitting_type): Type of splitting creation.
        @param[in] ccore (bool): Defines should be CCORE (C++ pyclustering library) used instead of Python code or not.

        """

        print("Revise xmeans code to take DISTANCE matrix as input!");

        self.__gram_matrix = data;

        # All calculation is only based on cluster, there will be no center for this Kernel Solution
        self.__clusters = [];
        self.__iterations_time = iterations_time;

        # To initial we dedicated randomly selected point for clusters
        init_clusters_points =  random.sample(range(len(data[0])), min(len(data[0]), min(len(data[0]), int(k))));
        self.__last_clusters = [[] for i in range(k)];
        for i in range(len(init_clusters_points)):
            self.__last_clusters[i].append(init_clusters_points[i])

        print ("debug initial:", self.__last_clusters);
        # This is dedicated
        self.__k = k;
        self.__tolerance = tolerance;
        # self.__criterion = criterion;

        # self.__ccore = ccore;
        # [Marco Revise] This shall be forced disabled
        # self.__ccore = False

    def process(self):
        """!
        @brief Performs cluster analysis in line with rules of X-Means algorithm.

        @remark Results of clustering can be obtained using corresponding gets methods.

        @see get_clusters()
        @see get_centers()

        """

        # [Marco Revised] As this is disabled will not be called
        self.__clusters = [];
        # [Marco Note] revise improving parameters
        (self.__clusters) = self.__improve_parameters(self.__last_clusters);


    def get_clusters(self):
        """!
        @brief Returns list of allocated clusters, each cluster contains indexes of objects in list of data.

        @return (list) List of allocated clusters.

        @see process()
        @see get_centers()

        """

        return self.__clusters;

    def get_cluster_encoding(self):
        """!
        @brief Returns clustering result representation type that indicate how clusters are encoded.

        @return (type_encoding) Clustering result representation.

        @see get_clusters()

        """

        return type_encoding.CLUSTER_INDEX_LIST_SEPARATION;

    def __improve_parameters(self, last_clusters, available_indexes = None):
        """!
        @brief Performs k-means clustering in the specified region.

        @param[in] last_clusters (list): Centers of clusters.
        @param[in] available_indexes (list): Indexes that defines which points can be used for k-means clustering, if None - then all points are used.

        @return (list) List of allocated clusters, each cluster contains indexes of objects in list of data.

        """

        changes = numpy.Inf;

        stop_condition = self.__tolerance * self.__tolerance; # Fast solution

        clusters = [];

        last_record = -1;

        iters = 0

        while (last_record == -1 or changes > stop_condition):
            clusters = self.__update_clusters(last_clusters, available_indexes);
            clusters = [ cluster for cluster in clusters if len(cluster) > 0 ];

            # if len (clusters) != self.__k:
            #     print("Error debug length cluster:", len(clusters));
            #     break;

            # print("debug clusters:", clusters)
            # changes = max([euclidean_distance_sqrt(centers[index], updated_centers[index]) for index in range(len(updated_centers))]);    # Fast solution

            # [Update] Converge Condition, get the maximum value for every point in the same clusters, get the maximum value among all cluster as the change
            current_record = 0
            for current_cluster in clusters:
                tmp = 0
                for point in current_cluster:
                    tmp = max(tmp, self.__kernel_distance(point, current_cluster));
                    # print("scores:", tmp)
                current_record = max(current_record, numpy.sqrt(tmp));
            #Stop by the biggest gap
            changes = numpy.fabs(current_record - last_record);
            print("Current Iterations", iters)
            print("Changes", changes)
            last_record = current_record;
            last_clusters = clusters
            iters = iters + 1
        # print ("total iterations:", iters);
        return (clusters);

    def __kernel_distance(self, point, cluster):
        """!
        @brief Calculates the kernel distance for a point x, with a cluster, return the kernel distance
            The formula is RKHS. x is the point , Xi are points from the cluster
            1/(n^2)sum(K(Xi, Xi)) + K(Xx, Xx) - 2/n sum(K(Xi, Xx))
        @param[in] point: index of the point Xx
        @param[in] cluster: contains all index from the cluster to be calculated

        @return (double) kernel distance
        """
        dist = 0.0;
        n = len(cluster);
        gram_distance = 0.0
        point_distance = 0.0
        # print("cluster input", len(cluster))

        for i in range(len(cluster)):
            for j in range(len(cluster)):
                gram_distance += self.__gram_matrix_distance(cluster[i], cluster[j]);

        for idx in cluster:
            point_distance += self.__gram_matrix_distance(idx, point);

        # print("distance:", gram_distance, point_distance);
        res = float(gram_distance) / n / n + float(self.__gram_matrix_distance(point, point)) - 2 * float(point_distance) / n;
        # print ("debug kernel distance:", res);
        return res;

    def __gram_matrix_distance(self, x, y):
        """
        given two index, return their kernel distance by gram matrix
        """
        return self.__gram_matrix[x][y];

    def __update_clusters(self, last_clusters, available_indexes = None):
        """!
        @brief Calculates Euclidean distance to each point from the each cluster.
               Nearest points are captured by according clusters and as a result clusters are updated.

        @param[in] centers (list): Coordinates of centers of clusters that are represented by list: [center1, center2, ...].
        @param[in] available_indexes (list): Indexes that defines which points can be used from imput data, if None - then all points are used.

        @return (list) Updated clusters.

        [Marco revise] I also revise this method to manipulate the gram_matrix function
        Distance is calculated from gram_matrix
        """

        bypass = None;
        if (available_indexes is None):
            # bypass = range(len(self.__pointer_data));
            # [Marco revise]: now bypass based on the gram_matrix len and we will not specified available_indexes
            bypass = range(len(self.__gram_matrix));
        else:
            bypass = available_indexes;

        clusters = [[] for i in range(len(last_clusters))];

        for index_point in bypass:
            index_optim = -1;
            dist_optim = 0.0;

            for index in range(len(last_clusters)):
                # dist = euclidean_distance(data[index_point], centers[index]);         # Slow solution
                # dist = euclidean_distance_sqrt(self.__pointer_data[index_point], centers[index]);      # Fast solution
                # [Marco revise] : I will use Fast solution but change the dict is represented by gram_matrix
                dist = self.__kernel_distance(index_point,last_clusters[index]);
                if ( (dist < dist_optim) or (index_optim < 0)):
                    index_optim = index;
                    dist_optim = dist;

            clusters[index_optim].append(index_point);

        return clusters;
