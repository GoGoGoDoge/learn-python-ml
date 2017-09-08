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

    def __init__(self, data, initial_centers = None, kmax = 20, tolerance = 0.025):
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

        # self.__pointer_data = data;
        # [Marco Revise] Instead of coordinates data points, this is gram matrix
        # self.__gram_matrix = data;
        self.__dist_matrix = data;

        self.__clusters = [];

        if (initial_centers is not None):
            self.__centers = initial_centers[:];
        else:
            # self.__centers = [ [random.random() for _ in range(len(data[0])) ] ];
            # [Marco Revise] Randomly selected center (It shall be index instead of coordinates)
            self.__centers = random.sample(range(len(data[0])), min(len(data[0]), int(kmax/2)));
            # print("debug centers: ", self.__centers);

        self.__kmax = kmax;
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
        (self.__clusters, self.__centers) = self.__improve_parameters(self.__centers);


    def get_clusters(self):
        """!
        @brief Returns list of allocated clusters, each cluster contains indexes of objects in list of data.

        @return (list) List of allocated clusters.

        @see process()
        @see get_centers()

        """

        return self.__clusters;


    def get_centers(self):
        """!
        @brief Returns list of centers for allocated clusters.

        @return (list) List of centers for allocated clusters.

        @see process()
        @see get_clusters()

        """

        return self.__centers;


    def get_cluster_encoding(self):
        """!
        @brief Returns clustering result representation type that indicate how clusters are encoded.

        @return (type_encoding) Clustering result representation.

        @see get_clusters()

        """

        return type_encoding.CLUSTER_INDEX_LIST_SEPARATION;

    def __improve_parameters(self, centers, available_indexes = None):
        """!
        @brief Performs k-means clustering in the specified region.

        @param[in] centers (list): Centers of clusters.
        @param[in] available_indexes (list): Indexes that defines which points can be used for k-means clustering, if None - then all points are used.

        @return (list) List of allocated clusters, each cluster contains indexes of objects in list of data.

        """

        changes = numpy.Inf;

        stop_condition = self.__tolerance * self.__tolerance; # Fast solution

        clusters = [];

        while (changes > stop_condition):
            clusters = self.__update_clusters(centers, available_indexes);
            clusters = [ cluster for cluster in clusters if len(cluster) > 0 ];

            updated_centers = self.__update_centers(clusters);
            print("debug clusters:", clusters)
            print("debug centers", updated_centers)
            # changes = max([euclidean_distance_sqrt(centers[index], updated_centers[index]) for index in range(len(updated_centers))]);    # Fast solution

            # [Marco Revise] to use gram matrix to Calculate distance instead of euclidean
            changes = max([self.__dist_matrix[centers[index]][updated_centers[index]] for index in range(len(updated_centers))]);    # Fast solution

            centers = updated_centers;

        return (clusters, centers);


    def __update_clusters(self, centers, available_indexes = None):
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
            bypass = range(len(self.__dist_matrix));
        else:
            bypass = available_indexes;

        clusters = [[] for i in range(len(centers))];
        for index_point in bypass:
            index_optim = -1;
            dist_optim = 0.0;

            for index in range(len(centers)):
                # dist = euclidean_distance(data[index_point], centers[index]);         # Slow solution
                # dist = euclidean_distance_sqrt(self.__pointer_data[index_point], centers[index]);      # Fast solution
                # [Marco revise] : I will use Fast solution but change the dict is represented by gram_matrix
                dist = self.__dist_matrix[index_point][centers[index]];
                if ( (dist < dist_optim) or (index is 0)):
                    index_optim = index;
                    dist_optim = dist;

            clusters[index_optim].append(index_point);

        return clusters;


    def __update_centers(self, clusters):
        """!
        @brief Updates centers of clusters in line with contained objects.

        @param[in] clusters (list): Clusters that contain indexes of objects from data.

        @return (list) Updated centers.

        [Marco revise] I have heavily revised this function
        Now within the cluster the new center point is the index where it is the shortest distance to all points
        """

        centers = [[] for i in range(len(clusters))];

        # dimension = len(self.__pointer_data[0])

        # [Marco revise] find the center point index with the minimum distance to all others
        for index in range(len(clusters)):
            min_distance = numpy.Inf; #float('inf');
            cur_cluster = clusters[index];
            for i in range(len(cur_cluster)):
                cur_distance_sum = 0.0;
                for j in range(len(cur_cluster)):
                    tmp = self.__dist_matrix[cur_cluster[i]][cur_cluster[j]]
                    tmp = tmp*tmp
                    cur_distance_sum = cur_distance_sum + tmp;
                if cur_distance_sum < min_distance :
                    min_distance = cur_distance_sum;
                    centers[index] = cur_cluster[i];


        return centers;
