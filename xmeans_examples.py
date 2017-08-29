"""!

@brief Examples of usage and demonstration of abilities of X-Means algorithm in cluster analysis.

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

from pyclustering.samples.definitions import SIMPLE_SAMPLES, FCPS_SAMPLES;

from pyclustering.cluster import cluster_visualizer;
# from pyclustering.cluster.xmeans import xmeans, splitting_type;
from xmeans import xmeans, splitting_type
from pyclustering.utils import read_sample, timedcall;


def template_clustering(start_centers, path, tolerance = 0.025, criterion = splitting_type.BAYESIAN_INFORMATION_CRITERION, ccore = False):
    sample = read_sample(path);

    xmeans_instance = xmeans(sample, start_centers, 20, tolerance, criterion, ccore);
    (ticks, result) = timedcall(xmeans_instance.process);

    clusters = xmeans_instance.get_clusters();
    centers = xmeans_instance.get_centers();

    criterion_string = "UNKNOWN";
    if (criterion == splitting_type.BAYESIAN_INFORMATION_CRITERION): criterion_string = "BAYESIAN INFORMATION CRITERION";
    elif (criterion == splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH): criterion_string = "MINIMUM NOISELESS DESCRIPTION_LENGTH";

    print("Sample: ", path, "\nInitial centers: '", (start_centers is not None), "', Execution time: '", ticks, "', Number of clusters:", len(clusters), ",", criterion_string, "\n");

    visualizer = cluster_visualizer();
    visualizer.set_canvas_title(criterion_string);
    visualizer.append_clusters(clusters, sample);
    visualizer.append_cluster(centers, None, marker = '*');
    visualizer.show();


def cluster_sample1():
    "Start with wrong number of clusters."
    start_centers = [[3.7, 5.5]];
    template_clustering(start_centers, SIMPLE_SAMPLES.SAMPLE_SIMPLE1, criterion = splitting_type.BAYESIAN_INFORMATION_CRITERION);
    template_clustering(start_centers, SIMPLE_SAMPLES.SAMPLE_SIMPLE1, criterion = splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH);

def cluster_sample1_without_initial_centers():
    template_clustering(None, SIMPLE_SAMPLES.SAMPLE_SIMPLE1, criterion = splitting_type.BAYESIAN_INFORMATION_CRITERION);
    template_clustering(None, SIMPLE_SAMPLES.SAMPLE_SIMPLE1, criterion = splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH);

def cluster_sample2():
    "Start with wrong number of clusters."
    start_centers = [[3.5, 4.8], [2.6, 2.5]];
    template_clustering(start_centers, SIMPLE_SAMPLES.SAMPLE_SIMPLE2, criterion = splitting_type.BAYESIAN_INFORMATION_CRITERION);
    template_clustering(start_centers, SIMPLE_SAMPLES.SAMPLE_SIMPLE2, criterion = splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH);

def cluster_sample2_without_initial_centers():
    template_clustering(None, SIMPLE_SAMPLES.SAMPLE_SIMPLE2, criterion = splitting_type.BAYESIAN_INFORMATION_CRITERION);
    template_clustering(None, SIMPLE_SAMPLES.SAMPLE_SIMPLE2, criterion = splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH);

def cluster_sample3():
    "Start with wrong number of clusters."
    start_centers = [[0.2, 0.1], [4.0, 1.0]];
    template_clustering(start_centers, SIMPLE_SAMPLES.SAMPLE_SIMPLE3, criterion = splitting_type.BAYESIAN_INFORMATION_CRITERION);
    template_clustering(start_centers, SIMPLE_SAMPLES.SAMPLE_SIMPLE3, criterion = splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH);

def cluster_sample3_without_initial_centers():
    template_clustering(None, SIMPLE_SAMPLES.SAMPLE_SIMPLE3, criterion = splitting_type.BAYESIAN_INFORMATION_CRITERION);
    template_clustering(None, SIMPLE_SAMPLES.SAMPLE_SIMPLE3, criterion = splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH);

def cluster_sample4():
    start_centers = [[1.5, 0.0], [1.5, 2.0], [1.5, 4.0], [1.5, 6.0], [1.5, 8.0]];
    template_clustering(start_centers, SIMPLE_SAMPLES.SAMPLE_SIMPLE4, criterion = splitting_type.BAYESIAN_INFORMATION_CRITERION);
    template_clustering(start_centers, SIMPLE_SAMPLES.SAMPLE_SIMPLE4, criterion = splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH);

def cluster_sample4_without_initial_centers():
    template_clustering(None, SIMPLE_SAMPLES.SAMPLE_SIMPLE4, criterion = splitting_type.BAYESIAN_INFORMATION_CRITERION);
    template_clustering(None, SIMPLE_SAMPLES.SAMPLE_SIMPLE4, criterion = splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH);

def cluster_sample5():
    "Start with wrong number of clusters."
    start_centers = [[0.0, 1.0], [0.0, 0.0]];
    template_clustering(start_centers, SIMPLE_SAMPLES.SAMPLE_SIMPLE5, criterion = splitting_type.BAYESIAN_INFORMATION_CRITERION);
    template_clustering(start_centers, SIMPLE_SAMPLES.SAMPLE_SIMPLE5, criterion = splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH);

def cluster_sample5_without_initial_centers():
    template_clustering(None, SIMPLE_SAMPLES.SAMPLE_SIMPLE5, criterion = splitting_type.BAYESIAN_INFORMATION_CRITERION);
    template_clustering(None, SIMPLE_SAMPLES.SAMPLE_SIMPLE5, criterion = splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH);

def cluster_elongate():
    "Not so applicable for this sample"
    start_centers = [[1.0, 4.5], [3.1, 2.7]];
    template_clustering(start_centers, SIMPLE_SAMPLES.SAMPLE_ELONGATE, criterion = splitting_type.BAYESIAN_INFORMATION_CRITERION);
    template_clustering(start_centers, SIMPLE_SAMPLES.SAMPLE_ELONGATE, criterion = splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH);

def cluster_lsun():
    "Not so applicable for this sample"
    start_centers = [[1.0, 3.5], [2.0, 0.5], [3.0, 3.0]];
    template_clustering(start_centers, FCPS_SAMPLES.SAMPLE_LSUN, criterion = splitting_type.BAYESIAN_INFORMATION_CRITERION);
    template_clustering(start_centers, FCPS_SAMPLES.SAMPLE_LSUN, criterion = splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH);

def cluster_target():
    "Not so applicable for this sample"
    start_centers = [[0.2, 0.2], [0.0, -2.0], [3.0, -3.0], [3.0, 3.0], [-3.0, 3.0], [-3.0, -3.0]];
    template_clustering(start_centers, FCPS_SAMPLES.SAMPLE_TARGET, criterion = splitting_type.BAYESIAN_INFORMATION_CRITERION);
    template_clustering(start_centers, FCPS_SAMPLES.SAMPLE_TARGET, criterion = splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH);

def cluster_two_diamonds():
    "Start with wrong number of clusters."
    start_centers = [[0.8, 0.2]];
    template_clustering(start_centers, FCPS_SAMPLES.SAMPLE_TWO_DIAMONDS, criterion = splitting_type.BAYESIAN_INFORMATION_CRITERION);
    template_clustering(start_centers, FCPS_SAMPLES.SAMPLE_TWO_DIAMONDS, criterion = splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH);

def cluster_two_diamonds_without_initial_centers():
    template_clustering(None, FCPS_SAMPLES.SAMPLE_TWO_DIAMONDS, criterion = splitting_type.BAYESIAN_INFORMATION_CRITERION);
    template_clustering(None, FCPS_SAMPLES.SAMPLE_TWO_DIAMONDS, criterion = splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH);

def cluster_wing_nut():
    start_centers = [[-1.5, 1.5], [1.5, 1.5]];
    template_clustering(start_centers, FCPS_SAMPLES.SAMPLE_WING_NUT, criterion = splitting_type.BAYESIAN_INFORMATION_CRITERION);
    template_clustering(start_centers, FCPS_SAMPLES.SAMPLE_WING_NUT, criterion = splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH);

def cluster_chainlink():
    start_centers = [[1.1, -1.7, 1.1], [-1.4, 2.5, -1.2]];
    template_clustering(start_centers, FCPS_SAMPLES.SAMPLE_CHAINLINK, criterion = splitting_type.BAYESIAN_INFORMATION_CRITERION);
    template_clustering(start_centers, FCPS_SAMPLES.SAMPLE_CHAINLINK, criterion = splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH);

def cluster_hepta():
    "Start with wrong number of clusters."
    start_centers = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, -3.0, 0.0], [0.0, 0.0, 2.5]];
    template_clustering(start_centers, FCPS_SAMPLES.SAMPLE_HEPTA, criterion = splitting_type.BAYESIAN_INFORMATION_CRITERION);
    template_clustering(start_centers, FCPS_SAMPLES.SAMPLE_HEPTA, criterion = splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH);

def cluster_tetra():
    start_centers = [[1, 0, 0], [0, 1, 0], [0, -1, 0], [-1, 0, 0]];
    template_clustering(start_centers, FCPS_SAMPLES.SAMPLE_TETRA, criterion = splitting_type.BAYESIAN_INFORMATION_CRITERION);
    template_clustering(start_centers, FCPS_SAMPLES.SAMPLE_TETRA, criterion = splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH);

def experiment_execution_time(ccore_flag = False):
    template_clustering([[3.7, 5.5]], SIMPLE_SAMPLES.SAMPLE_SIMPLE1, ccore = ccore_flag);
    template_clustering([[3.5, 4.8], [2.6, 2.5]], SIMPLE_SAMPLES.SAMPLE_SIMPLE2, ccore = ccore_flag);
    template_clustering([[0.2, 0.1], [4.0, 1.0]], SIMPLE_SAMPLES.SAMPLE_SIMPLE3, ccore = ccore_flag);
    template_clustering([[1.5, 0.0], [1.5, 2.0], [1.5, 4.0], [1.5, 6.0], [1.5, 8.0]], SIMPLE_SAMPLES.SAMPLE_SIMPLE4, ccore = ccore_flag);
    template_clustering([[0.0, 1.0], [0.0, 0.0]], SIMPLE_SAMPLES.SAMPLE_SIMPLE5, ccore = ccore_flag);
    template_clustering([[1.0, 4.5], [3.1, 2.7]], SIMPLE_SAMPLES.SAMPLE_ELONGATE, ccore = ccore_flag);
    template_clustering([[1.0, 3.5], [2.0, 0.5], [3.0, 3.0]], FCPS_SAMPLES.SAMPLE_LSUN, ccore = ccore_flag);
    template_clustering([[0.2, 0.2], [0.0, -2.0], [3.0, -3.0], [3.0, 3.0], [-3.0, 3.0], [-3.0, -3.0]], FCPS_SAMPLES.SAMPLE_TARGET, ccore = ccore_flag);
    template_clustering([[0.8, 0.2]], FCPS_SAMPLES.SAMPLE_TWO_DIAMONDS, ccore = ccore_flag);
    template_clustering([[-1.5, 1.5], [1.5, 1.5]], FCPS_SAMPLES.SAMPLE_WING_NUT, ccore = ccore_flag);
    template_clustering([[1.1, -1.7, 1.1], [-1.4, 2.5, -1.2]], FCPS_SAMPLES.SAMPLE_CHAINLINK, ccore = ccore_flag);
    template_clustering([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, -3.0, 0.0], [0.0, 0.0, 2.5]], FCPS_SAMPLES.SAMPLE_HEPTA, ccore = ccore_flag)
    template_clustering([[1, 0, 0], [0, 1, 0], [0, -1, 0], [-1, 0, 0]], FCPS_SAMPLES.SAMPLE_TETRA, ccore = ccore_flag);
    template_clustering([[1, 0, 0], [0, 1, 0], [0, -1, 0], [-1, 0, 0]], FCPS_SAMPLES.SAMPLE_ATOM);

def cluster_example_with_gram_matrix_without_initial_centers():
    # gram_matrix = [[0.0,1.0,1.0,13.0,25.0],[1.0,0.0,2.0,8.0,120.0],[1.0,2.0,0.0,10.0,18.0],[13.0,8.0,10.0,0.0,4.0],[25.0,120.0,18.0,4.0,0.0]];
    # gram_matrix = [[0.0,1.0,1.0,13.0,25.0],[1.0,0.0,2.0,8.0,2.0],[1.0,2.0,0.0,10.0,18.0],[13.0,8.0,10.0,0.0,4.0],[25.0,2.0,18.0,4.0,0.0]];
    gram_matrix = [[42.529316490802, 42.717195285258, 41.805586172258, 42.978501640665996, 44.947537609731, 67.11094076958, 63.678011905026004, 66.310339525968, 66.346686765112, 64.240854664512], [42.717195285258, 42.98070563613, 41.977454326098, 43.247573480802, 45.215452860866996, 67.865308959276, 64.438999860762, 66.960868291632, 67.121517748872, 64.925375521176], [41.805586172258, 41.977454326098, 41.096366664404, 42.23355903671, 44.17077567328, 65.890462310272, 62.512220529489994, 65.12066685559199, 65.135130172156, 63.079009970274], [42.978501640665996, 43.247573480802, 42.23355903671, 43.516306488106, 45.495703276581, 68.304626633076, 64.858508648106, 67.389357355104, 67.55754427144, 65.343690739164], [44.947537609731, 45.215452860866996, 44.17077567328, 45.495703276581, 47.567561523780995, 71.351422915636, 67.743495902731, 70.412301958164, 70.56581610466999, 68.26513874790899], [67.11094076958, 67.865308959276, 65.890462310272, 68.304626633076, 71.351422915636, 108.703568183632, 103.419789987916, 106.82622402504, 107.64351116280801, 103.824638625396], [63.678011905026004, 64.438999860762, 62.512220529489994, 64.858508648106, 67.743495902731, 103.419789987916, 98.419459482706, 101.577857553744, 102.42836573036, 98.755881560544], [66.310339525968, 66.960868291632, 65.12066685559199, 67.389357355104, 70.412301958164, 106.82622402504, 101.577857553744, 105.098448426912, 105.748587762288, 102.077980935768], [66.346686765112, 67.121517748872, 65.135130172156, 67.55754427144, 70.56581610466999, 107.64351116280801, 102.42836573036, 105.748587762288, 106.60478522218399, 102.797937856836], [64.240854664512, 64.925375521176, 63.079009970274, 65.343690739164, 68.26513874790899, 103.824638625396, 98.755881560544, 102.077980935768, 102.797937856836, 99.18308837133]]

    # template_clustering(start_centers, path, tolerance = 0.025, criterion = splitting_type.BAYESIAN_INFORMATION_CRITERION, ccore = False):
    # xmeans_instance = xmeans(gram_matrix, None, 20, 0.025, splitting_type.BAYESIAN_INFORMATION_CRITERION, False);
    xmeans_instance = xmeans(gram_matrix, None, 10, 0.025, splitting_type.BAYESIAN_INFORMATION_CRITERION, False);
    (ticks, result) = timedcall(xmeans_instance.process);

    clusters = xmeans_instance.get_clusters();
    centers = xmeans_instance.get_centers();

    print ("finish...")
    print ("centers:", centers)
    print ("clusters:", clusters)

# cluster_sample1();
# cluster_sample2();
# cluster_sample3();
# cluster_sample4();
# cluster_sample5();
# cluster_elongate();
# cluster_lsun();
# cluster_target();
# cluster_two_diamonds();
# cluster_wing_nut();
# cluster_chainlink();
# cluster_hepta();
# cluster_tetra();
#
# cluster_sample1_without_initial_centers(); # This is suitalbe to our case
# cluster_sample2_without_initial_centers();
# cluster_sample3_without_initial_centers();
# cluster_sample4_without_initial_centers();
# cluster_sample5_without_initial_centers();
# cluster_two_diamonds_without_initial_centers();
#
# experiment_execution_time(False);   # Python code
# experiment_execution_time(True);    # C++ code + Python env.

cluster_example_with_gram_matrix_without_initial_centers(); # Marco revise example to explain how to use 
