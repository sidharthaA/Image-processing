# NAME(s):
# Homework 4 github repo: https://classroom.github.com/a/tvgh_mw0
# NOTE: For all of the code below, you are strongly encouraged to 
# automate and reuse as much as possible. Simply cutting and pasting
# the code for one part and editing the variables of interest will
# result in a loss of points. Instead, write new subroutines whenever
# possible. 
import os
import numpy as np
import matplotlib
from numpy.linalg import eig
from pgm_reader import Reader
from networkx.linalg.graphmatrix import adjacency_matrix
import matplotlib.pyplot as plt
import numpy as np
import math
import networkx as nx
import pdb
from scipy.spatial import distance as dist

import matplotlib.pyplot as plt

"""
1. Download the CMU face image dataset
	https://archive.ics.uci.edu/ml/machine-learning-databases/faces-mld/faces.tar.gz
	DO NOT ADD THIS DATASET TO YOUR GITHUB REPO, or if you do, KEEP IT LOCAL by 
	adding the parent directory to your .gitignore file.
2. [5 points] Load all of the 60x64 images into memory and "flatten" the pixel data by 
	reshaping it from an array to a vector. Call the matrix with all image data in
	it (one row per item, one column per pixel) X. Numpy will be useful for this.
"""
import os

images = []
names = []
IMG_DIR = "faces"
reader = Reader()

for root, dirs, files in os.walk(IMG_DIR):
    for filename in files:
        if filename.endswith(".pgm"):
            image = reader.read_pgm(os.path.join(root, filename))
            if image.shape == (60, 64):
                # print(filename)
                names.append(filename)
                images.append(reader.read_pgm(os.path.join(root, filename)))
# You need to do the flattening somewhere around here.

data = [x.reshape(3840) for x in images]
data = np.array(data)

"""
3. [5 points] Construct a list, where each item is a pair of image names, and the second 
	image in the pair corresponds to the image that is closest to the first name, with respect 
	to euclidean distance. Sound familiar? Note that multiple pairs may have the same second object, 
	but the first object should be unique (i.e., only occur in one pair). Call the list
	fullpairs. 
"""


def distance(u, v):
    """
    Computes the euclidean distance between 2 normalized rows
    :param u: Row 1
    :param v: Row 2
    :return: Distance between u and v
    """

    return np.sqrt(np.sum((u-v)**2))


def compute_list(image_data):
    list_of_pairs = []
    for i in range(len(image_data)):
        distance_dict = {}
        for j in range(len(image_data)):
            if i == j:
                continue
            distance_dict[j] = distance(image_data[i], image_data[j])
        min_val_dict = min(distance_dict, key=distance_dict.get)
        list_of_pairs.append((i, min_val_dict))
        print('Minimum Distance is from Index:', min_val_dict)
        print('Min distance:', distance_dict[min_val_dict])
    return list_of_pairs


# fullpairs = compute_list(data)

"""
4. [5 points] Create a networkx object from your list, where there is one node for each dataset 
	object, and one edge for each pair in your list. Draw your graph, save the drawing as 
	"full_graph.svg," and place in your README.md file
	
"""
import networkx as nx


def compute_graph(pair_list):
    """
    Using the pair list, compute the graph
    :param pair_list:
    :return: Graph
    """
    G = nx.Graph()
    for i in range(len(pair_list)):
        G.add_node(i)
        G.add_edge(pair_list[i][0], pair_list[i][1])
    return G


# G = compute_graph(fullpairs)
# plt.figure(figsize=(50, 50))
# nx.draw_spring(G, node_size=1000, width=5, with_labels=True, font_size=20)
# plt.draw()
# plt.savefig("full_graph.svg")

"""
5. [10 points] Perform PCA over all of the images. You may use code from the 
	Google collaboratory Jupyter notebook that Cyril demonstrated in the Week 7 
	recitation. Extract the 32 top principal components into a matrix called 
	top32.
"""


def pca_dims(data):
    """
    Returns the component after performing PCA
    :param data:
    :return: component
    """
    face_averager = np.ones([data.shape[0], data.shape[0]]) / data.shape[0]
    mean_face = face_averager @ data
    centered_faces = data - mean_face
    face_covariance = centered_faces.T @ centered_faces
    (eigenvalues, components) = eig(face_covariance)
    return components


comp = pca_dims(data)
# top32 = comp[:, :32]

"""
6. [5 points] Project each image into the top 32 principle components, so that each image can 
	now be represented by a 32 dimension vector, corresponding the the coefficient of 
	the image into each component. Place all of these projections in a single matrix 
	called X32, where each row represents one image.
"""


# NDIMS = 32


def compress(im, component, dims):
    return im.T @ component[:, :dims]


def decompress(d, component, dims):
    return component[:, :dims] @ d


def project_image(data, component, dims):
    """

    :param data:
    :param component:
    :param dims:
    :return:
    """
    X32 = []
    for i in data:
        d = compress(i, component, dims)
        dec = decompress(d, component, dims)
        X32.append(dec)
    return X32


# X32 = project_image(data, top32, 32)

"""
7. [10 points] Repeat steps 3-4, but with the data from X32 instead of X. Call the resulting
	list pcapairs and the image "pcagraph.svg" (also post this image in the READEME.md
	file)
"""
# pcapairs = compute_list(X32)
# G_32 = compute_graph(pcapairs)
# plt.figure(figsize=(50, 50))
# nx.draw_spring(G, node_size=1000, width=5, with_labels=True, font_size=20)
# plt.draw()
# plt.savefig("pcagraph.svg")
"""
8. [3 points] Construct a list of all pairs that appear in fullpairs and pcapairs.
	Call this list bothpairs. Give len(bothpairs) in README.md.
	[3 ponts] Construct a list of all pairs that appear in fullpairs but not pcapairs.
	Call this list fullonlypairs. Give len(bothpairs) in README.md.
	[4 points] Construct a list of all pairs that appear in pcapairs but not fullpairs
	Call this list pcaonlypairs. Give len(bothpairs) in README.md.
"""
# set_fullpairs = set(fullpairs)
# set_pcapairs = set(pcapairs)
# bothpairs = set_fullpairs.intersection(set_pcapairs)
# print('Len of Both pairs: ' , len(bothpairs))
# fullonlypairs = set_fullpairs.difference(set_pcapairs)
# print('Length of fullOnly Pairs: ' , len(fullonlypairs))
# pcaonlypairs = set_pcapairs.difference(set_fullpairs)
# print('Length of fullOnly Pairs: ' , len(pcaonlypairs))

"""
9. [10 points] Choose two images of different people from X32 and reconstruct them by summing together
	the coefficients of the principal components, put both images in your README.md file, along
	with the original images.   
"""
#
# im1 = X32[43]
# im2 = X32[98]
# plt.imshow(im1.astype(float).reshape([60, 64]), cmap=plt.cm.gray)
# plt.show()
# plt.savefig('projected_image_32_components.png')
# plt.imshow(im2.astype(float).reshape([60, 64]), cmap=plt.cm.gray)
# plt.show()
# plt.savefig('projected_image_2.png')

"""
One crude way to know whether you have enough dimensions in PCA 
is to construct a model of the data and test how much 
the model changes after PCA. One such model is the sort of graph you
constructed earlier in this assignment. So for this experiment, for
all values of i between 32 and 1024, please...
"""

# A = nx.adjacency_matrix(G)
# distance_list = []
# for i in range(32, 1024):
#     # 10. [3 points] project your images into the top i principal components
#     # X_components=pca_dims(data,i)
#     print("Components:",i)
#     X_images_i = project_image(data, comp, i)
#     # 11. [3 points] construct the graph from #4 but using the data from #10
#     pair_list = compute_list(X_images_i)
#     G_i = compute_graph(pair_list)
#     # 12. [4 points] extract the adjacency matrix from the graph from #11 and the graph from #4
#     #     (You shoul compute the adjacencey matrix for #4 above the loop, since you only need to
#     #		to compute it once)
#     A_i = nx.adjacency_matrix(G_i)
#     # 13. Compute the differences between thes two matrices (done below)
#     distance_i = np.sum(abs(A - A_i))
#     distance_list.append(distance_i)
# # [5 points] Create a bar chart where i (# of PCA dimensions) are on the x-axis
# # 	and the distances between the current graph and the graph from #4 are the
# #	y-axis. Show this graph in your README.md document, with proper labels.
# x_axis = range(32, 1024)
# plt.bar(x_axis, distance_list)
# plt.title('Distance Bar Chart')
# plt.xlabel('Number of Dimensions')
# plt.ylabel('Distance')
# plt.show()

""" 
14. [5 points] Now choose one of the grayscale images from Homework 2
	(or collect another image from somewhere). If the number of rows 
	and/or columns is not divisible by 12, remove rows and/or columns
	from the image until both are divisible by 12. Display the original
	image in your README.md file
"""


def remove_row(image):
    while (image.shape[0] % 12 != 0):
        image = image[1:, :]
    return image


image_unmodified = plt.imread('image1_gray.jpeg')
image_modified = remove_row(image_unmodified)
plt.imshow(image_modified)
plt.savefig('gray_image_removed.png')
"""
15. [5 points] Divide your image into as many 12x12 pixel nonoverlapping minimages as is
	needed to tile the entire image. Create a new array,
	call it Ximage, where each row is a minimage and the columns are the 
	pixels of the image, reshaped into a row vector. Essentially, 
	Ximage is just like the variable X above, except each entry is a 
	minimage, instead of an original image.
"""


def mini_image(im):
    Ximage = []
    for i in range(0, im.shape[0], 12):
        for j in range(0, im.shape[1], 12):
            temp = im[i:i + 12, j:j + 12]
            temp = temp.flatten()
            Ximage.append(temp.reshape(144))
    return np.array(Ximage)


Ximage = mini_image(image_modified)

"""
16. [5 points] Perform PCA on your minimage data. Call the resulting components
	image_comps
"""
image_comps = pca_dims(Ximage)
"""
17. [5 points] Project your minimages into the first 32 principal components, then
	reconstruct each minimage from these 32 components, THEN reconstruct the image
	from the reconstructed minimages, (by reshaping them and restitching them.) Show
	this image in your README.md file.
"""
projected_minimages = project_image(Ximage, image_comps, 32)


def reconstruction(projected_image):
    X_reconstructed = []
    for i in projected_image:
        temp = i.reshape(12, 12)
        X_reconstructed.append(temp)
    return np.array(X_reconstructed)


reconstructed_32 = reconstruction(projected_minimages)
plt.imshow(reconstructed_32.reshape(image_modified.shape[0], image_modified.shape[1]), cmap=plt.cm.gray)
plt.savefig('reconstructed.png')

"""
18.	[5 points] Do the same but with the first 4 principal components.
"""
projected_minimages_1 = project_image(Ximage, image_comps, 4)
reconstructed_4 = reconstruction(projected_minimages_1)
plt.imshow(reconstructed_4.reshape(image_modified.shape[0], image_modified.shape[1]), cmap=plt.cm.gray)
plt.savefig('reconstructed_1.png')
