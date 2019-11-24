import numpy as np 
import os

import cv2
import PIL.Image as Image

from scipy.signal import savgol_filter

from itertools import product
from string import ascii_lowercase

import sys
sys.path.append("../../IAD-Generator/iad-generation/")
from feature_rank_utils import get_top_n_feature_indexes
from csv_utils import read_csv

def preprocess(iad):
	return iad[:, 3:-3]

def find_start_stop(feature, iad):

	# smooth the IAD expression
	feature = savgol_filter(feature, 25, 3)

	# threshold the expression we are looking at
	avg_val = np.mean(feature)
	std_dev = np.std(feature)
	above_threshold = np.argwhere(feature > avg_val+std_dev).reshape(-1)

	# identify the start and stop times of the events
	start_stop_times = []
	if(len(above_threshold) != 0):
		start = above_threshold[0]
		for i in range(1, len(above_threshold)):

			if( above_threshold[i-1]+1 < above_threshold[i] ):
				start_stop_times.append([start, above_threshold[i-1]+1])
				start = above_threshold[i]

		start_stop_times.append([start, above_threshold[len(above_threshold)-1]+1])

	return start_stop_times


def postprocess(sparse_map):
	'''
	Take a sparse map and offset it by three to account for the trimmed IAD.
	Remove any start stop times that are shorter than 3 time instances
	'''

	for f, feat in enumerate(sparse_map):
	
		remove_pairs = []
		for p, pair in enumerate(feat):

			if pair[1]-pair[0] > 3:

				#offset accoridng to beginning and end trimming
				pair[0] += 3 
				pair[1] += 3 

			else:
				remove_pairs.append(pair)

		# remove pairs that are smaller than 3 in length
		for pair in remove_pairs:
			feat.remove(pair)
		
	return sparse_map


def sparsify_iad(iad_filename, pruning_indexes, layer, name="output.txt"):
	'''
	Convert an IAD into a sparse map that indicates the start and stop times 
	when a feature is expressed. Write the map to a file.
	'''

	# open the IAD and prune any un-wanted features
	iad = np.load(iad_filename)["data"]

	idx = pruning_indexes[layer]
	iad = iad[idx]

	# determine start_stop_times for each feature in the IAD. Apply
	# any pre or post processing dteps to clean up the IAD and sparse map
	iad = preprocess(iad)
	sparse_map = []
	for feature in iad:
		sparse_map.append(find_start_stop(feature, iad))
	sparse_map = postprocess(sparse_map)

	# write start_stop_times to file. Each feature is given a unique 3 char  
	# alphabetical code to identify it. This should cover up to 17K unique
	# features
	ofile = open(name, 'w')
	action_labels = [''.join(i) for i in product(ascii_lowercase, repeat = 3)]

	for f, feat in enumerate(sparse_map):
		action = action_labels[f]

		for p, pair in enumerate(feat):
			ofile.write("{0}_{1}_{2} {3}\n".format(action, p, 's', pair[0]))
			ofile.write("{0}_{1}_{2} {3}\n".format(action, p, 'e', pair[1]))

	ofile.close()

	return sparse_map

def display(iad_filename, pruning_indexes, layer, sparse_map, name="image", show=True, save=True):
	
	# open the IAD and prune any un-wanted features
	iad = np.load(iad_filename)["data"]

	idx = pruning_indexes[layer]
	iad = iad[idx]
	iad[:] = 0.0

	# inverse IAD so that black indicates high expression
	iad = 1-iad

	# convert from Greyscale to HSV for easier color manipulation
	iad = cv2.cvtColor(iad,cv2.COLOR_GRAY2BGR)
	iad = cv2.cvtColor(iad,cv2.COLOR_BGR2HSV)

	# overlay sparse map on top of IAD. Color is only used to help differentiate the
	# different features and has no other significance. We use a value of 512 to make 
	# the color difference more significant
	for f, vals in enumerate(sparse_map):
		for v in vals:
			iad[f, v[0]: v[1], 0] = 512*float(f)/len(sparse_map)
			iad[f, v[0]: v[1], 1] = 0.75
	
	# convert from HSV to BGR to display correctly
	iad = cv2.cvtColor(iad,cv2.COLOR_HSV2BGR)

	# resize image
	scale = 6
	iad = cv2.resize(iad, (iad.shape[1]*scale, iad.shape[0]*scale), interpolation=cv2.INTER_NEAREST)

	# display/save image 
	if show:
		cv2.imshow(name,iad)
		cv2.waitKey(0)
	if save:
		iad *= 255
		iad = iad.astype(np.uint8)
		cv2.imwrite(name,iad)
	
	

def main(dataset_dir, csv_filename, dataset_type, dataset_id, feature_retain_count):

	#setup feature_rank_parser
	iad_data_path = os.path.join(dataset_dir, 'iad_'+dataset_type+'_'+str(dataset_id))
	ranking_file = os.path.join(iad_data_path, "feature_ranks_"+str(dataset_id)+".npz")
	assert os.path.exists(ranking_file), "Cannot locate Feature Ranking file: "+ ranking_file
	pruning_indexes = get_top_n_feature_indexes(ranking_file, feature_retain_count)
	
	#setup file-io
	txt_path = os.path.join(dataset_dir, 'txt_'+dataset_type+'_'+str(dataset_id))
	if(not os.path.exists(txt_path)):
		os.makedirs(txt_path)

	#get files from dataset
	csv_contents = read_csv(csv_filename)
	try:
		csv_contents = read_csv(csv_filename)
	except:
		print("ERROR: Cannot open CSV file: "+ csv_filename)

	file_list = [ex for ex in csv_contents if ex['dataset_id'] >= dataset_id or ex['dataset_id'] != 0]
	for ex in file_list:
		file_location = os.path.join(ex['label_name'], ex['example_id'])
		for layer in range(5):

			iad_filename = os.path.join(iad_data_path, file_location+"_"+str(layer)+".npz")
			assert os.path.exists(iad_filename), "Cannot locate IAD file: "+ iad_filename
			
			txt_filename = os.path.join(txt_path, file_location+"_"+str(layer)+".txt")
			sparsify_iad(iad_filename, pruning_indexes, layer, name=txt_name)

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Generate IADs from input files')
	#required command line args
	parser.add_argument('dataset_dir', help='the directory where the dataset is located')
	parser.add_argument('csv_filename', help='a csv file denoting the files in the dataset')
	parser.add_argument('dataset_type', help='the dataset type', choices=['frames', 'flow'])
	parser.add_argument('dataset_id', type=int, help='a csv file denoting the files in the dataset')

	parser.add_argument('--feature_retain_count', type=int, default=10000, help='the number of features to remove')

	FLAGS = parser.parse_args()

	main(FLAGS.dataset_dir, 
		FLAGS.csv_filename,
		FLAGS.dataset_type,
		FLAGS.dataset_id,
		FLAGS.feature_retain_count
		)

