import numpy as np 
import os

import cv2
import PIL.Image as Image

from scipy.signal import savgol_filter

from itertools import product
from string import ascii_lowercase

import sys, math
sys.path.append("../../IAD-Generator/iad-generation/")
from csv_utils import read_csv

from multiprocessing import Pool


def open_iad(ex, dataset_type_list, layer, pruning_indexes=None):

	# open the IAD and prune any un-wanted features
	iad, min_len = [], sys.maxint
	for dtype in dataset_type_list:
		iad_data = np.load(ex['iad_path_{0}_{1}'.format(dtype, layer)])["data"]

		if(pruning_indexes != None):
			idx = pruning_indexes[dtype][layer]
			iad.append(iad_data[idx])
			min_len = min(min_len, iad_data[idx].shape[1])
		else:
			iad.append(iad_data)

	#if the IADs for frames and flow are different lengths make them the same
	if(pruning_indexes != None):
		for i in range(len(iad)):
			iad[i] = iad[i][:, :min_len]

	#combine frames and flow together
	iad = np.concatenate(iad, axis=0)

	return iad

def preprocess(iad, layer):

	# if the length of the IAD is atleast 10 then we trim the beginning and ending frames 
	# to remove noise
	if(iad.shape[1] > 10):
	 	iad = iad[:, 3:-3]

	# use savgol filter to smooth the IAD
	smooth_value = 25
	if(layer >= 1):
		smooth_value = 35

	if(iad.shape[1] > smooth_value):
		for i in range(iad.shape[0]):
			iad[i] = savgol_filter(iad[i], smooth_value, 3)

	return iad
'''
def find_start_stop(feature, iad, threshold_value):
	
	# threshold the expression we are looking at
	above_threshold = np.argwhere(feature > threshold_value).reshape(-1)

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
'''
def find_start_stop(feature_row):
	
	# identify the start and stop times of the events
	start_stop_times = []
	if(len(feature_row) != 0):
		start = feature_row[0]
		for i in range(1, len(feature_row)):

			if( feature_row[i-1]+1 < feature_row[i] ):
				start_stop_times.append([start, above_threshold[i-1]+1])
				start = above_threshold[i]

		start_stop_times.append([start, above_threshold[len(above_threshold)-1]+1])

	return start_stop_times


def postprocess(sparse_map, layer):
	'''
	Take a sparse map and offset it by three to account for the trimmed IAD.
	Remove any start stop times that are shorter than 3 time instances
	'''

	noise_limit = 3
	if (layer >= 2 ):
		noise_limit = 1

	for f, feat in enumerate(sparse_map):	
		remove_pairs = []
		for p, pair in enumerate(feat):
			if pair[1]-pair[0] > noise_limit:

				#offset accoridng to beginning and end trimming
				pair[0] += 3 
				pair[1] += 3 

			else:
				remove_pairs.append(pair)
			
		# remove pairs that are smaller than 3 in length
		for pair in remove_pairs:
			feat.remove(pair)
		
	return sparse_map


def write_sparse_matrix(filename, sparse_map):
	txt = ''
	for i, data in enumerate(sparse_map):
		for d in data:
			txt += d[0]+' '+d[1]+' '
		txt += '\n'
	ofile = open(filename, "wb")
	ofile.write(bytearray(txt))
	ofile.close()

def read_sparse_matrix(filename):
	sparse_map = []
	for line in list(open(filename, "rb")):
		data = [int(x) for x in line.split()]
		sparse_map.append([(data[i], data[i+1]) for i in range(0, len(data), 2) ])
	return sparse_map

def sparsify_iad(ex, layer, dataset_type_list, threshold_matrix, name="output.txt"):
	'''
	Convert an IAD into a sparse map that indicates the start and stop times 
	when a feature is expressed. Write the map to a file.
	'''
	threshold_values = threshold_matrix[layer]

	# open the IAD
	iad = open_iad(ex, dataset_type_list, layer)
	iad = preprocess(iad, layer)

	# threshold, reverse the locations to account for the transpose
	locs = np.where(iad.T > threshold_values)
	locs = zip( locs[1], locs[0] )

	sparse_map = []
	for i in range(iad.shape[0]):
		print( np.where(locs[:,0] == i) )
		print( locs[np.where(locs[:,0] == i)] )
		feature_times = locs[np.where(locs[:,0] == i)][:,1]
		sparse_map.append( find_start_stop( feature_times ))
	sparse_map = postprocess(sparse_map, layer)


	# write start_stop_times to file.

	print(ex['txt_path_{0}'.format(layer)])
	write_sparse_map(ex['txt_path_{0}'.format(layer)], sparse_map)
	smx = read_sparse_matrix(ex['txt_path_{0}'.format(layer)])

	print(smx)





	'''
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
	'''
	return

def sparsify_iad_dataset(inp):
	'''
	Convert an IAD into a sparse map that indicates the start and stop times 
	when a feature is expressed. Write the map to a file.
	'''

	csv_dataset, depth_size, dataset_type_list, threshold_matrix = inp

	for layer in range(depth_size):
		for ex in csv_dataset:
			sparsify_iad(ex, layer, dataset_type_list, threshold_matrix)


class Avg:
	def __init__(self):
		self.mean = 0
		self.count = 0

	def update(self, other_arr):
		self.mean *= self.count
		self.mean += np.sum(other_arr)
		self.count += len(other_arr)
		self.mean /= self.count


def determine_threshold(inp):
	'''
	Convert an IAD into a sparse map that indicates the start and stop times 
	when a feature is expressed. Write the map to a file.
	'''
	csv_dataset, depth_size, dataset_type_list, num_features = inp
	
	'''
	Threshold has the following shape: [num_layers, num_features, {mean, count}]
	'''

	threshold = []
	for layer in range(depth_size):
		local_threshold = [Avg() for i in range(num_features)]

		for ex in csv_dataset:

			# open IAD
			iad = open_iad(ex, dataset_type_list, layer)

			#update local averages
			for i, f in enumerate(iad):
				local_threshold[i].update(f)

		threshold.append(local_threshold)
	return threshold

def split_dataset_run_func(p, func, dataset, other_args):
	chunk_size = len(dataset)/float(p._processes)
	inputs = []
	last = 0.0

	while last < len(dataset):
		inputs.append(
			([dataset[int(last):int(last+chunk_size)]] + other_args)
					)
		last += chunk_size

	return [func(inputs[0])]
	#return p.map(func, inputs)

def main(model_type, dataset_dir, csv_filename, dataset_type, dataset_id, 
	num_features, num_procs):

	if(model_type == 'i3d'):
		from gi3d_wrapper import depth_size
	if(model_type == 'trn'):
		from trn_wrapper import depth_size
	if(model_type == 'tsm'):
		from tsm_wrapper import depth_size

	
	dataset_type_list = []
	if(dataset_type=="frames" or dataset_type=="both"):
		dataset_type_list.append("frames")
	if(dataset_type=="flow" or dataset_type=="both"):
		dataset_type_list.append("flow")

	#get files from dataset
	csv_contents = read_csv(csv_filename)[:23]
	
	for ex in csv_contents:
		
		print(ex['example_id'])
		
		for layer in range(depth_size):

			#get IAD files for read
			for dtype in dataset_type_list:

				iad_path = 'iad_path_{0}_{1}'.format(dtype, layer)
				ex[iad_path] = os.path.join(dataset_dir, 'iad_{0}_{1}_{2}'.format(model_type, dtype, dataset_id), ex['label_name'], '{0}_{1}.npz'.format(ex['example_id'], layer))
				assert os.path.exists(ex[iad_path]), "Cannot locate IAD file: "+ ex[iad_path]

			#generate txt directory for write
			txt_dir = os.path.join(dataset_dir, 'txt_{0}_{1}_{2}'.format(model_type, dataset_type, dataset_id), ex['label_name']) 
			if ( not os.path.exists(txt_dir) ):
				os.makedirs(txt_dir)
			txt_path = 'txt_path_{0}'.format(layer)
			ex[txt_path] = os.path.join(txt_dir, '{0}_{1}.txt'.format(ex['example_id'], layer))
	
	p = Pool(num_procs)

	#get the threshold values for each feature in the training dataset
	training_dataset = [ex for ex in csv_contents if ex['dataset_id'] >= dataset_id][:20]
	other_args = [depth_size,dataset_type_list,num_features]

	split_threshold_info = split_dataset_run_func(p, determine_threshold, training_dataset, other_args)

	#combine chunked threshodl info together
	threshold_matrix = np.zeros((depth_size, num_features))
	for x in split_threshold_info:
		for layer in range(depth_size):
			for feature in range(num_features):
				threshold_matrix[layer, feature] += x[layer][feature].mean * x[layer][feature].count

	print(threshold_matrix.shape)
	print(threshold_matrix)

	
	#process the IADs and save the parsed files 
	full_dataset = [ex for ex in csv_contents if ex['dataset_id'] >= dataset_id or ex['dataset_id'] == 0]
	other_args = [depth_size,dataset_type_list,threshold_matrix]
	split_dataset_run_func(p, sparsify_iad_dataset, full_dataset, other_args)





if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Generate IADs from input files')
	#required command line args
	parser.add_argument('model_type', help='the type of model to use', choices=['i3d', 'trn', 'tsm'])
	parser.add_argument('dataset_dir', help='the directory where the dataset is located')
	parser.add_argument('csv_filename', help='a csv file denoting the files in the dataset')
	parser.add_argument('dataset_type', help='the dataset type', choices=['frames', 'flow', 'both'])
	parser.add_argument('dataset_id', type=int, help='a csv file denoting the files in the dataset')

	parser.add_argument('--num_features', type=int, default=128, help='the maximum number of features in each IAD')
	parser.add_argument('--num_procs', type=int, default=1, help='number of process to split IAD generation over')

	FLAGS = parser.parse_args()
	
	main(FLAGS.model_type,
		FLAGS.dataset_dir, 
		FLAGS.csv_filename,
		FLAGS.dataset_type,
		FLAGS.dataset_id,
		FLAGS.num_features,
		FLAGS.num_procs
		)
	