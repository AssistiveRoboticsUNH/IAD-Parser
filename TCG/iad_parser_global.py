import numpy as np 
import os

import cv2
import PIL.Image as Image

from scipy.signal import savgol_filter

from itertools import product
from string import ascii_lowercase

import sys, math
sys.path.append("../../IAD-Generator/iad-generation/")
from feature_rank_utils import get_top_n_feature_indexes, get_top_n_feature_indexes_combined
from csv_utils import read_csv

def preprocess(iad, layer):

	if(iad.shape[1] > 10):
	 	iad = iad[:, 3:-3]

	smooth_value = 25
	if(layer >= 1 and layer != 3):
		smooth_value = 35

	if(iad.shape[1] > smooth_value):
		for i in range(iad.shape[0]):
			iad[i] = savgol_filter(iad[i], smooth_value, 3)

	return iad

def find_start_stop(feature, iad, threshold_value):

	# smooth the IAD expression
	#if(iad.shape[1] > 25):
		#w_size =  (iad.shape[1]/2) if (iad.shape[1]/2) % 2 != 0 else (iad.shape[1]/2)-1
		#feature = savgol_filter(feature, 25, 3)  ## 25
	
	
	# threshold the expression we are looking at

	#avg_val = np.mean(feature)
	#std_dev = np.std(feature)

	#print("mean: {:2.4f}, std: {:2.4f}, thresh: {:2.4f}, max: {:2.4f}, min: {:2.4f}".format(avg_val, std_dev, avg_val+std_dev, max(feature), min(feature)))
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
			#pair[0] += 3 
			#pair[1] += 3
			
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


def sparsify_iad(datatset_type_list, iad_filenames, pruning_indexes, layer, global_threshold, name="output.txt"):
	'''
	Convert an IAD into a sparse map that indicates the start and stop times 
	when a feature is expressed. Write the map to a file.
	'''

	# open the IAD and prune any un-wanted features
	iad, min_len = [], sys.maxint
	for dt in datatset_type_list:
		iad_data = np.load(iad_filenames[dt])["data"]

		idx = pruning_indexes[dt][layer]
		#print("shape:", iad_data[idx].shape, len(idx))

		iad.append(iad_data[idx])
		min_len = min(min_len, iad_data[idx].shape[1])

	#if the IADs for frames and flow are different lengths make them the same
	for i in range(len(iad)):
		iad[i] = iad[i][:, :min_len]
	iad = np.concatenate(iad, axis=0)

	#print("IAD shape", iad.shape, min_len)


	# determine start_stop_times for each feature in the IAD. Apply
	# any pre or post processing dteps to clean up the IAD and sparse map
	iad = preprocess(iad, layer)
	sparse_map = []
	for i, feature in enumerate(iad):
		threshold_value = global_threshold["mean"][layer][i]# + global_threshold["std_dev"][layer][i]
		sparse_map.append(find_start_stop(feature, iad, threshold_value))
	sparse_map = postprocess(sparse_map, layer)

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




def add_to_global_threshold(datatset_type_list, iad_filenames, pruning_indexes, layer, global_threshold):
	'''
	Convert an IAD into a sparse map that indicates the start and stop times 
	when a feature is expressed. Write the map to a file.
	'''

	iad, min_len = [], sys.maxint
	for dt in datatset_type_list:
		iad_data = np.load(iad_filenames[dt])["data"]

		idx = pruning_indexes[dt][layer]
		#print("shape:", iad_data[idx].shape, len(idx))

		iad.append(iad_data[idx])
		min_len = min(min_len, iad_data[idx].shape[1])

	#if the IADs for frames and flow are different lengths make them the same
	for i in range(len(iad)):
		iad[i] = iad[i][:, :min_len]
	iad = np.concatenate(iad, axis=0)

	
	#obtain the average and std_dev for each feature in the IAD
	prev_count = global_threshold["count"][layer] 
	new_count = prev_count + iad.shape[1]

	for i, feature in enumerate(iad):

		count = global_threshold["count"] [layer]
		if(count > 0):
			for j, x in enumerate(feature):


				prev_mean = global_threshold["mean"][layer][i]
				new_mean = ((prev_mean * (count + j)) + x) / (count + j + 1)

				#print(x)
				#print(prev_mean)
				#print(new_mean)
				#print('')
				global_threshold["mean"][layer][i] = new_mean

				variance = global_threshold["std_dev"][layer][i] ** 2
				diff_sums = (x - new_mean) * (x - prev_mean)
				new_variance = ((count+j-1)*variance + diff_sums) / (count+j)

				global_threshold["std_dev"][layer][i] = math.sqrt(new_variance)
		else:
			global_threshold["mean"][layer][i] = np.mean(feature)
			global_threshold["std_dev"][layer][i] = np.std(feature)
		


	global_threshold["count"][layer] += iad.shape[1]


"""
def sparsify_iad2(datatset_type_list, iad_filenames, pruning_indexes, layer, name="output.txt"):
	'''
	Convert an IAD into a sparse map that indicates the start and stop times 
	when a feature is expressed. Write the map to a file.
	'''

	# open the IAD and prune any un-wanted features
	iad, min_len = [], sys.maxint
	for dt in datatset_type_list:
		iad_data = np.load(iad_filenames[dt])["data"]

		idx = pruning_indexes[dt][layer]
		#print("shape:", iad_data[idx].shape, len(idx))

		iad.append(iad_data[idx])
		min_len = min(min_len, iad_data[idx].shape[1])

	#if the IADs for frames and flow are different lengths make them the same
	for i in range(len(iad)):
		iad[i] = iad[i][:, :min_len]
	iad = np.concatenate(iad, axis=0)

	#print("IAD shape", iad.shape, min_len)


	# determine start_stop_times for each feature in the IAD. Apply
	# any pre or post processing dteps to clean up the IAD and sparse map
	iad = preprocess(iad, layer)
	#sparse_map = []
	#for feature in iad:
	#	sparse_map.append(find_start_stop(feature, iad))
	#sparse_map = postprocess(sparse_map, layer)

	# write start_stop_times to file. Each feature is given a unique 3 char  
	# alphabetical code to identify it. This should cover up to 17K unique
	# features
	#ofile = open(name, 'w')
	#ofile.close()

	#print(np.max(iad, axis=1).shape)
	np.savez(name, max=np.max(iad, axis=1), mean=np.mean(iad, axis=1), min=np.min(iad, axis=1))

	#return sparse_map
"""
def display(iad_filename, pruning_indexes, layer, sparse_map, name="image", show=True, save=True):
	
	# open the IAD and prune any un-wanted features
	iad = np.load(iad_filename)["data"]
	scale = 6

	idx = pruning_indexes[layer]
	iad = iad[idx]
	#iad[:] = 0.0

	print("IAD shape:", iad.shape)
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

	datatset_type_list = []
	if(dataset_type=="frames" or dataset_type=="both"):
		datatset_type_list.append("frames")
	if(dataset_type=="flow" or dataset_type=="both"):
		datatset_type_list.append("flow")

	#setup feature_rank_parser
	frame_ranking_file = os.path.join( dataset_dir, 'iad_frames_'+str(dataset_id), "feature_ranks_"+str(dataset_id)+".npz") 
	flow_ranking_file = os.path.join( dataset_dir, 'iad_flow_'+str(dataset_id), "feature_ranks_"+str(dataset_id)+".npz") 

	pruning_indexes = {}
	if(dataset_type=="frames"):
		assert os.path.exists(frame_ranking_file), "Cannot locate Feature Ranking file: "+ frame_ranking_file
		pruning_indexes["frames"] = get_top_n_feature_indexes(frame_ranking_file, feature_retain_count)
	elif(dataset_type=="flow"):
		assert os.path.exists(flow_ranking_file), "Cannot locate Feature Ranking file: "+ flow_ranking_file
		pruning_indexes["flow"] = get_top_n_feature_indexes(flow_ranking_file, feature_retain_count)
	elif(dataset_type=="both"):
		assert os.path.exists(frame_ranking_file), "Cannot locate Feature Ranking file: "+ frame_ranking_file
		assert os.path.exists(flow_ranking_file), "Cannot locate Feature Ranking file: "+ flow_ranking_file

		# UCF 1 -> ready to get load_ensemble
		#weight_ranking = [[],[]]

		# UCF 2* - > training
		#weight_ranking = [[0.120804,0.24663,0.31483,0.674861,0.842982],[0.208829,0.256675,0.354216,0.557494,0.6894]]

		# UCF 3* - > training
		#weight_ranking = [[0.100449,0.204335,0.270949,0.528152,0.726408],[0.161248,0.187946,0.21438,0.378007,0.473169]]

		# HMDB 1 
		weight_ranking = [[0.085621,0.201307,0.21634,0.527451,0.675817],[0.190196,0.233333,0.263399,0.296732,0.332026]]

		# HMDB 2* - > training
		#weight_ranking = [[0.075163,0.145752,0.169281,0.365359,0.566013],[0.131373,0.184314,0.205882,0.282353,0.443791]]

		# HMDB 3* - > training
		#weight_ranking = [[0.054248,0.127451,0.14183,0.25098,0.462092],[0.10915,0.145098,0.137255,0.138562,0.231373]]

		# BLOCKMOVING 1
		#weight_ranking = [[],[]]

		# BLOCKMOVING 2
		#weight_ranking = [[],[]]
		
		# BLOCKMOVING 3
		#weight_ranking = [[],[]]
		
		pruning_indexes = get_top_n_feature_indexes_combined(frame_ranking_file, flow_ranking_file, feature_retain_count, weight_ranking)

	#setup file-io
	txt_path = os.path.join(dataset_dir, 'txt_'+dataset_type+'_'+str(dataset_id))
	if(not os.path.exists(txt_path)):
		os.makedirs(txt_path)

	#get files from dataset
	try:
		csv_contents = read_csv(csv_filename)
	except:
		print("ERROR: Cannot open CSV file: "+ csv_filename)

	global_threshold_values = {"mean": [], "std_dev":[], "count":[]}
	for i in range(5):
		global_threshold_values["mean"].append([0]*feature_retain_count)
		global_threshold_values["std_dev"].append([0]*feature_retain_count)
		global_threshold_values["count"].append(0)


	train_list = [ex for ex in csv_contents if ex['dataset_id'] >= dataset_id]
	file_list = [ex for ex in csv_contents if ex['dataset_id'] >= dataset_id or ex['dataset_id'] == 0]
	
	for ex in train_list:
		file_location = os.path.join(ex['label_name'], ex['example_id'])
		print("Converting "+file_location)
		for layer in range(5):

			iad_filenames = {}
			for dt in datatset_type_list:
				iad_filenames[dt] = os.path.join(dataset_dir, 'iad_'+dt+'_'+str(dataset_id), file_location+"_"+str(layer)+".npz") 
				assert os.path.exists(iad_filenames[dt]), "Cannot locate IAD file: "+ iad_filenames[dt]

			txt_filename = os.path.join(txt_path, str(layer), file_location+"_"+str(layer)+".txt")
			add_to_global_threshold(datatset_type_list, iad_filenames, pruning_indexes, layer, global_threshold_values)
	'''
	for i in range(5):
		print(str(i))
		print("mean:", global_threshold_values["mean"][i])
		print("std_dev:", global_threshold_values["std_dev"][i])
		print("count:", global_threshold_values["count"][i])
		print('')
	'''

	
	for ex in file_list:
		file_location = os.path.join(ex['label_name'], ex['example_id'])
		print("Converting "+file_location)
		for layer in range(5):

			iad_filenames = {}
			for dt in datatset_type_list:
				iad_filenames[dt] = os.path.join(dataset_dir, 'iad_'+dt+'_'+str(dataset_id), file_location+"_"+str(layer)+".npz") 
				assert os.path.exists(iad_filenames[dt]), "Cannot locate IAD file: "+ iad_filenames[dt]
			
			label_dir = os.path.join(txt_path, str(layer),ex['label_name'])
			if ( not os.path.exists(label_dir) ):
				os.makedirs(label_dir)

			txt_filename = os.path.join(txt_path, str(layer), file_location+"_"+str(layer)+".txt")
			sparsify_iad(datatset_type_list, iad_filenames, pruning_indexes, layer, global_threshold_values, name=txt_filename)
			#sparsify_iad2(datatset_type_list, iad_filenames, pruning_indexes, layer, name=txt_filename)
	


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Generate IADs from input files')
	#required command line args
	parser.add_argument('dataset_dir', help='the directory where the dataset is located')
	parser.add_argument('csv_filename', help='a csv file denoting the files in the dataset')
	parser.add_argument('dataset_type', help='the dataset type', choices=['frames', 'flow', 'both'])
	parser.add_argument('dataset_id', type=int, help='a csv file denoting the files in the dataset')

	parser.add_argument('--feature_retain_count', type=int, default=128, help='the number of features to remove')

	FLAGS = parser.parse_args()
	
	main(FLAGS.dataset_dir, 
		FLAGS.csv_filename,
		FLAGS.dataset_type,
		FLAGS.dataset_id,
		FLAGS.feature_retain_count
		)
	'''
	
	#for dataset_type in ['frames']:#, 'flow', 'both']:
	for dataset_id in [1,2,3]:#2,3]:

		main(FLAGS.dataset_dir, 
			FLAGS.csv_filename,
			#dataset_type,
			FLAGS.dataset_type,
			dataset_id,
			FLAGS.feature_retain_count
			)
	'''
