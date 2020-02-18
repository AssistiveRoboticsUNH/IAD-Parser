from iad_parser import *


if __name__ == '__main__':
	
	src = "../input"
	#label = "after"
	label = "equals"


	files = [x for x in os.listdir(os.path.join(src, label)) if ".npz" in x and "feature" not in x]
	print(files)
	files = [files[0]]

	dataset_dir = src#"~/datasets/BlockMovingSep/"
	dataset_type = 'frames'
	dataset_id = 1
	feature_retain_count = 128
	#iad_data_path_frames = os.path.join(dataset_dir, 'iad_'+dataset_type+'_'+str(dataset_id))
	ranking_file = os.path.join(src, "feature_ranks_"+str(dataset_id)+".npz")#os.path.join(iad_data_path_frames, "feature_ranks_"+str(dataset_id)+".npz")
	assert os.path.exists(ranking_file), "Cannot locate Feature Ranking file: "+ ranking_file
	pruning_indexes = get_top_n_feature_indexes(ranking_file, feature_retain_count)
	

	for f in range(1):#4, 5):
		#file = "352_"+str(f)+".npz"
		file = "22_"+str(f)+".npz"
		layer = f
		iad_filename = os.path.join(src, label, file)
		assert os.path.exists(ranking_file), "Cannot locate IAD file: "+ iad_filename

		print(len(pruning_indexes[layer]))

		print("Opening: ", f)
		sparse_map = sparsify_iad(["frames"], {"frames":iad_filename}, {"frames":pruning_indexes}, layer)

		display(iad_filename, pruning_indexes, layer, sparse_map, name=label+"_"+str(f)+".png", show=False)
