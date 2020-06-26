'''
Binary format:
first value is the number of features
followed by pairs of unsigned ints (may need to change 
this if the value IAD is longer than 256 frames). First 
pair is the index and has the format N, 0. Following pairs 
are start and stop indexes:

//binary string
128 0 0 2 5 12 15 27 0 0 3 14 18 127 0 5 10

//organized as follows 
128                       // Num features
(0 0) : (2 5) (12 15)     // Feature 0, start and stop times between 2 and 5 and 12 and 15
(27 0) : (0 3) (14 18)    // Feature 27, start and stop times between 0 and 3 and 14 and 18
(127 0) : (5 10)          // Feature 127, start and stop times between 5 and 10
'''

from struct import pack, unpack

def write_sparse_matrix(filename, sparse_map):
	
	ofile = open(filename, "wb")
	ofile.write(pack('I', len(sparse_map)))
	for i, data in enumerate(sparse_map):
		if(len(data) > 0):
			ofile.write(pack('II', i, 0))
			for d in data:
				ofile.write(pack('II', d[0], d[1]))

	ofile.close()

def read_sparse_matrix(filename):

	f = open(filename,'rb')
	num_features = int(unpack('I',f.read(4))[0])

	print("num_features:", num_features)
	sparse_map = [[] for x in range(num_features)]
	track = -1
	while True:
		try:
			p1 = unpack('I',f.read(4))[0]
			p2 = unpack('I',f.read(4))[0]
		except:
			break

		if(p2 == 0):
			track = p1	
		else:
			sparse_map[track].append([p1,p2])
		
	return sparse_map

if __name__ == '__main__':
	read_sparse_matrix("/home/mbc2004/datasets/Something-Something/b_tsm_frames_1/0/2.b")
