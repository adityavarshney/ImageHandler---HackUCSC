from PIL import Image
import numpy as np
import queue
import time

im = Image.open("group.jpg")
im = im.convert("RGBA")


def find_edges(arr):
	for index in range(len(arr)):
		for i in range(len(arr[index])):
			if arr[index][i]==True:
				return i

data = np.array(im)

red,green,blue,alpha = data.T # transpose the np array with image data (color-based rows)

white_areas = np.logical_and(red > 200, blue > 200, green > 200)
green_areas = np.logical_and(green > 180, red < 200, blue < 200)
black_areas = np.logical_and(red < 55, blue < 55, green < 55)
red_areas = np.logical_and(red > 200, green < 200, blue < 200)
brown_areas = np.logical_and(red > 140, 70 < green, blue < 140)

data[...,:-1][red_areas.T] = (255,0,0)
data[...,:-1][white_areas.T] = (255,255,255)
data[...,:-1][green_areas.T] = (0,255,0)
data[...,:-1][black_areas.T] = (0,0,0)
data[...,:-1][brown_areas.T] = (139,69,19)

min_col_ind, min_row_ind = 100000, 100000
max_col_ind, max_row_ind = -100000, -100000

# found = false
def sum_col(col):
	r,g,b,a = 0,0,0,0
	for elem in col:
		r+=elem[0]
		g+=elem[1]
		b+=elem[2]
		a+=elem[3]
	return np.array([r,g,b,a])

def list_eq(seq1,seq2):
	if len(seq1) != len(seq2):
		return False
	for e1,e2 in zip(seq1,seq2):
		if e1!=e2:
			return False
	return True

data2 = data[:,:]
start_time = time.time()
r,g,b,a = 0,0,0,0
filter_size = 21
q = queue.Queue(filter_size)
half = filter_size//2
square = (filter_size**2)
for row_ind in range(half,len(data)-half):
	# print("hello")
	# if(elapsed_time>=15):
	# 	break
	# print("i told u so")
	total = np.zeros([1,4])
	q = queue.Queue(filter_size)
	# print("but really tho")
	for col_index in range(filter_size): # create the blur block
		# print('bye')
		col = data[(row_ind-half):(row_ind+half+1),col_index]
		# print('hi')
		temp = sum_col(col)
		# print('88')
		total+=temp
		# print('here\'s the error')
		q.put(temp)
		# print(queue)
	#need to store first blurred pixel
	for col_ind in range(filter_size,len(data[row_ind])): # move down the row, blur one col at a time
		data2[row_ind][col_ind-half-1] = (total[0,0]//square,total[0,1]//square,total[0,2]//square,total[0,3]//square) #tuple(elem//square for elem in total)
		total -= q.get()
		col = data[(row_ind-half):(row_ind+half+1),col_ind]
		temp = sum_col(col)
		total+=temp
		q.put(temp)
		# print(col_ind)
	# print(row_ind)



		#block = data[(row_ind-half):(row_ind+half+1),(col_ind-half):(col_ind+half+1)] # the save
		# if(elapsed_time>=15):
		# 	break
		# for i in range(len(block)):
		# 	for j in range(len(block[i])):
		# 		r += block[i][j][0]
		# 		g += block[i][j][1]
		# 		b += block[i][j][2]
		# 		a += block[i][j][3]
		# c = (r//square,g//square,b//square,a//square)
		# data[row_ind][col_ind] = c
		# r,g,b,a = 0,0,0,0
elapsed_time = time.time()-start_time
print(elapsed_time)
im3 = Image.fromarray(data2) # changed to data2
im3.show()

# for row_ind in range(len(data)):
# 	for col_ind in range(len(data[row_ind])):
# 		if(list_eq(data[row_ind][col_ind],[255,255,255,255])):
# 			min_col_ind = min(col_ind,min_col_ind)
# 			min_row_ind = min(row_ind,min_row_ind)
# 			max_col_ind = max(col_ind,max_col_ind)
# 			max_row_ind = max(row_ind,max_row_ind)
# print(min_col_ind, min_row_ind,max_col_ind, max_row_ind)
# 		#print(pixel)
# im2 = Image.fromarray(data)
# im2.show()

def k_means(n_clusters, n_samples_per_cluster,n_features,embiggen_factor,seed):
	np.random.seed(seed)
	slices = []
	centroids = []
	for i in range(n_clusters):
		samples = 

def k_means(restaurants, k, max_updates=100):
    """Use k-means to group restaurants by location into k clusters."""
    assert len(restaurants) >= k, 'Not enough restaurants to cluster'
    old_centroids, n = [], 0
    # Select initial centroids randomly by choosing k different restaurants
    centroids = [restaurant_location(r) for r in sample(restaurants, k)]
    while old_centroids != centroids and n < max_updates:
        old_centroids = centroids
        # BEGIN Question 6
        centroids=[find_centroid(i) for i in group_by_centroid(restaurants,centroids)]
        # END Question 6
        n += 1
    return centroids


def group_by_first(pairs):
    """Return a list of pairs that relates each unique key in the [key, value]
    pairs to a list of all values that appear paired with that key.

    Arguments:
    pairs -- a sequence of pairs

    >>> example = [ [1, 2], [3, 2], [2, 4], [1, 3], [3, 1], [1, 2] ]
    >>> group_by_first(example)
    [[2, 3, 2], [2, 1], [4]]
    """
    keys = []
    for key, _ in pairs:
        if key not in keys:
            keys.append(key)
    return [[y for x, y in pairs if x == key] for key in keys]


def group_by_centroid(restaurants, centroids):
    """Return a list of clusters, where each cluster contains all restaurants
    nearest to a corresponding centroid in centroids. Each item in
    restaurants should appear once in the result, along with the other
    restaurants closest to the same centroid.
    """
    # BEGIN Question 4
    return group_by_first([[find_closest(restaurant_location(i),centroids),i] for i in restaurants])
    # END Question 4


def find_centroid(cluster):
    """Return the centroid of the locations of the restaurants in cluster."""
    # BEGIN Question 5
    return [mean([restaurant_location(i)[0] for i in cluster]),mean([restaurant_location(i)[1] for i in cluster])]
        # END Question 5