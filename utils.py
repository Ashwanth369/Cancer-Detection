import os,cv2
import large_image
import numpy as np
import matplotlib.pyplot as plt
import shutil
import sys
def check_image(image):

	count = 0
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			# print(image[i][j])
			if image[i][j][1] > 200 or image[i][j][2] > 200:  ##### Tentative
				count += 1
			if count>750*1000:
				return False

	if count/(image.shape[0]*image.shape[1]) > 0.70:		### Again Tentative
		return False
	else:
		return True

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

nm_im=0
def tiling(path,folder,counter):

	# if counter == 0:
	ts = large_image.getTileSource(path)
	num_tiles = 0
	# print(wsi_path)
	tile_means = []
	tile_areas = []
	it=0
	for tile_info in ts.tileIterator(
	    region=dict(left=5000, top=5000, width=20000, height=20000, units='base_pixels'),
	    scale=dict(magnification=20),
	    tile_size=dict(width=400, height=400),
	    tile_overlap=dict(x=0, y=0),
	    format=large_image.tilesource.TILE_FORMAT_PIL
	):
	    print("------------------------"+str(nm_im)+"------"+str(counter)+"----"+str(it)+"------------------------")
	    im_tile = (tile_info['tile'])
	    im_tile = np.array(im_tile)
	    flag = check_image(im_tile)
	    if flag:
	    	plt.imshow(im_tile)
	    	plt.show()
	    	print(im_tile.shape)

	    	cv2.imwrite('test_images/gbm/'+str(counter)+'.png',im_tile)
	    	counter += 1
	    it+=1
	    # print(it, tile_info['gheight'],tile_info['gheight'])
	    # plt.imshow(im_tile)
	    # plt.show()
	    # if tile_info['gwidth']==2000 and tile_info['gheight']==2000:
	    #     im_tile.convert('LA').save(str("im1/grayscale/")+str(it)+'.png')
	    #     # plt.savefig(str("im1/color/")+str(it)+'.png')
	    # # tile_mean_rgb = np.mean(im_tile[:, :, :3], axis=(0, 1))

	    # # tile_means.append( tile_mean_rgb )
	    # tile_areas.append( tile_info['width'] * tile_info['height'] )

	    num_tiles += 1

	# slide_mean_rgb = np.average(tile_means, axis=0, weights=tile_areas)
	# print(it)
	print('Number of tiles = {}'.format(num_tiles))
	# print('Slide mean color = {}'.format(slide_mean_rgb))
	ts.getNativeMagnification()
	ts.getMagnificationForLevel(level=0)
	return counter


counter = 0
for file in os.listdir('gbm'):
	for img in os.listdir('gbm/'+file):
		if '.svs' in img:
			# print(img)
			# imgs = tiling(img)
			# print('gbm/'+file+'/'+img)
			nm_im+=1
			print("____________________________________________________")
			folder = 'gbm/'+file
			counter += tiling('gbm/'+file+'/'+img,folder,counter)
			if counter>=1000:
				sys.exit()
			# if counter>=0:
			shutil.rmtree(folder)
			# os.remove(folder)
			# print(image.shape)
			print("****************************************************")
			# print(image.shape)
