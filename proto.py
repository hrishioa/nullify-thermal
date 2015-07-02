import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import sys
print(sys.version)

np.set_printoptions(threshold=np.inf)

temp_max = 150 #maximum temperature setting for camera
temp_min = 0 #minimum temperature setting for camera
threshold_temp = 125 #threshold temperature in Celcius
hist_samples = 256 #number of pixel values

def save(path, ext='png', close=True, verbose=True):
	"""Save a figure from pyplot.

	Parameters
	----------
	path : string
		The path (and filename, without the extension) to save the
		figure to.

	ext : string (default='png')
		The file extension. This must be supported by the active
		matplotlib backend (see matplotlib.backends module).  Most
		backends support 'png', 'pdf', 'ps', 'eps', and 'svg'.

	close : boolean (default=True)
		Whether to close the figure after saving.  If you want to save
		the figure multiple times (e.g., to multiple formats), you
		should NOT close it in between saves or you will have to
		re-plot it.

	verbose : boolean (default=True)
		Whether to print information about when and where the image
		has been saved.

	"""
	
	# Extract the directory and filename from the given path
	directory = os.path.split(path)[0]
	filename = "%s.%s" % (os.path.split(path)[1], ext)
	if directory == '':
		directory = '.'
 
	# If the directory does not exist, create it
	if not os.path.exists(directory):
		os.makedirs(directory)
 
	# The final path to save to
	savepath = os.path.join(directory, filename)
 
	if verbose:
		print("Saving figure to '%s'..." % savepath),
 
	# Actually save the figure
	plt.savefig(savepath)
	
	# Close it
	if close:
		plt.close()

	if verbose:
		print("Done")

def main():
	print("Program Running...")

	video = cv2.VideoCapture('C:\Users\Hrishi\Dropbox\Projects\Thermal\MOV_2146.mp4')
	print(video.grab())

	counter = 0
	
	while(video.isOpened()):
		counter+=1

		ret, frame = video.read()

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		print("Read Frame "+str(counter)+".")

		'''
		cv2.imshow('frame',frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		'''

		#hist = cv2.calcHist([gray],[0],None,[256],[0,256])
		
		print(gray.ravel())

		plt.hist(gray.ravel(),256,[0,300])
		plt.show()

		''' For plotting a color histogram
		color = ('b','g','r')
		for i,col in enumerate(color):
			histr = cv2.calcHist([frame],[i],None,[256],[0,256])
			plt.plot(histr,color = col)
			plt.xlim([0,256])
			'''

		#save("Histograms/MOV_2146_2/histogram#"+str(counter), ext="png", close=True, verbose=False)
		print("Frame "+str(counter)+" done.")


	video.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()