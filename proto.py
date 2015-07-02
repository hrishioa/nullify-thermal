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

	video = cv2.VideoCapture('MOV_2146.mp4')
	print(video.grab())

	counter = 0
	
	while(video.isOpened()):
		counter+=1

		ret, frame = video.read()

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		#hist = cv2.calcHist([gray],[0],None,[256],[0,256])
		
		framedat = gray.ravel()

		plt.hist(framedat,256,[0,256])

		#Process the incoming data

		#find out the limit of valid data
		pointer = int((hist_samples/(temp_max-temp_min))*(threshold_temp-temp_min))

		data = framedat[pointer:]

		data = np.histogram(framedat,bins=256,range=(0,256))[0][pointer:]

		#Plot statistics for now
		stats = "Pointer: %d \nMean : %.2f \nMedian: %d \nMin : %d \nMax : %d \nSD: %.2f \nMax-Mean: %d" % (pointer,np.mean(data),np.median(data),np.min(data),np.max(data),np.std(data),(np.max(data)-np.mean(data)))

		plt.annotate(stats, xy=(1, 1), xycoords='axes fraction', fontsize=16,horizontalalignment='right', multialignment='left', verticalalignment='top',bbox=dict(facecolor='black', alpha=0.1))

		''' For plotting a color histogram
		color = ('b','g','r')
		for i,col in enumerate(color):
			histr = cv2.calcHist([frame],[i],None,[256],[0,256])
			plt.plot(histr,color = col)
			plt.xlim([0,256])
			'''
		#plt.show()
		save("Histograms/MOV_2146_2/histogram#"+str(counter), ext="png", close=True, verbose=False)
		print("Frame "+str(counter)+" done.")

	video.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()