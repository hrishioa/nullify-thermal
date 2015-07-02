import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import sys
from subprocess import call

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

	filename = 'MOV_2150.mp4'

	video = cv2.VideoCapture('Video/'+filename)
	print(video.grab())

	counter = 0
	
	data = []

	while(video.isOpened()):

		counter+=1

		ret, frame = video.read()

		if ret==False:
			break

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		#hist = cv2.calcHist([gray],[0],None,[256],[0,256])
		
		framedat = gray.ravel()

		#Process the incoming data

		#find out the limit of valid data
		pointer = int((hist_samples/(temp_max-temp_min))*(threshold_temp-temp_min))

		temp = np.histogram(framedat,bins=256,range=(0,256))[0][pointer:]

		delta = [0]
		if(data!=[]):
			delta=temp-data
		data = temp

		firecon=0
		maxmean = (np.max(data)-np.mean(data))

		if(maxmean>520):
			firecon=1
		if(maxmean>600):
			firecon=2
		if(maxmean>650):
			firecon=3
		if(maxmean>700):
			firecon=4

		#Plot statistics for now
		stats = "Pointer: %d \nMean : %.2f \nMedian: %d \nMin : %d \nMax : %d \nSD: %.2f \nMax-Mean: %d\n\nFire Alert:      %d" % (pointer,np.mean(data),np.median(data),np.min(data),np.max(data),np.std(data),maxmean,(firecon))

		plt.hist(framedat,256,[pointer,256])
		plt.annotate(stats, xy=(1, 1), xycoords='axes fraction', fontsize=16,horizontalalignment='right', multialignment='left', verticalalignment='top',bbox=dict(facecolor='black', alpha=0.1))

		''' For plotting a color histogram
		color = ('b','g','r')
		for i,col in enumerate(color):
			histr = cv2.calcHist([frame],[i],None,[256],[0,256])
			plt.plot(histr,color = col)
			plt.xlim([0,256])
			'''
		#plt.show()
		save("Histograms/"+filename+"/histogram#"+str(counter), ext="png", close=True, verbose=False)
		print(filename+": Frame "+str(counter)+" done.")

		#Now save delta data
		plt.plot(np.arange(len(delta)),delta)
		plt.xlabel('Temperature')
		plt.ylabel('Delta')
		plt.ylim([-50,50])
		delstats = "Pointer: %d \nMean : %.2f \nMedian: %d \nMin : %d \nMax : %d \nSD: %.2f \nMax-Mean: %d" % (pointer,np.mean(delta),np.median(delta),np.min(delta),np.max(delta),np.std(delta),(np.max(delta)-np.mean(delta)))
		plt.annotate(delstats, xy=(1, 1), xycoords='axes fraction', fontsize=16,horizontalalignment='right', multialignment='left', verticalalignment='top',bbox=dict(facecolor='black', alpha=0.1))
		#plt.show()
		save("Histograms/"+filename+"/del_histogram#"+str(counter), ext="png", close=True, verbose=False)


	video.release()
	cv2.destroyAllWindows()

	#Run ffmpeg to create video
	print("Generating video")
	call("ffmpeg -framerate 30 -y -i del_histogram#%%d.png del_hist"+filename,shell=True)
	call("ffmpeg -framerate 30 -y -i histogram#%%d.png hist"+filename,shell=True)




if __name__ == '__main__':
	main()