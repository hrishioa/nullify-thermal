import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import sys
from subprocess import call
import thread
import pyqtgraph as pg 
import pyqtgraph.exporters

print(sys.version)

np.set_printoptions(threshold=np.inf)

threadcount=0
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

lock = thread.allocate_lock()

def video_engine(folder,filename):
	cpuplot=False

	global threadcount

	lock.acquire()
	print("\nStarting thread for file %s. Number of Threads: %d" % (filename,threadcount))
	lock.release()

	video = cv2.VideoCapture(folder+'/'+filename)

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

		colorcode = '#00FF00'

		if(firecon>3):
			colorcode = '#FF0000'
		elif(firecon>1):
			colorcode = '#FFFF00'

		stats = "<br /><br />Pointer: %d \nMean : %.2f \nMedian: %d \nMin : %d \nMax : %d \n<br >SD: %.2f \nMax-Mean: %d\n\n<font color='%s'>Fire Alert:      %d</font>" % (pointer,np.mean(data),np.median(data),np.min(data),np.max(data),np.std(data),maxmean,colorcode,firecon)

		if(cpuplot==False):
			plt = pg.plot(data)
			plt.setTitle(stats)
			exporter = pg.exporters.ImageExporter(plt.plotItem)
			exporter.parameters()['width'] = 750

			exporter.export('Histograms/%s/histogram#%d.png' % (filename,counter))
			plt.close()
		else:
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
		
		print(filename+": Frame "+str(counter)+" done. Threads: "+str(threadcount))

		delstats = "<br /><br />Pointer: %d \nMean : %.2f \nMedian: %d \nMin : %d \nMax : %d \n<br />SD: %.2f \nMax-Mean: %d" % (pointer,np.mean(delta),np.median(delta),np.min(delta),np.max(delta),np.std(delta),(np.max(delta)-np.mean(delta)))

		if(cpuplot==False):
			plt = pg.plot(delta)
			exporter = pg.exporters.ImageExporter(plt.plotItem)
			exporter.parameters()['width'] = 750
			exporter.export('Histograms/%s/del_histogram#%d.png' % (filename,counter))
			plt.close()
		else:
			#Now save delta data
			plt.plot(np.arange(len(delta)),delta)
			plt.xlabel('Temperature')
			plt.ylabel('Delta')
			plt.ylim([-50,50])
			plt.annotate(delstats, xy=(1, 1), xycoords='axes fraction', fontsize=16,horizontalalignment='right', multialignment='left', verticalalignment='top',bbox=dict(facecolor='black', alpha=0.1))
			#plt.show()
			#thread.start_new_thread(save,("Histograms/"+filename+"/del_histogram#"+str(counter),))
			save("Histograms/"+filename+"/del_histogram#"+str(counter), ext="png", close=True, verbose=False)

	video.release()
	cv2.destroyAllWindows()

	#Run ffmpeg to create video
	lock.acquire()
	print("Generating video")
	call("ffmpeg -framerate 30 -y -i Histograms/"+filename+"/del_histogram#%d.png del_hist"+filename,shell=True)
	call("ffmpeg -framerate 30 -y -i Histograms/"+filename+"/histogram#%d.png hist"+filename,shell=True)
	threadcount-=1
	print("Completed file %s. Threads running: %d" % (filename,threadcount))
	lock.release()

def main():
	global threadcount

	print("Program Running...")

	folder = 'X:/HrishiOlickel/Desktop/Thermal/Processing'

	#filename = 'MOV_2150.mp4'
	#video_engine('Video/'+filename)

	files = os.listdir(folder)

	for video in files:
		'''
		while(threadcount>9):
			pass
		print("Running file "+folder+"/"+video)
		threadcount+=1
		try:
			thread.start_new_thread(video_engine, (folder,video))
		except:
			print "Error: Unable to start thread"
		'''
		print("Running file "+folder+"/"+video)
		video_engine(folder,video)

if __name__ == '__main__':
	main()