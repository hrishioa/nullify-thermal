import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import sys
from subprocess import call
import thread
import pyqtgraph as pg 
import pyqtgraph.exporters
from PyQt4 import QtGui
import PyQt4
import time
import subprocess
import scipy
from mpl_toolkits.mplot3d import Axes3D

print(sys.version)

np.set_printoptions(threshold=np.inf)

pgm_start_time = time.time()

threadcount=0
temp_max = 150 #maximum temperature setting for camera
temp_min = 0 #minimum temperature setting for camera
threshold_temp = 140 #threshold temperature in Celcius
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
	
	# Extract the directory and 	filename from the given path
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

def video_engine(folder,filename, startframe=0, endframe=0):
	global pgm_start_time

	returnval = 0;

	cpuplot=False

	global threadcount

	print("\nStarting Video engine for file %s." % (filename))

	video = cv2.VideoCapture(folder+'/'+filename)

	counter = 0
	
	data = []

	video_start_time = time.time()

	if not os.path.exists("X:/HrishiOlickel/Desktop/Thermal/Histograms/%s" % (filename)):
		os.makedirs("X:/HrishiOlickel/Desktop/Thermal/Histograms/%s" % (filename))

	while(video.isOpened()):

		app = pg.mkQApp()

		start_time = time.time()

		counter+=1

		ret, frame = video.read()

		if ret==False:
			returnval=1
			break

		# if(counter%10!=0):
		# 	continue

		if(endframe!=0):
			if(counter<startframe):
				print ("Skipping frame %d"%counter)
				continue
			if(counter>endframe):
				break

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
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
			plot = pg.plot(data)
			plot.setTitle(stats)
			exporter = pg.exporters.ImageExporter(plot.plotItem)
			exporter.parameters()['width'] = 750
			exporter.export('X:/HrishiOlickel/Desktop/Thermal/Histograms/%s/histogram#%d.png' % (filename,counter))
			#plot.close()
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

		delstats = "<br /><br />Pointer: %d \nMean : %.2f \nMedian: %d \nMin : %d \nMax : %d \n<br />SD: %.2f \nMax-Mean: %d" % (pointer,np.mean(delta),np.median(delta),np.min(delta),np.max(delta),np.std(delta),(np.max(delta)-np.mean(delta)))

		if(cpuplot==False):
			plot = pg.plot(delta)
			exporter = pg.exporters.ImageExporter(plot.plotItem)
			exporter.parameters()['width'] = 750
			exporter.export('X:/HrishiOlickel/Desktop/Thermal/Histograms/%s/del_histogram#%d.png' % (filename,counter))
			#plot.close()
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

		app.closeAllWindows()
		del plot

		if(counter%20==0):
			app.exit()

		print("%s: Frame %d done in %f seconds. Time elapsed: Video - %f s, Total - %f s." % (filename,counter,time.time()-start_time,time.time()-video_start_time,time.time()-pgm_start_time))

	video.release()
	cv2.destroyAllWindows()

	if(endframe==0):
		#Run ffmpeg to create video
		print("Generating video")
		call("ffmpeg -framerate 30 -y -i Histograms/"+filename+"/del_histogram#%d.png del_hist"+filename,shell=True)
		call("ffmpeg -framerate 30 -y -i Histograms/"+filename+"/histogram#%d.png hist"+filename,shell=True)

	print("Completed file %s in %f seconds. Returning %d" % (filename,time.time()-video_start_time,returnval))

	return returnval

def fire_analysis_engine(folder,filename):
	
	video = cv2.VideoCapture(folder+'/'+filename)

	pointer = int((hist_samples/(temp_max-temp_min))*(threshold_temp-temp_min))

	counter = 0
	
	background = None

	ret, frame = video.read()

	areas = []
	deltaXY = []

	while(ret!=False):

		posX = 0
		posY = 0

		if(counter==0):
			counter+=1
			background_mask = cv2.bitwise_not(cv2.inRange(frame, np.array([245,245,245]), np.array([255,255,255])))
			continue

		original = frame

		frame = cv2.bitwise_and(frame,frame,mask=background_mask)

		frame = frame[32:144,0:277]

		#Start Processing
		framehist = np.histogram(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).ravel(),bins=256,range=(0,256))
		framehist = (framehist[0][pointer:],framehist[1][pointer:])

		maxval = np.max(framehist[0])
		maxpixelv = 255

		for i in range(0,len(framehist[0])):
			if(framehist[0][i]==maxval):
				print "Frame %d: Pointer -  %d, Maximum Value is %d at value %d" % (counter, pointer, maxval, framehist[1][i])
				maxpixelv = framehist[1][i]

		mask = cv2.inRange(frame, np.array([maxpixelv-100,maxpixelv-100,maxpixelv-100]), np.array([maxpixelv+30,maxpixelv+30,maxpixelv+30]))

		#Draw contours
		image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		contourframe = cv2.drawContours(original,contours,-1,(0,255,0),3)

		if contours:
			moments = cv2.moments(contours[0])
			area = cv2.contourArea(contours[0])
			print "Area: %f" % (area)
			areas.append(area)
			if(int(moments['m00'])!=0):
				newposX = int(moments['m10']/moments['m00'])
				newposY = int(moments['m01']/moments['m00'])
				deltaXY.append(np.sqrt(((newposX-posX)**2)+((newposY-posY)**2)))
				posX = newposX
				posY = newposY
				print "Centre: %d x %d" % (posX,posY)
			else:
				deltaXY.append(0)
		else:
			areas.append(0)
			deltaXY.append(0)


		cv2.imshow('frame',frame)
		cv2.imshow('mask',mask)
		cv2.imshow('contours',contourframe)

		k = cv2.waitKey(1) & 0xFF

		if k==27:
			break

		counter+=1

		ret, frame = video.read()

		if(counter>1000):
			break

	video.release()

	cv2.destroyAllWindows()

	fftsamplesize = 180

	X = []
	Y = []
	Z = []

	for i in range (0,int(len(areas)/fftsamplesize)+1):
		sample = areas[(i*fftsamplesize):((i*fftsamplesize)+fftsamplesize)]
		if(len(sample)==0):
			continue
		tmpx, tmpz = plotSpectrum(sample,30,silent=True)
		print("Length_sample %d: %dx%dx%d" % (i,len(tmpx),len(tmpz),len(tmpz)))
		for j in range(0,len(tmpz)):
			Y.append(i/10)
			X.append(tmpx[j])
			Z.append(tmpz[j])

	fig = plt.figure(1)
	# ax = fig.gca(projection='3d')
	ax = fig.add_subplot(111, projection='3d')

	print "Size: %dx%dx%d" % (len(X),len(Y),len(Z))

	ax.scatter(X,Y,Z,c='r', marker='.')
	# ax.plot_surface(X,Y,Z,rstride=1,cstride=1)
	ax.set_xlabel('Frequency')
	ax.set_ylabel('Run')
	ax.set_zlabel('FFTArea')

	plt.show()

	plt.figure(2)
	#Compute fft
	plt.subplot(221)
	plt.plot(areas)
	plt.ylabel('Area')
	plt.subplot(222)
	plt.plot(deltaXY)
	plt.ylabel('deltaPos')
	plt.subplot(223)
	plotSpectrum(areas,30,ylabel='AreaFFT')
	plt.subplot(224)
	plotSpectrum(deltaXY,30,ylabel='deltaXYFFT')
	plt.show()

def plotSpectrum(y,Fs, ylabel='y',silent=False):
	"""
	Plots a Single-Sided Amplitude Spectrum of y(t)
	"""
	n = len(y) # length of the signal
	k = np.arange(n)
	T = n/Fs
	frq = k/T # two sides frequency range
	frq = frq[range(n/2)] # one side frequency range

	Y = scipy.fft(y)/n # fft computing and normalization
	Y = Y[range(n/2)]

	if(silent==False):
		plt.plot(frq[5:],abs(Y)[5:],'r') # plotting the spectrum
		plt.xlabel('Freq (Hz)')
		plt.ylabel(ylabel)
	return frq,Y

def multicore_vengine(folder, filename):
	print "Running multicore engine. Spinning cylinders..."

	no_of_processes = 16

	processes = []

	frames_per_process = 50

	retval=0

	frame=0

	active_processes=0

	while(retval==0):
		while(active_processes<no_of_processes):
			f = os.tmpfile()
			processes.append(subprocess.Popen(['python','X:/HrishiOlickel/Desktop/Thermal/proto.py',folder,filename,str(frame),str(frames_per_process)],stdout=f))
			frame+=frames_per_process
			active_processes+=1

		for p in processes:
			print "Opened processes %d, frame %d, Time %f s. Waiting for process end..."%(len(processes),frame,time.time()-pgm_start_time)
			p.wait()
			if(int(p.returncode)!=0):
				for q in processes:
					q.wait()
				print "Completed."
				call("ffmpeg -framerate 30 -y -i Histograms/"+filename+"/del_histogram#%d.png del_hist"+filename,shell=True)
				call("ffmpeg -framerate 30 -y -i Histograms/"+filename+"/histogram#%d.png hist"+filename,shell=True)
				return None
			f = os.tmpfile()
			processes.append(subprocess.Popen(['python','X:/HrishiOlickel/Desktop/Thermal/proto.py',folder,filename,str(frame),str(frames_per_process)],stdout=f))
			frame+=frames_per_process

		# retval = int(subprocess.call(["python","X:/HrishiOlickel/Desktop/Thermal/proto.py",folder,filename,str(frame),str(frame+frames_per_process)]))


	print("Completed.")

def main():
	global pgm_start_time
	global threadcount

	print("Program Running...")

	if(len(sys.argv)>1):
		print "Arguments detected. Running in subprocess mode."

		sys.exit(video_engine(sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4])))

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
		#multicore_vengine(folder,video)
		fire_analysis_engine(folder,video)

		print("Completed in %f seconds."%(time.time()-pgm_start_time))

if __name__ == '__main__':
	main()