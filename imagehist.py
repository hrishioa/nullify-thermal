import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

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
	for imgno in range(23,68):
		imgname = 'IR_21'+str(imgno)+'.jpg'

		gray = cv2.imread('Video/'+imgname,0)

		cv2.imshow('frame',gray)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		
		#hist = cv2.calcHist([gray],[0],None,[256],[0,256])
		
		plt.hist(gray.ravel(),256,[150,300])

		'''
		color = ('b','g','r')
		for i,col in enumerate(color):
			histr = cv2.calcHist([frame],[i],None,[256],[0,256])
			plt.plot(histr,color = col)
			plt.xlim([0,256])
		'''

		save('Video/'+'IR_21'+str(imgno)+'-hist', ext="png", close=True, verbose=False)
		print "Image "+imgname+" done."

if __name__ == '__main__':
	main()