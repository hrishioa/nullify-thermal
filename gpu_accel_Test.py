import pyqtgraph as pg 
import pyqtgraph.exporters
import numpy as np
from PySide import QtGui

q = pg.mkQApp()

x = np.array([3,4,0,8,3,7,3,3,5,0,4,5,7,5,4,2,1,0
,3,5,4,9,8,0,6,4,15,17,32,6,13,50,56,39,19,11
,11,3,4,6,6,2,7,1,1,5,7,5,5,10,5,3,4,7
,10,5,7,9,7,1,9,4,6,4,2,3,1,4,8,4,8,11
,6,1,8,4,8,6,7,4,1,7,7,5,5,7,2,9,9,5
,5,5,3,1,7,10,2,11,12,14,1,11,20,11,12,8,7,4
,17,23,22,15,7,9,1,5,11,7,13,14,54,17,81,247,441,294
,194,132,29,28,22])

plt = pg.plot(x)
pg.show()

c = raw_input()

q.closeAllWindows()

c = raw_input()