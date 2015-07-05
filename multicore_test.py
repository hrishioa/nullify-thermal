import subprocess
import time

start_time = time.time()

subprocess.Popen(['python','X:/HrishiOlickel/Desktop/Thermal/proto.py','X:/HrishiOlickel/Desktop/Thermal/Video2','MOV_2148.mp4',str(0),str(25)])
subprocess.Popen(['python','X:/HrishiOlickel/Desktop/Thermal/proto.py','X:/HrishiOlickel/Desktop/Thermal/Video2','MOV_2148.mp4',str(25),str(50)])
subprocess.Popen(['python','X:/HrishiOlickel/Desktop/Thermal/proto.py','X:/HrishiOlickel/Desktop/Thermal/Video2','MOV_2148.mp4',str(50),str(75)])
subprocess.Popen(['python','X:/HrishiOlickel/Desktop/Thermal/proto.py','X:/HrishiOlickel/Desktop/Thermal/Video2','MOV_2148.mp4',str(75),str(100)])

print("Completed in %f seconds."%(time.time()-start_time))