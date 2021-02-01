import wmi
import numpy as np
import matplotlib.pyplot as plt


w = wmi.WMI(namespace="root\\OpenHardwareMonitor")

def temperature():
    temperature_infos = w.Sensor()
    for sensor in temperature_infos:
        if sensor.SensorType==u'Temperature':
            if sensor.Name=='CPU Core #1':
                CPU1=sensor.Value
            if sensor.Name=='CPU Core #2':
                CPU2=sensor.Value
            if sensor.Name=='CPU Core #3':
                CPU3=sensor.Value
            if sensor.Name=='CPU Core #4':
                CPU4=sensor.Value
    return CPU1,CPU2,CPU3,CPU4

def coreload():
    temperature_infos = w.Sensor()
    for sensor in temperature_infos:
        if sensor.SensorType==u'Load':
            if sensor.Name=='CPU Core #1':
                CPU1=sensor.Value
            if sensor.Name=='CPU Core #2':
                CPU2=sensor.Value
            if sensor.Name=='CPU Core #3':
                CPU3=sensor.Value
            if sensor.Name=='CPU Core #4':
                CPU4=sensor.Value
    return CPU1,CPU2,CPU3,CPU4

def power():
    temperature_infos = w.Sensor()
    for sensor in temperature_infos:
        if sensor.SensorType==u'Power':
            if sensor.Name=='CPU Cores':
                CPUP=sensor.Value
    return CPUP

data=[]
dataL=[]
vdata=[]
vdataL=[]
powerC=[]
mydata=[]
n=1000
for i in range(n):
    CPU1,CPU2,CPU3,CPU4=temperature()
    CPUL1,CPUL2,CPUL3,CPUL4=coreload()
    CPUP=power()
    data.append([CPU1,CPU2,CPU3,CPU4])
    dataL.append([CPUL1,CPUL2,CPUL3,CPUL4])
    powerC.append(CPUP)
    a=(CPU1+CPU2+CPU3+CPU4)/4
    vdata.append(a)
    b=(CPUL1+CPUL2+CPUL3+CPUL4)/4
    vdataL.append(b)
    mydata.append([CPUP,b,a])


Y1=[data[i][0] for i in range(n)]
Y2=[data[i][1] for i in range(n)]
Y3=[data[i][2] for i in range(n)]
Y4=[data[i][3] for i in range(n)]
X1=[dataL[i][0] for i in range(n)]
X2=[dataL[i][1] for i in range(n)]
X3=[dataL[i][2] for i in range(n)]
X4=[dataL[i][3] for i in range(n)]
X=[i for i in range(n)]

plt.scatter(X1,Y1, label='Core 1')
plt.scatter(X2,Y2, label='Core 2')
plt.scatter(X3,Y3, label='Core 3')
plt.scatter(X4,Y4, label='Core 4')

plt.legend()
plt.show()

np.savetxt("E:/testtemp.csv", mydata, delimiter=",")

mydata1=np.genfromtxt('E:/testtemp.csv',delimiter=',')

plt.hist(Y1, density=True, bins=5)
plt.show()

from scipy.interpolate import griddata

mydata1=np.genfromtxt('E:/testtemp.csv',delimiter=',')



n=1000
allX=[mydata1[i][0] for i in range(n)] #power
allY=[mydata1[i][1] for i in range(n)] #load percentage
I=[mydata1[i][2] for i in range(n)]  #temperature


    
heatmap, xedges, yedges = np.histogram2d(allX, allY, bins=(50,50))
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

# Plot heatmap
plt.clf()
plt.title('when does my computer spend the time with')
plt.ylabel('load percentage (%)')
plt.xlabel('power /W')
plt.imshow(heatmap, extent=extent)
plt.show()   

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

fig = plt.figure()
ax = fig.gca(projection='3d')
# Plot the 3D surface
ax.plot_surface(allX, allY, I, rstride=8, cstride=8, alpha=0.3)

# Plot projections of the contours for each dimension.  By choosing offsets
# that match the appropriate axes limits, the projected contours will sit on
# the 'walls' of the graph
cset = ax.contourf(allX, allY, I, zdir='z', offset=-100, cmap=cm.coolwarm)
cset = ax.contourf(allX, allY, I, zdir='x', offset=-40, cmap=cm.coolwarm)
cset = ax.contourf(allX, allY, I, zdir='y', offset=40, cmap=cm.coolwarm)

ax.set_xlim(0, 100)
ax.set_ylim(0, )
ax.set_zlim(30, 100)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

