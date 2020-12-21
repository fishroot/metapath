import numpy as np
#import numpy.random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
##
### Generate some test data
#x, y = np.random.randn(2, 100)
###heatmap, xedges, yedges = np.histogram2d(x, y, bins=(5, 8))
##
#data = np.random.randn(100, 20)
#corr = np.corrcoef(data)
##
data = np.array([[1, 2, 3], [2, 3, 4], [1, 2, 4]])
axis = ['a', 'b' ,'c']
#rdata = np.random.uniform(0, 1, (10, 100))
corr = np.corrcoef(data)
###corr = np.cov(data)
##
##print data
##print corr
#print data.shape

#print corr.shape

#r = np.random.randn(100,3)
#heatmap, edges = np.histogramdd(r, bins = (5, 8, 4))

#heatmap, xedges, yedges = np.histogram2d(x, y, bins=(50, 50))
#extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

#plt.clf()
fig = plt.figure()
fig.subplots_adjust(bottom = 0.2)
ax = fig.add_subplot(111)
ax.set_yticks((2,5,7))

ax.imshow(corr, interpolation="nearest")
ax.grid(True)

formatter = ticker.FormatStrFormatter(axis[int('%s') - 1])
ax.yaxis.set_major_formatter(formatter)

i = 0
for tick in ax.yaxis.get_major_ticks():
    tick.label1 = axis[i]
    i += 1
    tick.label1On = True
    #tick.label2On = True
    #tick.label2.set_color('green')
    
plt.show()

##import numpy as NP
##from matplotlib import pyplot as PLT
##from matplotlib import cm as CM
##
##A = NP.random.randint(10, 100, 100).reshape(10, 10)
##mask =  NP.tri(A.shape[0], k=-1)
##A = NP.ma.array(A, mask=mask) # mask out the lower triangle
##fig = PLT.figure()
##ax1 = fig.add_subplot(111)
##cmap = CM.get_cmap('jet', 10) # jet doesn't have white color
##cmap.set_bad('w') # default value is 'k'
##ax1.imshow(A, interpolation="nearest", cmap=cmap)
##ax1.grid(True)
##PLT.show()
