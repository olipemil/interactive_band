
import matplotlib.pyplot as plt
from bandstructure_widget import Widget

#optional input
characterSi = ['Si s','Si pz','Si py','Si px','Si s','Si pz','Si py','Si px']
kpathSi = [[0.5, 0.5, 0.5],[0.0, 0.0, 0.0], [0.5, 0, 0.5]]#, [3/8,3/8,3/4] ,[0.0, 0.0, 0.0]]
k_labelSi = [r'$L$',r'$\Gamma$',r'$X$']#,r'$K$',r'$\Gamma$']

#required input
wanDirect = "silicon_interact"
wanTag = "wannier90"
numWanSi = 8
#initialize the widget
model = Widget(wanDirect,wanTag,numWanSi,characterSi,kpathSi,k_labelSi)

fig = model.plotWidget()

plt.show()
plt.close()
