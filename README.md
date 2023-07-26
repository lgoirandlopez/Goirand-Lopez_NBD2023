# Analysis code for Transient Calcium Onsets matrix with Network Events (data from Calcium Imaging Recording), Goirand-Lopez et al., NBD(2023)
Code analysis associated to the article Goirand-Lopez et al. 2023
This repository contains analysis code introduced and used in from Goirand-Lopez et al., NBD(2023).

## Prerequisites
- [Python3](https://www.python.org/) (v. 3.8 used for this code)
- [NumPy](https://numpy.org/) (v. 1.19.2 used for this code)
- [scikit-learn](https://scikit-learn.org/stable/) (v. 0.23.2 used for this code)
- [matplotlib](https://matplotlib.org/) (v. 3.3.2 used for this code)

## Description
The analysis code is all writen within a class object. 
### Data needed to run the code
Before initializing the analysis you need to have data in a correct format. The data to enter has to be a binary spiking matrix (numpy array) with each line represents the calcium spiking activity of a single cell. At a single frame for a single neuron, a 1 is associated with a transient calcium onsets.

```
            |0 0 0 0 1 0 0 0 0 0 1 0 0 0 1 0 0|
            |0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0| 
spk : #Cell |0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0|
            |0 1 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0|
            |0 0 0 0 1 0 0 0 0 0 1 0 0 0 1 0 0|
            |0 0 1 0 0 0 0 0 1 0 0 0 1 0 0 0 0|
                          # Frame
```

### Initialization of the analysis
To initialize the different functions :
```
from network_events_analysis import NetworkEventsAnalysis
NEAnalysis = NetworkEventsAnalysis(spk,**kwargs)
```
The initialization parameters are defined by the user :
- spk : Numpy array (ncell,frames length). Binary matrix of transient calcium onsets.
- NE : 1D-numpy array, optional (default = None). Network Events dates list If you already have dectected the Network Events.
- Framerate : Float, optional (default = None). The recording framerate in frame/s.
- SliceName : String, optional (default = ''). Name of your data. The default is ''.

If needed you can detect the network events as follow after a first initialization :
```
(NetworkEvents,NEOnsets,NEoffsets) = NEAnalysis.detection_network_events(spk,**kwargs)
```
The detection parameters will be :
- spk : Numpy array (ncell,frames length). Binary matrix of transient calcium onsets.
- Windowslength : Int, optional (default = 4 frames). Extent in frames of the slinding window to calculate Network Events.
- Iterations : Int, optional (default = 1000). Number of iteration where surrogate data sets are created.
- optionsDisplay : String, optional (default = 'Frames'). The time metric for visual purpose only. Can be 'Frames' or 'Seconds'.
- Filename : String, optional (default = ''). Name of your data.
- FrameRate : Float, optional (default = None)
- **kwargplot : keyword argument for ploting. Scatter options for the rasterplot.

As result you will have the network event date (calculated as described in the method of Goirand-Lopez et al., NBD(2023)) stored in NetworkEvents and the onsets and offsets of the detected network events will be respectively stored in NEOnsets and NEOffsets.

### Example of analysis
Once the network events are detected (After using the detection_network_events function or enter the NE dates during the NetworkEventsAnalysis initialization) you can run the clustering analysis :
```
NEAnalysis.NEwindowfunClusters(NEmaxlag=7,**kwargs)
```
This code lauches clustering analysis with several user choosen parameters, correctly labels the clusters and stores the details of clustering.
Parameters : 
- NEmaxlag : Int, optional (default = 7 frames). The delay (in frames) defining extent of the window around the NE peak. The extent of the window will be 2*NEmaxlag+1 frames.
- NEwind : String, optional (default = 'full'). Selection of the part of the NE window you want to study, NEwind can be 'pre' if you want to check only preNE part, 'post' for postNE part only or 'full' for the whole NEwindow. The default is 'full'.
- measoption : String, optional (default = 'delay'). Selection of the type of measure you want to transform the data.
  - If 'bin', the data will be binarize, 1 if activity and 0 if no activity in the window.
  - If 'delay', the data will be the delay to the NE date and 0 if no activity in the window. 
  - If NEwind = 'full', the delay is calculatd from the start of window centered in the NE (1 - 2*NEmaxlag+1).
  - If 'date', the data will be the date of the activity in the window, 0 if no activity in the window.
- measwinddim : String, optional (default = 'fullwind'). Selection of the dimension of the transformed data.
  - If '1D', you will have one measure for each window: window participation (measoption='bin'), 
                delay to NE if activity within the window (measoption='delay') 
                or the date of activity within the window (measoption='date')
  - If 'fullwind', you will have measure for each frame of the choosen window : binarized activity (measoption='bin'),
                sparse delay to NE if activity and 0 else (measoption='delay') 
                or sparse date of activity and 0 if not (measoption='date').
- clustermethod : String, optional (default = 'CHA'). Type of clustering method.
- bothdirectionmeas : Bool, optional (default = False). For delay measure option only, 2 NE window will be created one with normal frame and the other with inverse one. 
            The pairwise distance calculated for the surrogate data if done will be the mean pairwise distance obtain with each NE window. 
            The resulting  distance is then unidirected.
- **cluster_kwarg : Keyword arguments to pass to the clustering functions.
  
The following example will run the code for a Hierarchical Clustering Algorithm associated with cosine distance calculated as introduced in the article.
The options here is to stop the CHA with a threshold calculated with a surrogate data method (500 iterations) at a significance level of 5%.
```
NEAnalysis.NEwindowfunClusters(NEmaxlag=7,NEwind='full',measoption='delay',measwinddim='fullwind',clustermethod='CHA',
bothdirectionmeas=True,metric='cosine',linkagechoice='average',threscalcul=True,nsurr=500,siglevel=.05)
```
            
You can also obtain the significant correlation :
```
NEspk = np.zeros(np.shape(NEAnalysis.spk)[1])
NEspk[NEAnalysis.NE] = np.ones(len(NEAnalysis.NE))
NEAnalysis.SigCorrelation(NEAnalysis.spk,NEspk,NEAnalysis.NEmaxlag,nsurr=1000) 
```
Here the parameters of the significant correlation are : 
- spk : Numpy array (ncell,frames length). Binary matrix of transient calcium onsets.
- NEspk : Numpy 1D-array (1,frames length). Binary matrix of network events, 1 at the frame of network event peak 0 otherwise.
- NEmaxlag : Int, optional (default = 7 frames). The delay (in frames) defining extent of the window around the NE peak. The extent of the window will be 2*NEmaxlag+1 frames.
- nsurr : Int, optional (default = 1000). Number of surrogate data to generate to calculate threshold.

## Authors
lucas.goirand-lopez@inserm.fr

## License
This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but without any warranty; without even the implied warranty of merchantability or fitness for a particular purpose. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.

