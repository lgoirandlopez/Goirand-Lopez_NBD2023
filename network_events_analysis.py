# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 17:38:06 2023

@author: goirand-lopez
"""
import numpy as np
import time
import sklearn
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# ----------------- Color settings for display purpose -----------------

class NetworkEventsAnalysis():
    def __init__(self,spk,NE=None,Framerate=None,SliceName='') :
        '''        
        Class to perform the NE Analysis described in Goirand-Lopez et al.2023. 
        
        Parameters
        ----------
        spk : Numpy array (ncell,frames length)
            Binary matrix of calcium onsets.
        NE : List
            List of peak-time of detected network events. Peak-time correspond to the frame where the maximum number 
            of coactive calcium transient onsets were detected within the network events.
        Framerate : Float, optional
            The recording Framerate in frame/s. The default is None.
        SliceName : String, optional
            Name of your data. The default is ''.

        
        '''
        self.spk = spk
        if NE is None :
            print('***************************'+' You did not give a Network events frame date list, you can use the detect_network_event fonction of this class to obtain it '
                  +'***************************')
        else :                
            self.NE = NE
        self.ActiveCellslist = [sum(spk[ii])>0 for ii in range(len(spk))]
        self.Framerate = Framerate
        if self.Framerate is not None :
            self.time = np.arange(self.dim[1])/self.Framerate
        self.SliceName = SliceName
        
    
    @property
    def timeexp(self) :
        '''
        Function to obtain the recording time (in seconds) of your data.

        Returns
        -------
        1D-Ndarray of time scale of the data.
            
        '''
        if self.Framerate is None :
            print('ERROR!!!! You do not give the framerate so the experimental time is not calculable')
        else :
            return self.time
    def AvgMinDistFun(self,x,y) :
        '''
        Funtion to calculate Average minimal distance between every spiking date from 2 signals.
        
        Parameters
        ----------
        x : Numpy array,(lenx,).
            1D-array of spiking date.
        y : Numpy array,(leny,).
            1D-array of spiking date.
    
        Returns
        -------
        Float
            Average minimal temporal distance between all date of x and y.            
        '''
        # We need sparseactivity
        if len(x)*len(y)==0 :
            return np.nan
        delta = abs(x-y.reshape(len(y),1))
        return 0.5*(np.mean(np.min(delta,axis=1))+np.mean(np.min(delta,axis=0)))
    
    def xcov(self,x,y,delay,norm=True) :
        '''
        Calulate the cross covariance between 2 signals. In Marissal et al. 2012 the correlation is more a covariance.
        We obtain the covariance between the first signal and the second 
        
        Parameters
        ----------
        x : Numpy array
            Data we want to have delay to the other data.
        y : Numpy array
            Reference data to have delay.
        delay : Int
            Delay to define the correlation window. The default value is the one entered as time delay of the class 
            (In our case, the delay is linked to the NE window extent which are characteristic of the data).
    
        Returns
        -------
        CCxy : Numpy 1D-array ,(2*delay+1)
            CrossCorrelation along the delay window.    
        '''
        import numpy as np
        if not isinstance(x,np.ndarray) :
            x = np.array(x)
        if not isinstance(y,np.ndarray) :
            y = np.array(y)
        if sum(x)*sum(y)==0 :
            return np.nan*np.ones(2*delay+1)
        mux = np.mean(x)
        sigx = np.std(x)
        muy = np.mean(y)
        sigy = np.std(y)
        ydelayed = [np.convolve(y,delaymattemp)[delay:-delay] for delaymattemp in np.eye(2*delay+1)]
        if norm :
            return np.array([np.sum((x-mux)*(ydel-muy))/(sigx*sigy) for ydel in ydelayed])*1/(len(x))
        else:
            return np.array([np.sum((x-mux)*(ydel-muy))/(sigx*sigy) for ydel in ydelayed])
    
    def xcorr(self,x,y,delay,norm=True) :
        '''
        Calulate the cross correlation between 2 signals. We obtain the delay between the first signal and the second.
        
        Parameters
        ----------
        x : Numpy array
            Data we want to have delay to the other data.
        y : Numpy array
            Reference data to have delay.
        delay : Int
            Delay to define the correlation window. The default value is the one entered as time delay of the class 
            (In our case, the delay is linked to the NE window extent which are characteristic of the data).
    
        Returns
        -------
        CCxy : Numpy 1D-array ,(2*delay+1)
            CrossCorrelation along the delay window.    
        '''
        import numpy as np
        if not isinstance(x,np.ndarray) :
            x = np.array(x)
        if not isinstance(y,np.ndarray) :
            y = np.array(y)
        if sum(x)*sum(y)==0 :
            return np.nan*np.ones(2*delay+1)
        ydelayed = [np.convolve(y,delaymattemp)[delay:-delay] for delaymattemp in np.eye(2*delay+1)]
        if norm : 
            return np.array([np.sum(x*ydel) for ydel in ydelayed])/np.sum(y)
        else :
            return np.array([np.sum(x*ydel) for ydel in ydelayed])
    
    def detection_network_events(self,spk,Windowslength=4,Iterations=1000,optionsDisplay='Frames',
                                 Filename='',FrameRate=None,**kwargplot) :
        '''
        Detection of the Network events as described in Goirand-Lopez et al. 2023. The method is using surrogate data created by shuffling the original spike matrix.
        
        
        Parameters
        ----------
        spk : Numpy array (ncell,frames length)
            Binary matrix of calcium onsets.
        Windowslength : Int, optional
            Extent in frames of the slinding window to calculate Network Events. The default is 4.
        Iterations : Int, optional
            Number of iteration where surrogate data sets are created. The default is 1000.
        optionsDisplay : String, optional
            The time metric for visual purpose only. Can be 'Frames' or 'Seconds'. The default is 'Frames'.
        Filename : String, optional
            Name of the data. The default is ''.
        FrameRate : Float, optional
            The recording framerate to convert frame length to seconds length. Needed only if you want 'Seconds' as optionsDisplay. 
            The default is None.
        **kwargplot : keyword argument for ploting
            Scatter options for the rasterplot.

        Returns
        -------
        NetworkEvents : List
            List of peak-time of detected network events. Peak-time correspond to the frame where the maximum number 
            of coactive calcium transient onsets were detected within the network events.
        NetEvOnsets : List
            List of Network Events onsets.
        NetEvOffsets : List
            List of Network Events offsets.

        '''
        onsets = [np.where(x>0)[0] for x in spk]
        ISIs = [np.diff(x) for x in onsets] # Calculate the InterSpike Intervals
        DistributionNbEv=np.zeros(Iterations,dtype=int)
        msg2print = '-'*10+' {} : LOADING Network Events Detection '.format(Filename)+'-'*10
        print('*'*len(msg2print))
        print(msg2print)
        print('*'*len(msg2print))
        stepwind = np.floor(np.linspace(0,Iterations,10))
        for ii in range(Iterations):
            # -------- # Loading Bar --------
            if ii in stepwind : # D(100*i/np.shape(data1)[0])%10 < 1 :
                step = sum(ii>=stepwind) #int(100*i/np.shape(data1)[0]/10)
                print('|'+'#'*step + ' '*(10-step)+'|') 
            # ----- #1 Shuffle the ISIs -----
            ISIsreshuffled = [np.random.permutation(x) for x in ISIs]
            # ----- #2 Calculate the onsets and ensure all new onsets correspond to a frame date within the frame length -----
            onsetssurr = [np.cumsum(np.concatenate((np.random.randint(np.size(spk,1),size=1),x)))%np.size(spk,1) for x in ISIsreshuffled]
            # -------- #3 Create the binary matrix of surrogate spikes --------
            spksurr = np.zeros((np.size(spk,0),np.size(spk,1)))
            for ind in range(np.size(spk,0)):
                spksurr[ind][onsetssurr[ind]] = np.ones(len(onsetssurr[ind]),dtype=int)
                NbEv = np.convolve(np.sum(spksurr,axis=0),np.ones(Windowslength,dtype=int),mode='same')
                DistributionNbEv[ii] = np.max(NbEv)
        # -------- #4 Determine Threshold --------
        thres = np.quantile(DistributionNbEv,.95)
        if thres <= 1 :
            thres = 2
        base = np.convolve(np.sum(spk,axis=0),np.ones(Windowslength,dtype=int),mode='same')
        base_above_thres = base>thres
        # -------- #5 Detect the Network onsets et offsets --------
        bins = np.where(abs(np.diff(np.concatenate(([0],base_above_thres))))>0)[0] #with the np.diff the onsets will be associated with 1 and offset with -1, all others case are 0
        # Check whether the record begin with a Network Events
        if len(bins)>0:
            if base_above_thres[0] and bins[0]!=0 :
                bins = np.concatenate(([0],bins))
        NetEvOnsets = list(bins[0::2])
        NetEvOffsets = list(bins[1::2])
        # NetworkEvents will be the list of Peak-time corresponding to the frame where the maximum number of calcium transient onsets are coactive
        NetworkEvents = []
        coactv = np.sum(spk,axis=0)
        for ii in range(len(NetEvOnsets)) :
            coactvtemp = coactv[NetEvOnsets[ii]:NetEvOffsets[ii]]
            peakframe = NetEvOnsets[ii] + np.where(coactvtemp==max(coactvtemp))[0]
            if len(peakframe)>1 :
                NetworkEvents.append(round(np.mean(peakframe))) #We take the mean if several peaks 
            else : 
                NetworkEvents.append(peakframe[0])
        # -------- #6 Detect false Network Events --------
        # Here we check if the detected Network Events are space enough (more than half the NE length between the end of a NE and the beginning of the following)
        # Merge the Events when the interval between is below the threshold and the new Network events date will be the more peak corresponding to the frame with the most coactive neurons
        list2delete =  [ii for ii in range(len(NetEvOnsets)-1) if abs(NetEvOffsets[ii]-NetEvOnsets[ii+1])<round(Windowslength/2)]
        nberase = 0
        for ii in list2delete :
            NetEvOnsets.pop(ii+1-nberase)
            NetEvOffsets.pop(ii-nberase)
            if base[NetworkEvents[ii]]>=base[NetworkEvents[ii+1]] :
                NetworkEvents.pop(ii+1-nberase)
            else :
                NetworkEvents.pop(ii-nberase)
            nberase+=1                
        if optionsDisplay.lower() == 'frames' or optionsDisplay.lower() == 'seconds' :
            f = plt.figure(figsize=[12.8,9.6],dpi=300)
            gs = f.add_gridspec(4, 1)
            axrast = f.add_subplot(gs[0:3,0])
            numactv = 0
            if optionsDisplay.lower() == 'frames' :
                x = np.arange(np.shape(spk)[1])
            else :
                x = np.arange(np.shape(spk)[1])/self.Framerate
            yind = np.arange(np.shape(spk)[0])
            for ind in range(np.shape(spk)[0]):
                axrast.scatter(x[spk[ind]!=0],[yind[numactv]]*sum(spk[ind]!=0),marker='o',edgecolors = [0,0,0],
                        color=[0,0,0],**kwargplot)
                numactv+=1
            for ind in range(len(NetworkEvents)-1) :
                axrast.plot([x[NetworkEvents[ind]]]*2,[0,np.size(spk,0)],'--r',linewidth=1)
            axrast.plot([x[NetworkEvents[-1]]]*2,[0,np.size(spk,0)],'--r',linewidth=1,label='Synchrony peak')
            axrast.set_ylabel('# Cells')
            axrast.legend()
            plt.title(Filename)
            axcumsum = f.add_subplot(gs[3,0])
            axcumsum.plot(x,base,'-k',linewidth=.75)
            for ind in range(len(NetworkEvents)-1) :
                netbox_coords = np.array([[x[NetEvOnsets[ind]],0],[x[NetEvOffsets[ind]],0],[x[NetEvOffsets[ind]],np.size(spk,0)],[x[NetEvOnsets[ind]],np.size(spk,0)],[x[NetEvOnsets[ind]],0]])
                axcumsum.add_patch(Polygon(netbox_coords, facecolor=[1,0,0],alpha=.75))
            netbox_coords = np.array([[x[NetEvOnsets[-1]],0],[x[NetEvOffsets[-1]],0],[x[NetEvOffsets[-1]],np.size(spk,0)],[x[NetEvOnsets[-1]],np.size(spk,0)],[x[NetEvOnsets[-1]],0]])
            axcumsum.add_patch(Polygon(netbox_coords, facecolor=[1,0,0],alpha=.75,label='Network Event range'))
            axcumsum.set_xlabel(optionsDisplay)
            axcumsum.set_ylabel('Number of Ca onsets \n within the window')
            axcumsum.legend()
        self.NE = NetworkEvents
        return NetworkEvents,NetEvOnsets,NetEvOffsets

    # ----------------- Data definition Functions -----------------

    def NEwindowfun(self,data,NE,NEmaxlag=7,NEwind='pre',measoption='bin',measwinddim='1D',bothdirectionmeas = False) :
        '''
        Transform the data to allow the clustering analysis as described in Goirand-Lopez et al. 2023 and also others options not described in the article.
        
        Parameters
        ----------
        data : Numpy array (ncell,nframe)
            Binary matrix, 0 if cell unactive, 1 if active.
        NE : List (nNE,)
            List of frame date (here Epileptic Network Events).
        NEmaxlag : Int, optional
            The delay (in frames) defining extent of the window. The default is 7 frames.
        NEwind : String, optional
            Selection of the part of the NE window you want to study,
                NEwind can be 'pre' if you want to check only preNE part, 'post' for postNE part only 
                or 'full' for the whole NEwindow. 
            The default is 'pre'.
        measoption : String, optional
            Selection of the type of measure you want to transform the data.
            If 'bin', the data will be binarize, 1 if activity and 0 if no activity in the window.
            If 'delay', the data will be binarise and then non-null value will be the delay to the NE date.
                If NEwind = 'full', the delay is calculatd from the start of window centered in the NE (1 - 2*NEmaxlag+1).
            If 'date', the data will be the date of the activity in the window, 0 if no activity in the window.
            The default is 'bin'.
        measwinddim : String, optional
            Selection of the dimension of the transformed data.
            If '1D', you will have one measure for each window: window participation (measoption='bin'), 
                delay to NE if activity within the window (measoption='delay') 
                or the date of activity within the window (measoption='date')
            If 'fullwind', you will have measure for each frame of the choosen window : binarized activity (measoption='bin'),
                sparse delay to NE if activity and 0 else (measoption='delay') 
                or sparse date of activity and 0 if not (measoption='date').
            The default is '1D'.
        bothdirectionmeas : bool, optional
            For delay measure option only, the NE window frame will be inverse. Usefull to complete with no reverse to have a non directed measure for the data with cosine metric for example.
    
        Returns
        -------
        Data transformed according to the options.
        '''
        # --------------------- Check Options ---------------------
        measoptionlist = ['bin','delay','date']
        if not any([x.lower()==measoption.lower() for x in measoptionlist]) : # Avoid problème of transformation
            print('Warning : The transformation measure option entered is not valable, the option was set to bin')
            measoption = 'bin'
        measwinddimlist = ['1d','fullwind']
        if not any([x.lower()==measwinddim.lower() for x in measwinddimlist]) : # Avoid problème of transformation
            print('Warning : The transformation dimension option entered is not valable, the option was set to 1D')
            measwinddim = '1D'
        NEwindlist = ['pre','post','full']
        if not any([x.lower()==NEwind.lower() for x in NEwindlist]) : # Avoid problème of transformation
            print('Warning : The window option entered is not valable, the option was set to preNE')    
            NEwind = 'pre'
        
        # --------------------- Print information for users ---------------------
        if measoption.lower() == 'bin' :
            print('-----------------' + 'Threshold research for participation within the window' +'-----------------')
            if measwinddim.lower() == '1d' :
                print('-----------------' + 'Data transformed into binary data of participation, 1 measure for each NE' + '-----------------')
            else :
                print('-----------------' + 'Data transformed into binary activity within the window, measure for each frame of each window' + '-----------------')
        elif measoption.lower() == 'delay' :
            print('-----------------' + 'Threshold research with delay to NE within the windows' +'-----------------')
            print('-----------------' + 'If the window option is full, the delay is calculatd from the start of window centered in the NE (1 - 2*NEmaxlag+1).' + '-----------------')
            if measwinddim.lower() == '1d' :
                print('-----------------' + 'Data transformed into list of delay to each NE. 0 means no activity at the corresponding NE.' + '-----------------')
            else : 
                print('-----------------' + 'Data transformed into concatenation of delay within each NEwindow. 0 means no activity at the corresponding delay.' + '-----------------')
        elif measoption.lower() == 'date' :
            print('-----------------' + 'Threshold research with spikink date within the window' +'-----------------')
            if measwinddim.lower() == '1d' :
                 print('-----------------' + 'Data transformed into list of window spikink date : 0 if no spike within the window, the date instead. 1 measure for each NE' + '-----------------')
            else : 
                 print('-----------------' + 'Data transformed into sparse window spikink date : 0 if no spike within the window, the date instead. Measure for each frame of each window' + '-----------------')
            
        # --------------------- Preparation of data ---------------------
        if NEwind.lower() == 'pre' :
            workwindow = np.array([np.arange(ne-NEmaxlag,ne) for ne in NE])
            workwindownodate = np.array([np.arange(-NEmaxlag,0) for ne in NE])
        elif NEwind.lower() == 'post' :
            workwindow = np.array([np.arange(ne+1,ne+NEmaxlag+1) for ne in NE])
            workwindownodate = np.array([np.arange(1,NEmaxlag+1) for ne in NE])
        elif NEwind.lower() == 'full' :
            workwindow = np.array([np.arange(ne-NEmaxlag,ne+NEmaxlag+1) for ne in NE])
            if not bothdirectionmeas :
                workwindownodate = np.array([np.arange(1,2*NEmaxlag+2) for ne in NE])
            else :
                workwindownodate = 2*NEmaxlag+2-np.array([np.arange(1,2*NEmaxlag+2) for ne in NE])
            
        '''
        Ensure to not be outside the data with our window.
        # If it's the case we replace with a date corresponding to a 0 for every neuron'
        '''
        outsidewindow = (workwindow<0) + (workwindow>np.shape(data)[1]-1)
        workwindow[outsidewindow] = np.where(np.sum(data,axis=0)==0)[0][0]*np.ones(np.sum(outsidewindow))
    
        temp0 = [data[:,newind] for newind in workwindow]
        
        # --------------------- Transformation ---------------------
        if len(workwindow)>0 :
            if measoption.lower() == 'bin' :
                if measwinddim.lower() == '1d' :
                    datatansformed =  np.transpose(np.array([np.sum(temp0[newind],axis=1) for newind in range(len(workwindow))]))
                else :
                    datatansformed =  np.concatenate(temp0,axis=1)
            elif measoption.lower() == 'delay' :
                if measwinddim.lower() == '1d' :
                    datatansformed = np.transpose([np.dot(temp0[ind],workwindownodate[ind]) for ind in range(len(temp0))]) 
                else :
                    datatansformed =  np.concatenate([temp0[ind]*workwindownodate[ind] for ind in range(len(temp0))],axis=1)
            elif measoption.lower() == 'date' :
                if measwinddim.lower() == '1d' :
                    datatansformed = np.transpose([np.dot(temp0[ind],workwindow[ind]) for ind in range(len(temp0))])
                else :
                    datatansformed =  np.concatenate([temp0[ind]*workwindow[ind] for ind in range(len(temp0))],axis=1)
            return datatansformed
        else :
            return np.array([[]])
    
    def thresNEwindshufflefun(self,data,NE,NEmaxlag=7,nsurr=1000,siglevel = .05,NEwind='pre',measoption='bin',measwinddim='1D',bothdirectionmeas = False,**pwdist_kwargs) : 
        '''
        Determine the threshold for a Hierarchical Clustering Algorithm of the data transformed according to the choosen options.
        
        Parameters
        ----------
        data : Numpy array (ncell,nframe)
            Binary matrix, 0 if cell unactive, 1 if active.
        NE : List (nNE,)
            List of frame date (here Epileptic Network Events).
        NEmaxlag : Int, optional
            The delay (in frames) defining extent of the window. The default is 7 frames.
        nsurr : Int, optional
            Number of surrogate data created. The default is 1000.
        siglevel : Float between 0 and 1, optional
            Significance level we want for quantile. The default is .05.
        NEwind : String, optional
            Selection of the part of the NE window you want to study,
            NEwind can be 'pre' if you want to check only preNE part, 'post' for postNE part only 
                or 'full' for the whole NEwindow. The default is 'pre'.
        measoption : String, optional
            Selection of the type of measure you want to transform the data.
            If 'bin', the data will be binarize, 1 if activity and 0 if no activity in the window.
            If 'delay', the data will be binarise and then non-null value will be the delay to the NE date.
                If NEwind = 'full', the delay is calculated from the start of window centered in the NE (1 - 2*NEmaxlag+1).
            If 'date', the data will be the date of the activity in the window, 0 if no activity in the window.
            The default is 'bin'.
        measwinddim : String, optional
            Selection of the dimension of the transformed data.
            If '1D', you will have one measure for each window: window participation (measoption='bin'), 
                delay to NE if activity within the window (measoption='delay') 
                or the date of activity within the window (measoption='date')
            If 'fullwind', you will have measure for each frame of the choosen window : binarized activity (measoption='bin'),
                sparse delay to NE if activity and 0 else (measoption='delay') 
                or sparse date of activity and 0 if not (measoption='date').
            The default is '1D'.
        bothdirectionmeas : bool, optional
            For delay measure option only, 2 NE window will be created one with normal frame and the other with inverse one. 
            The pairwise distance calculated will be the mean pf pw distance obtain with each NE window. 
            The resulting  distance is then unidirected.
        **pwdist_kwargs 
            Keyword arguments to pass to sklearn.metrics.pairwise_distances.
    
        Returns
        -------
        Threshold calculated has the mean of the list of quantile at the significance level.
        '''
        # --------------------- Check Options ---------------------
        measoptionlist = ['bin','delay','date']
        if not any([x.lower()==measoption.lower() for x in measoptionlist]) : # Avoid problème of transformation
            print('Warning : The transformation measure option entered is not valable, the option was set to bin')
            measoption = 'bin'
        measwinddimlist = ['1d','fullwind']
        if not any([x.lower()==measwinddim.lower() for x in measwinddimlist]) : # Avoid problème of transformation
            print('Warning : The transformation dimension option entered is not valable, the option was set to 1D')
            measwinddim = '1D'
        NEwindlist = ['pre','post','full']
        if not any([x.lower()==NEwind.lower() for x in NEwindlist]) : # Avoid problème of transformation
            print('Warning : The window option entered is not valable, the option was set to preNE')    
            NEwind = 'pre'
        
        if NEwind.lower() == 'full' and measoption.lower() == 'bin' and measwinddim.lower() == 'fullwind' :
            print('!!!Threshold research is useless, the shift of surrogate data has no sens in full window and participation only!!!')
            return None
        
        # --------------------- Print information for users ---------------------
        if measoption.lower() == 'bin' :
            print('-----------------' + 'Threshold research for participation within the window' +'-----------------')
            if measwinddim.lower() == '1d' :
                print('-----------------' + 'Data transformed into binary data of participation, 1 measure for each NE' + '-----------------')
            else :
                print('-----------------' + 'Data transformed into binary activity within the window, measure for each frame of each window' + '-----------------')
            bothdirectionmeas = False
        elif measoption.lower() == 'delay' :
            print('-----------------' + 'Threshold research with delay to NE within the windows' +'-----------------')
            print('-----------------' + 'If the window option is full, the delay is calculatd from the start of window centered in the NE (1 - 2*NEmaxlag+1).' + '-----------------')
            if measwinddim.lower() == '1d' :
                print('-----------------' + 'Data transformed into list of delay to each NE. 0 means no activity at the corresponding NE.' + '-----------------')
            else : 
                print('-----------------' + 'Data transformed into concatenation of delay within each NEwindow. 0 means no activity at the corresponding delay.' + '-----------------')
        elif measoption.lower() == 'date' :
            print('-----------------' + 'Threshold research with spikink date within the window' +'-----------------')
            if measwinddim.lower() == '1d' :
                 print('-----------------' + 'Data transformed into list of window spikink date : 0 if no spike within the window, the date instead. 1 measure for each NE' + '-----------------')
            else : 
                 print('-----------------' + 'Data transformed into sparse window spikink date : 0 if no spike within the window, the date instead. Measure for each frame of each window' + '-----------------')
            bothdirectionmeas = False
             
        # --------------------- Preparation of data ---------------------
        NEwindow = np.array([np.arange(ne-NEmaxlag,ne+NEmaxlag+1) for ne in NE])
        if NEwind.lower() == 'pre' :
            workwindow = np.array([np.arange(ne-NEmaxlag,ne) for ne in NE])
            workwindownodate = np.array([np.arange(-NEmaxlag,0) for ne in NE])
            selectedindx = np.arange(0,NEmaxlag)
        elif NEwind.lower() == 'post' :
            workwindow = np.array([np.arange(ne+1,ne+NEmaxlag+1) for ne in NE])
            workwindownodate = np.array([np.arange(1,NEmaxlag+1) for ne in NE])
            selectedindx = np.arange(NEmaxlag+1,2*NEmaxlag+1)
        elif NEwind.lower() == 'full' :
            workwindow = np.array([np.arange(ne-NEmaxlag,ne+NEmaxlag+1) for ne in NE])
            workwindownodate = np.array([np.arange(1,2*NEmaxlag+2) for ne in NE])
            if bothdirectionmeas  :
                workwindownodate2 = 2*NEmaxlag+2 - np.array([np.arange(1,2*NEmaxlag+2) for ne in NE])
                print('#'*10+'Data are measured in both direction for cosine measure'+'#'*10)
            selectedindx = np.arange(0,2*NEmaxlag+1)
        
        outsidewindow = (workwindow<0) + (workwindow>np.shape(data)[1]-1)
        workwindow[outsidewindow] = np.where(np.sum(data,axis=0)==0)[0][0]*np.ones(np.sum(outsidewindow))   
        outsidewindow = (NEwindow<0) + (NEwindow>np.shape(data)[1]-1)
        NEwindow[outsidewindow] = np.where(np.sum(data,axis=0)==0)[0][0]*np.ones(np.sum(outsidewindow))
        
        temp0 = [data[:,newind] for newind in NEwindow]
        val = np.zeros(nsurr)
        # ----- Loading bar for visual purpose -----
        if len(workwindow)>0 :
            msg2print = '-'*10+' {} : LOADING CHA Distance Threshold Research '.format(self.SliceName)+'-'*10
            print('*'*len(msg2print))
            print(msg2print)
            print('*'*len(msg2print))
            stepwind = np.floor(np.linspace(0,nsurr,10))
            for n in range(nsurr) :
                if n in stepwind :
                    step = sum(n>=stepwind)
                    print('|'+'#'*step + ' '*(10-step)+'|') 
                surr = [np.array([np.random.permutation(temp0[ind0][ind1]) for ind1 in range(np.shape(data)[0])]) for ind0 in range(len(NE))]
                surr = [s[:,selectedindx] for s in surr]
                if measoption.lower() == 'bin' :
                    if measwinddim.lower() == '1d' :
                        datatansformed =  np.transpose(np.array([np.sum(surr[newind],axis=1) for newind in range(len(workwindow))]))
                    else :
                        datatansformed =  np.concatenate(surr,axis=1)
                elif measoption.lower() == 'delay' :
                    if measwinddim.lower() == '1d' :
                        # datatansformed = np.transpose([np.dot(surr[ind],workwindownodate[ind]) for ind in range(len(surr))]) 
                        datatansformed = np.transpose([np.dot(surr[ind],workwindownodate[ind])/np.maximum(np.ones(np.shape(data)[0]),np.sum(surr[ind],axis=1)) for ind in range(len(surr))]) 
                        if bothdirectionmeas  :
                             datatansformed2 = np.transpose([np.dot(surr[ind],workwindownodate2[ind])/np.maximum(np.ones(np.shape(data)[0]),np.sum(surr[ind],axis=1)) for ind in range(len(surr))]) 
                    else :
                        datatansformed =  np.concatenate([surr[ind]*workwindownodate[ind] for ind in range(len(surr))],axis=1)
                        if bothdirectionmeas  :
                            datatansformed2 =  np.concatenate([surr[ind]*workwindownodate2[ind] for ind in range(len(surr))],axis=1)
                elif measoption.lower() == 'date' :
                    if measwinddim.lower() == '1d' :
                        datatansformed = np.transpose([np.dot(surr[ind],workwindow[ind]) for ind in range(len(surr))])
                    else :
                        datatansformed =  np.concatenate([surr[ind]*workwindow[ind] for ind in range(len(surr))],axis=1)
                if not bothdirectionmeas  :
                    pdistmatrix = sklearn.metrics.pairwise_distances(datatansformed,**pwdist_kwargs)
                else :
                    pdistmatrix = .5*(sklearn.metrics.pairwise_distances(datatansformed,**pwdist_kwargs)+sklearn.metrics.pairwise_distances(datatansformed2,**pwdist_kwargs))
                # val[n] = np.mean(pdistmatrix) #np.mean(pdistmatrixNEprewind)+np.mean(pdistmatrixsparse)
                val[n] = np.quantile(pdistmatrix,siglevel)
            print('|'+'#'*10+'|') 
            return  np.mean(val) # np.quantile(val,siglevel) #
        else : 
            print('*'*10+' There is no NE '+10*'*')
            return np.nan
        
# ------------------------------ Display function ------------------------------
        
    def colorsme(self,K) :
        '''
        Script to create a k-list of colors. the order of the colors are made such as following colors are distinguishable.
        We have 6 big colors panel we vary (red,yellow,green,cyan,blue,magenta). There is no gray-scale colors.
        
        Parameters
        ----------
        K : Int
            Minimal number of different colors to create.
    
        Returns
        -------
        None.
    
        '''
    # K is the number of color we want to generate
    # We have 6 big colors panel (red,yellow,green,cyan,blue,magenta)
        varcolstep = int(np.ceil(K/6))
        colorscreated = np.zeros((varcolstep*6,3))
        colorscreated[0:varcolstep] = (np.arange(0,1,1/varcolstep).reshape(varcolstep,1))*[0,1,0] + np.array([1,0,0]) + (.5*np.random.rand(varcolstep)).reshape(varcolstep,1)*[0,0,1]
        colorscreated[varcolstep:varcolstep*2] = (-np.arange(0,1,1/varcolstep).reshape(varcolstep,1))*[1,0,0] + np.array([1,1,0]) + (.5*np.random.rand(varcolstep)).reshape(varcolstep,1)*[0,0,1]
        colorscreated[varcolstep*2:varcolstep*3] = (np.arange(0,1,1/varcolstep).reshape(varcolstep,1))*[0,0,1] + np.array([0,1,0]) + (.5*np.random.rand(varcolstep)).reshape(varcolstep,1)*[1,0,0]
        colorscreated[varcolstep*3:varcolstep*4] = (-np.arange(0,1,1/varcolstep).reshape(varcolstep,1))*[0,1,0] + np.array([0,1,1]) + (.5*np.random.rand(varcolstep)).reshape(varcolstep,1)*[1,0,0]
        colorscreated[varcolstep*4:varcolstep*5] = (np.arange(0,1,1/varcolstep).reshape(varcolstep,1))*[1,0,0] + np.array([0,0,1]) + (.5*np.random.rand(varcolstep)).reshape(varcolstep,1)*[0,1,0]
        colorscreated[varcolstep*5:varcolstep*6] = (-np.arange(0,1,1/varcolstep).reshape(varcolstep,1))*[0,0,1] + np.array([1,0,1]) + (.5*np.random.rand(varcolstep)).reshape(varcolstep,1)*[0,1,0]
        # return colorscreated
        return np.concatenate((colorscreated[0::2],colorscreated[1::2]))
    
    def scoreview(self,datascore,title = 'Data score', xlabel = '',ylabel='') :
        '''
        Display the data as a matrix with color scale corresponding to those in matlab.
    
        Parameters
        ----------
        datascore : Numpy-array 
            The data to draw.
        title : String, optional
            Title of the figure. The default is 'Data score'.
        xlabel : String, optional
            Label ox the x-axis. The default is ''.
        ylabel : String, optional
            Label ox the y-axis.. The default is ''.
    
        Returns
        -------
        None.
    
        '''
        f,ax = plt.subplots()
        im=ax.matshow(datascore,aspect='auto')
        plt.colorbar(im)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        plt.title(title)
        # plt.show()
        
    def rasterplot(self,data,x=None,yind=None,colorselem=None,title='',xlabel = '',ylabel='',**kwarg) :
        '''
        Script to display a rasterplot of the data. The data can be colorized by group defined by their corresponding color (list of colors).
        Xvalue can be given if we want for example the real time instead of frames.
        
        Parameters
        ----------
        data : numpy array, 
            The data for yaxis of rasterplot, (ndata,datalen)
        x : numpy array, optional
            The xvalue for the rasterplot. The default is None.
        yind : numpy array, optional
            The yvalue for the rasterplot. The default is None.
        colorselem : List of 3-elements, [ndata,3], optional
            Each data has its associated color. The default is None.
        title : String, optional
            Title of the rasterplot. The default is ''.
        xlabel : String, optional
            Label of x-axis. The default is ''.
        ylabel : String, optional
            Label of x-axis. The default is ''.
        **kwarg : Arguments for scatter function
    
        Returns
        -------
        None.
    
        '''
        if x is None :
            x = np.arange(np.shape(data)[1])
        if yind is None :
            yind = np.arange(np.shape(data)[0])
        if colorselem is None :
            colorselem=[[0,0,0] for x in range(np.shape(data)[0])]
        else :
            if np.shape(colorselem)[0]<np.shape(data)[0] :
                print('Error! The shape of colorselem is not fitting the number of elements')  
        f,ax = plt.subplots()
        numactv = 0
        for ind in range(np.shape(data)[0]):
            ax.scatter(x[data[ind]!=0],[yind[numactv]]*sum(data[ind]!=0),marker='o',edgecolors = [0,0,0],
                    color=colorselem[ind],**kwarg)
            numactv+=1
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.title(title)
            
    def redefineclusterlabels(self,labels,minsizeclust = 3) :
        '''
        This script allow to gather too small clusers in a non clustered group.
    
        Parameters
        ----------
        labels : List, (nelem,)
            List for each element of the corresponding cluster label(number).
        minsizeclust : Int, optional
            Minimal size of a cluster to be defined as a true cluster. The default is 3.
    
        Returns
        -------
        labels2 : List, (nelem,)
            List for each element of the corrected corresponding cluster label(number).
    
        '''
        indxlittleclust = np.unique(labels)[(np.sum(labels.reshape(len(labels),1)==np.unique(labels),axis=0)<minsizeclust)]
        # if len(indxlittleclust)>1 :
        indxclusters = np.unique(labels)[(np.sum(labels.reshape(len(labels),1)==np.unique(labels),axis=0)>(minsizeclust-1))]
        listlittleclust = np.sum(labels.reshape(len(labels),1)==indxlittleclust,axis=1)>0
        labels2 = np.sum((labels.reshape(len(labels),1)==indxclusters)*np.arange(len(indxclusters)),axis=1)
        labels2[listlittleclust] = -1*np.ones(sum(listlittleclust))
        return labels2
    
    # --------------------- Graph theory and cluster construction -----------------------
    def CHA(self,dataforcluster,metric='cosine',linkagechoice='average',threscalcul=True,thres=None,nsurr=1000,siglevel=.05,**cluster_kwarg) :
        '''
        Hierarchical Ascendant Clustering with automatic calculation of threshold as described in Goirand-Lopez et al. 2023

        Parameters
        ----------
        dataforcluster : Numpy array
            Data used to classify, (ncell,nmeas*dimmeas)
        metric : String, optional
            The metric used for distance calculation. The default is 'cosine'.
        linkagechoice : String, optional
            Type of linkage calculated at each step of the hierarchical ascendant clustering . The default is 'average'.
        threscalcul : Bool, optional
            Option to define if the threshold for the CHA is calculated. The default is True
        thres : float, optional
            The threshold for CHA if no calculation. The default value is None.
        nsurr : Int, optional
            Number of surrogate data to determine threshold. The default is 1000.
        siglevel : Float, optional
            Significance level to determine threshold with surrogate data set. Value between 0 and 1. The default is .05.
        **cluster_kwarg : 
            Keyword arguments to pass to sklearn.cluster.AgglomerativeClustering
        Returns
        -------
        Clustering object added to the global object.

        '''
        print('···· Clustering Hierarchical Ascendant selectioned ····')
        self.metric = metric
        self.linkagechoice = linkagechoice
        if threscalcul :
            self.nsurr = nsurr
            self.siglevel = siglevel
            self.distthresclust = self.thresNEwindshufflefun(self.spk,self.NE,NEmaxlag=self.NEmaxlag,
                                                        nsurr=self.nsurr,siglevel=self.siglevel, NEwind=self.NEwind,
                                                        measoption=self.measoption,measwinddim=self.measwinddim,
                                                        bothdirectionmeas=self.bothdirectionmeas, metric=self.metric)
        else :
            if thres is None :
                raise NameError('You need to define a threshold if no calculation')
            self.distthresclust = thres
        print('Distance threshold for cluster is : {:.3f}'.format(self.distthresclust))
        if len(dataforcluster) > 1 :
            if not self.bothdirectionmeas :
                self.predistforclust = sklearn.metrics.pairwise_distances(dataforcluster,metric=self.metric)
            else :
                dataforcluster2 = self.NEwindowfun(self.spk,self.NE,NEmaxlag=self.NEmaxlag,NEwind=self.NEwind,measoption=self.measoption,measwinddim=self.measwinddim,bothdirectionmeas=self.bothdirectionmeas)
                self.predistforclust = .5*(sklearn.metrics.pairwise_distances(dataforcluster,metric=self.metric)+sklearn.metrics.pairwise_distances(dataforcluster2,metric=self.metric))
            
            self.clustering = sklearn.cluster.AgglomerativeClustering(n_clusters=None,distance_threshold=self.distthresclust,
                     affinity='precomputed',linkage=self.linkagechoice).fit(self.predistforclust)
        else : 
            self.predistforclust = np.array([[]])
            self.clustering
    
    def KMeans(self,dataforcluster,**cluster_kwarg) :
        '''
        K-means Clustering

        Parameters
        ----------
        dataforcluster : Numpy array
            Data used to classify, (ncell,nmeas*dimmeas)
        **cluster_kwarg : 
            Keyword arguments to pass to sklearn.cluster.KMeans
        Returns
        -------
        Clustering object added to the global object.

        '''
        self.clustering =  sklearn.cluster.KMeans(**cluster_kwarg).fit(dataforcluster)
        
    def NEwindowfunClusters(self,NEmaxlag=7,NEwind='full',measoption='delay',measwinddim='fullwind',clustermethod='CHA',bothdirectionmeas=False,**cluster_kwarg):
        '''
        Function to lauch clustering analysis, correctly label the clusters and store the details of clustering.

        Parameters
        ----------
        NEmaxlag : Int, optional
            The delay (in frames) defining extent of the window around the NE peak. The default is 7 frames. 
            The extent of the window will be 2*NEmaxlag+1 frames.           
        NEwind : String, optional
            Selection of the part of the NE window you want to study,
            NEwind can be 'pre' if you want to check only preNE part, 'post' for postNE part only 
                or 'full' for the whole NEwindow. The default is 'full'.
        measoption : String, optional
            Selection of the type of measure you want to transform the data.
            If 'bin', the data will be binarize, 1 if activity and 0 if no activity in the window.
            If 'delay', the data will be the delay to the NE date and 0 if no activity in the window. 
                If NEwind = 'full', the delay is calculatd from the start of window centered in the NE (1 - 2*NEmaxlag+1).
            If 'date', the data will be the date of the activity in the window, 0 if no activity in the window.
            The default is 'delay'.
        measwinddim : String, optional
            Selection of the dimension of the transformed data.
            If '1D', you will have one measure for each window: window participation (measoption='bin'), 
                delay to NE if activity within the window (measoption='delay') 
                or the date of activity within the window (measoption='date')
            If 'fullwind', you will have measure for each frame of the choosen window : binarized activity (measoption='bin'),
                sparse delay to NE if activity and 0 else (measoption='delay') 
                or sparse date of activity and 0 if not (measoption='date').
            The default is 'fullwind'.
        clustermethod : String, optional
            Type of clustering method. The default is 'CHA'.
        bothdirectionmeas : bool, optional
            For delay measure option only, 2 NE window will be created one with normal frame and the other with inverse one. 
            The pairwise distance calculated for the surrogate data if done will be the mean pairwise distance obtain with each NE window. 
            The resulting  distance is then unidirected.
        **cluster_kwarg 
            Keyword arguments to pass to the clustering functions.

        Returns
        -------
        The data are clustered, labelled and stored with their details.

        '''
        self.NEmaxlag = NEmaxlag
        self.NEwind = NEwind
        self.measoption = measoption
        self.measwinddim = measwinddim
        self.clustermethod = clustermethod
        self.bothdirectionmeas=bothdirectionmeas
        self.dataforcluster = self.NEwindowfun(self.spk,self.NE,NEmaxlag=self.NEmaxlag,NEwind=self.NEwind,measoption=self.measoption,measwinddim=self.measwinddim)
        self.neverinwindow = np.sum(self.dataforcluster,axis=1)==0
        if self.clustermethod.lower() == 'cha' :
            self.CHA(self.dataforcluster,**cluster_kwarg)
            self.enschoosen = self.clustering.labels_
            self.clustdetails = 'NEWind : {}, Meas : {}, Meas Dim : {}, \n Clustering : {}, Metric : {}, Linkage : {}'.format(self.NEwind,self.measoption,self.measwinddim,self.clustermethod,self.metric,self.linkagechoice)
        elif self.clustermethod.lower() == 'kmeans' :
            self.KMeans(self.dataforcluster,**cluster_kwarg)
            self.enschoosen = self.clustering.labels_
            self.clustdetails = 'NEWind : {}, Meas : {}, Meas Dim : {}, \n Clustering : {}, K : {}'.format(self.NEwind,self.measoption,self.measwinddim,self.clustermethod,self.clustering.n_clusters)
        
        self.enschoosen = self.redefineclusterlabels(self.enschoosen,minsizeclust = 3) #np.zeros(len(ensembletemp))
        self.enschoosen[self.neverinwindow] = -1*np.ones(sum(self.neverinwindow)) #We can't cluster these for the moment
        self.ensindx = np.unique(self.enschoosen)
        self.colors = self.colorsme(len(self.ensindx))
        if len(self.enschoosen[self.enschoosen == -1])>0 :
            self.colors[-1] = [.65]*3
        
    def CrossCorrelation(self,data1,data2,delay) :
        '''
        CrossCorrelation calculated as the second method in Goirad-Lopez et al. 2023 (correlation between all neurons not only significant corr). 
        CrossCorrelation is defined as the timelag score ([0-1]) between pair of element from data1 and data2 for each timelag.
        If one of the data is not two dimensional (for example cross corr between cell activity et Network Events (1dim)),
        the max correlation will just be the correlation, so in this case the result will be the average correlation.
        
        Parameters
        ----------
        data1 : Numpy array
        data2 : Numpy array
        delay : Int
            Delay in frame to define the lag-window [-delay, +delay].

        Returns
        -------
        numpy-1Darray, numpy-1Darray
            If data1 and data2 are 2-dimensional data ,return the average maximal correlation and the average corresponding time-lags.
            If one of the data is 1-dimensionaldata, return the average correlation and the corresponding average time-lags.
        '''
        if 'xcorrdelaytimes' not in dir(self) :
            if self.Framerate is not None :
                self.xcorrdelaytimes = np.arange(delay,delay+1)/self.Framerate
            else : 
                self.xcorrdelaytimes = np.arange(-delay,delay+1)
                
        if len(np.shape(data1))<2 : 
            data1 = data1.reshape(1,len(data1))
        if len(np.shape(data2))<2 : 
            data2 = data2.reshape(1,len(data2))
        CCmatrix = []
        pairCC = []
        msg2print = '-'*10+' {} : LOADING Croscorrelation Calculation '.format(self.SliceName)+'-'*10
        print('*'*len(msg2print))
        print(msg2print)
        print('*'*len(msg2print))
        stepwind = np.floor(np.linspace(0,np.shape(data1)[0],10))
        for i in range(np.shape(data1)[0]) :
            if i in stepwind : # D(100*i/np.shape(data1)[0])%10 < 1 :
                step = sum(i>=stepwind) #int(100*i/np.shape(data1)[0]/10)
                print('|'+'#'*step + ' '*(10-step)+'|') 
            for j in range(np.shape(data2)[0]):
                CC = self.xcorr(data1[i],data2[j],delay)
                pairCC.append([i,j])                   
                CCmatrix.append(CC)
        print('|'+'#'*10+'|')
        self.CCmatrix = np.array(CCmatrix)
        return np.array(CCmatrix),np.array(pairCC)
        
    def MaxCrossCovariance(self,data1,data2,delay) :
        '''
        Normalized CrossCovariance calculated as in Marissal et al. 2012. In the article the cross covariance is called cross correlation. 
        Normalized  CrossCovariance is defined as average normalized  maximal covariance between the datas.
        If one of the data is not two dimensional (for example cross cov between cell activity et Network Events (1dim)),
        the max covariance will just be the covariance, so in this case the result will be the average covariance.
        
        Parameters
        ----------
        data1 : Numpy array
        data2 : Numpy array
        delay : Int
            Delay in frame to define the lag-window [-delay, +delay].

        Returns
        -------
        numpy-1Darray, numpy-1Darray
            If data1 and data2 are 2-dimensional data ,return the average maximal covariance and the average corresponding time-lags.
            If one of the data is 1-dimensionaldata, return the average covariance and the corresponding average time-lags.
        '''
        if 'xcovdelaytimes' not in dir(self) :
            if self.Framerate is not None :
                self.xcovdelaytimes = np.arange(delay,delay+1)/self.Framerate
            else : 
                self.xcorrdelaytimes = np.arange(-delay,delay+1)
                
        datas_2D = True
        if len(np.shape(data1))<2 : 
            data1 = data1.reshape(1,len(data1))
            datas_2D = False
        if len(np.shape(data2))<2 : 
            data2 = data2.reshape(1,len(data2))
            datas_2D = False
        Covmatrix = np.zeros((np.shape(data1)[0],np.shape(data2)[0]))
        taucovmatrix = np.zeros((np.shape(data1)[0],np.shape(data2)[0]))
        msg2print = '-'*10+' {} : LOADING CrossCovariance Calculation '.format(self.SliceName)+'-'*10
        print('*'*len(msg2print))
        print(msg2print)
        print('*'*len(msg2print))
        stepwind = np.floor(np.linspace(0,np.shape(data1)[0],10))
        for i in range(np.shape(data1)[0]) :
            if i in stepwind : # D(100*i/np.shape(data1)[0])%10 < 1 :
                step = sum(i>=stepwind) #int(100*i/np.shape(data1)[0]/10)
                print('|'+'#'*step + ' '*(10-step)+'|') 
            for j in range(np.shape(data2)[0]):
                CC = self.xcov(data1[i],data2[j],delay)
                if datas_2D :
                    maxCC = max(CC)
                    if not np.isnan(maxCC):
                        taucovmatrix[i,j] = np.mean(self.xcovdelaytimes[abs(CC-maxCC)<1e-5]) #test instead of == to avoid false inequality due to imprecision
                    else :
                        taucovmatrix[i,j] = np.nan
                    Covmatrix[i,j] = maxCC
                else : 
                    if not np.isnan(CC).any():
                        CC[CC<0] = np.zeros(sum(CC<0))
                        if sum(CC>0)>0 :
                            Covmatrix[i,j] = np.mean(CC[CC>0])
                        else :
                            Covmatrix[i,j] = 0
                        taucovmatrix[i,j] = np.sum(CC*self.xcovdelaytimes)
                    else : 
                        Covmatrix[i,j] = np.nan
                        taucovmatrix[i,j] = np.nan
        print('|'+'#'*10+'|')
        self.Covmatrix = Covmatrix
        self.taucovmatrix = taucovmatrix
        return np.nanmean(Covmatrix,axis = 1), np.nanmean(taucovmatrix,axis = 1)
    
    def sigxcorr(self,data1,data2,delay,nsurr=1000,i=0,j=0) :
        '''
        Significant CrossCorrelation between two data.
        Significant CrossCorrelation calculated as in Marissal et al. 2018. 
        Significant correlation between two neurons occurred when the number of coincidences exceeded a 
        chance threshold at any lag.
        Parameters
        ----------
        data1 from first neuron
        data2 from second neuron
        delay : Int
            Delay in frame to define the lag-window [-delay, +delay].
        nsurr : Int, optional
            Number of surrogate data to generate to calculate threshold. The default is 1000.

        Returns
        -------
        numpy-1Darray, numpy-1Darray
            Return the significant correlation between the two data and the corresponding time-lags.

        '''
        if not (i==j and (data1 == data2).all()) :
            if 'xcorrdelaytimes' not in dir(self) :
                if self.Framerate is not None :
                    self.xcorrdelaytimes = np.arange(-delay,delay+1)/self.Framerate
                else : 
                    self.xcorrdelaytimes = np.arange(-delay,delay+1)
            surr1 = [np.random.permutation(data1) for ind in range(nsurr)]
            surr2 = [np.random.permutation(data2) for ind in range(nsurr)]
            surrcorr = np.zeros((nsurr,2*delay+1))
            for ind in range(nsurr) : 
                surrcorr[ind] = self.xcorr(surr1[ind],surr2[ind],delay)
            thressigcorr = np.max(surrcorr,axis=0)
            sigxcorr = self.xcorr(data1,data2,delay) # Start with the real xcorr value
            sigtimelag = self.xcorrdelaytimes
            if np.sum(sigxcorr>thressigcorr)>0 : 
                        self.sigxcorrall.append(sigxcorr[sigxcorr>thressigcorr])
                        self.sigtimelagall.append(sigtimelag[sigxcorr>thressigcorr])
                        [self.sigtimelagall1D.append(x) for x in sigtimelag[sigxcorr>thressigcorr]]
                        self.sigpaircorrall.append([[i,j] for x in range(np.sum([sigxcorr>thressigcorr]))])
        # return sigxcorr[sigxcorr>thressigcorr], sigtimelag[sigxcorr>thressigcorr]
    
    def SigCorrelation(self,data1,data2,delay,nsurr = 1000) :
        '''
        Significant CrossCorrelation between every data within data1 and data2.
        Significant CrossCorrelation calculated as in Marissal et al. 2018. 
        Significant correlation between two neurons occurred when the number of coincidences exceeded a 
        chance threshold at any lag.
        
        The script find every pair of significant correlated data , their corresponding correlation and associated time-lag.
        The script also compute all significant time-lag within a 1D-list

        Parameters
        ----------
        data1 : numpy array, (ndata,datalen)
        data2 : numpy array, (ndata,datalen)
        delay :  Int
            Delay in frame to define the lag-window [-delay, +delay].
        nsurr : Int, optional
            Number of surrogate data to generate to calculate threshold. The default is 1000.

        Returns
        -------
        None.

        '''
        if 'xcorrdelaytimes' not in dir(self) :
            if self.Framerate is not None :
                self.xcorrdelaytimes = np.arange(-delay,delay+1)/self.Framerate
            else : 
                self.xcorrdelaytimes = np.arange(-delay,delay+1)
                
        if len(np.shape(data1))<2 : 
            data1 = data1.reshape(1,len(data1))
        if len(np.shape(data2))<2 : 
            data2 = data2.reshape(1,len(data2))
        self.sigpaircorrall = []
        self.sigxcorrall = []
        self.sigtimelagall = []
        self.sigtimelagall1D = []
        start_time = time.time() 
        msg2print = '-'*10+' {} : LOADING Significant Croscorrelation Calculation '.format(self.SliceName)+'-'*10
        print('*'*len(msg2print))
        print(msg2print)
        print('*'*len(msg2print))
        stepwind = np.floor(np.linspace(0,np.shape(data1)[0],10))
        for i in range(np.shape(data1)[0]) :
            if i in stepwind : 
                step = sum(i>=stepwind) 
                print('|'+'#'*step + ' '*(10-step)+'|') 
            if sum(data1[i])==0 :
                continue
            [self.sigxcorr(data1[i],data2[j],delay,nsurr=nsurr,i=i,j=j) for j in range(len(data2))]
        print('|'+'#'*10+'|')   
        end_time = time.time() 
        print(f"{end_time - start_time:0.4f} seconds")
            