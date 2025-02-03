"""Import Modules necessary to run"""

import numpy as np 
import pandas as pd
import scipy.linalg as la
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob
#import nfft 
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import time as tm

"""Ideally, we would have some default kwargs here that would be available to any function in faoommfAna. I haven't developed this yet"""

defaultKwargs = {
    "ri": [0,0]
}

"""Put functions that don't need a Simulation object here"""

def phaseToColor(angle, mag = 1): 
    """
    Takes an angle in radians and outputs the array [R, G, B, mag] which can be used to set the color of an object using plt.imshow()
    See the output of the function vizPhaseKey for more information on how this function can be used to express the magnitude and phase of a vector. 
    
    Paramaters: 
    --------------------------------------------------------------------------------------------
    angle: float 
        The angle to be converted to a color
    mag: float (optional)
        The magnitude of the vector to be converted to a color. If blank, default is 1. 
        
    Returns: 
    --------------------------------------------------------------------------------------------
    color: array, 4x1
        The color which can be fed to plt.imshow(); [R, G, B, mag]
    """
    
   # assert(angle>= -np.pi-.001 and angle <= np.pi+.001)
    G = max(min(np.abs(3*angle/np.pi)-1, 1), 0)
    B = max(min(-np.abs(3*(angle+np.pi/3)/np.pi)+2, 1), 0)
    R = max(min(-np.abs(3*(angle-np.pi/3)/np.pi)+2, 1), 0)
    return [R, G, B, mag]

def cart2polar(r): 
    
    """ 
    Converts cartesian coordinates to polar coordinates
    
    Paramaters: 
    --------------------------------------------------------------------------------
    r: array, 2x1
        The x and y coordinate 
        
    Returns: 
    ------------------------------------------------------------------------------
    rPolar: array, 2x1
        [r, phi]
    """
    rad = (r[0]**2+r[1]**2)**(1/2)
    theta = np.arctan2(r[1], r[0])
    return(rad, theta)

def cart2spherical(r): 

    """

    Converts cartesian coordinates to spherical coordinates 

    Paramaters: 
    ------------------------------------------------------------------------------------------
    r: array, 3x1
        [x,y,z]

    Returns: 
    -------------------------------------------------------------------------------------------
    rShperical: array, 3x1
        [pho, theta, phi] where theta is the polar angle and phi is the azimuthal angle
    
    """
    rho = la.norm(r)
    phi = np.arctan2(r[1], r[0])
    if rho != 0: 
        theta = np.arccos(r[2]/rho)
    else: 
        theta = 0
    return [rho, theta, phi]

def fourierTransform(xs, ts, plot = False, xlim = False, ylim = False, returnFreq = False, uniformSpacing = False, returnPhase = False, windowing = None): 
    """
    Performs a Discrete Fourier Transform using the FFT or a quazi-FFT algorithm depending on the spacing of the time points. 
    
    This function will perform a linear transform on the ts and map the range of ts to the interval [0,1]. The function will 
    then perform the (nonuniform or regular) FFT on the xs. Finally, it will preform a linear transformation on the frequencies 
    so that the Fourier Series matches the Fourier Transform for the origional function and not the transformed one. 
    
    The frequency range swept is (1/2)*len(t)/(t[-1]-t[0]) or .5n/delta
    
    Parameters: 
    ------------------------------------------------------------------------------------
    xs: array
        The x coordinates of the function to be transformed 
    ts: array (with same shape as xs)
        The t coordinates of the function to be transformed 
    plot: boolean (optional) 
        If true, a plot of the Fourier transform will be created
    xlim: array, 2x1 (optional)
        If plot is True, will set the x range to be plotted from xlim[0] to xlim[1]
    ylim: array, 2x1 (optional)
        If plot is True, will set the y range to be plotted from ylim[0] to ylim[1]
    returnFreq: boolean (optional)
        If True, the function will return the frequencies as well as the amplitudes of the Fourier Series 
    uniformSpacing: boolean (optional)
        If True, the regular FFT will be used to calculate the Fourier Series 
        If False, a nonuniform FFT algorithm will be used to calculate the fourier series
        ***Currently not used, as the nonuniform FFT performed better and was faster than the uniform FFT***
    returnPhase: boolean (optional) 
        If True, the phase of the FFT is returned in an array 
    
    Returns:
    ----------------------------------------------------------
    X_m: array 
        The fourier series amplitudes 
    X_p: array (if returnPhase == True)
        The fourier series phases
    frs: array (if returnFreq == True)
        The fourier series frequencies 
    """
    
    if (type(xs)!=np.ndarray):
        xs = np.array(xs)
    if (type(ts)!=np.ndarray):
        ts = np.array(ts)
    assert (xs.shape == ts.shape)
    if (len(ts)%2==1): 
        xs = xs[1::]
        ts = ts[1::]
        
    assert len(ts)%2 == 0
    assert len(xs)%2 == 0
#    xps = (xs-xs[0])/(xs[-1]-xs[0])-.5
    
    tps = (ts-ts[0])/(ts[-1]-ts[0]) #Maps ts to the interval [0,1]
    n = len(xs)
    frps = (n/2)* np.linspace(0, 1,int(n/2))
    frs = frps/(ts[-1]-ts[0]) #maps the transformed frequencies back to the appropriate frequencies of the origional function
    
    if (np.all(ts == 0)): 
        raise ValueError("All ts are 0")
     
#    X = nfft.nfft(tps,xs) #Performs the non-uniform FFT 

    if callable(windowing): 
        windowArr = np.array(windowing(len(xs)))
        xs = windowArr * xs 
    elif (windowing == None):
        None
    elif (len(windowing)==len(xs)): 
        xs = windowing * xs 
    elif windowing != None: 
        print("windowing failed")

    X = np.fft.fft(xs)

    X_m = (2/n) *abs(X[0:int(n/2)]) #Enforces that only the positive contributions to the FFT are returned. 

    X_p = np.angle(X[0:int(n/2)])

    if np.isnan(np.sum(X_m)): 
        X_m = np.zeros_like(X_m)
    if np.isnan(np.sum(X_p)): 
        X_m = np.zeros_like(X_p)

    
    if (plot == True): 
        fig1 = plt.figure(figsize=(10,8))
        ax1 = fig1.add_subplot(1,1,1)
        ax1.plot(ts, xs)
        fig3 = plt.figure(figsize=(10,8))
        ax3 = fig3.add_subplot(1,1,1)
        ax3.plot(frs, X_m, label = "[0,1] interval")
        ax3.legend()
        ax3.set_xlabel("frequency (w)")
        ax3.set_ylabel("Fourier amplitude")
        if (xlim != False): 
            ax3.set_xlim(xlim)
        if (ylim != False): 
            ax3.set_ylim(ylim)
    if (returnPhase == True): 
        if (returnFreq == True): 
            return(X_m, X_p, frs)
        else: 
            return(X_m, X_p)
    else: 
        if (returnFreq == True): 
            return(X_m, frs)
        else: 
            return(X_m)

def selectNN(M, coords):
    
    if (len(coords) == 2):       
        x, y = coords
        NN = [[M[x-1][y+1], M[x][y+1], M[x+1][y+1]], 
             [M[x-1][y], M[x][y], M[x+1][y]], 
             [M[x-1][y-1], M[x][y-1], M[x+1][y-1]]]
    if (len(coords) == 3):
        x, y, z = coords
        NN = [[[M[x-1][y+1][z]], [M[x][y+1][z]], [M[x+1][y+1][z]]], 
             [[M[x-1][y][z]], [M[x][y][z]], [M[x+1][y][z]]], 
             [[M[x-1][y-1][z]], [M[x][y-1][z]], [M[x+1][y-1][z]]]]
    
    return np.array(NN)
        
def vizPhaseKey(ax = None, minLabel = "Hue of cell with smallest FFT magnitude", maxLabel = "Hue of cell with largest FFT magnitude", phaseLabel = "Phase Angle $\phi$ for z-component of magnetization", vectorLabel = "FFT vector at a specific frequency for 1 cell"):
    """
    Plots the key for a phase diagram of an FFT given by (PUT COTTECT FUNCTION NAME HERE) EDIT 
    """
    
    if (ax == None): 
        fig1 = plt.figure(figsize=(10,8))
        ax = fig1.add_subplot(1,1,1)
    cells = 100
    xs = np.linspace(-1, 1, cells)
    ys = np.linspace(-1, 1, cells)
    grid = np.zeros((cells, cells, 4))
    for x in range(0, cells): 
        for y in range(0, cells): 
            mag = (xs[x]**2+ys[y]**2)
            if (mag > 1): 
                mag = 0
            grid[x][y] = phaseToColor(np.angle(xs[x]+1j*ys[y]), mag)

    ax.imshow(grid, extent = [-1, 1, -1, 1])
    ax.quiver(0, 0, 1/4, 1/4, scale = 1, label = vectorLabel)
    ax.plot(0,0, "ro", label = minLabel)

    t = np.linspace(0, 2*np.pi, 300)
    xs = np.cos(t)
    ys = np.sin(t)
    ax.plot(xs, ys, "co", label = maxLabel)

    t = np.linspace(0, np.pi/4, 100)
    xs = np.cos(t)/4
    ys = np.sin(t)/4
    ax.plot(xs, ys, label = phaseLabel)
    ax.plot([0,1], [0,0], "k--")

    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

"""
------------------------------------------------------------------------------------------------
Put Simulation (2D) Functions in this block
------------------------------------------------------------------------------------------------
"""

def createVectorFourierImage(Sim_obj, vec_t, uniformTime = False, dims = 3, windowing = None, **kwargs): 
    """
    Internal Function that creates fourier images used in the function createAFourierImage

    Parameters: 
    ----------------------------------------------------------------------------------------------
    Sim_obj: faoommf Simulation object
        The simulation object to be used
    vec_t: array of the form [x][y][dim][t]
        The array to be Fourier Ternsformed at each cell
    uniformTime: boolean
        Leave this as false, true case is untested
    dims: int
        The dimention of the field. For instance, a magnetization field would have dims = 3, while the energy density field would have dims = 1

    Returns: 
    --------------------------------------------------------------------------------------------------
    freq: array [frequencies]
        The array of frequencies associated with the FFT 
    fourierImage: array [x][y][field_dim][omega]
        The magnitude of the FFT at every cell 
    fourierPhaseImage: array [x][y][field_dim][omega]
        The phase angle of the FFT at every cell 
    fourierImage1D: array [omega]
        If dim = 1 for the field to be transformed, this is the sum of fourierImage over all cells
        If dim > 1 for the field to be transformed, this is the sum over all cells of sqrt(F_i^2 + F_j^2 + ...) where F_i are the various components of the field after the FFT has been performed
    fourierAverage: Not sure, not important 
    dimFourierAverage: Not sure, not important 
    """
    
    kwargs = {**defaultKwargs, **kwargs}
    num_x_cells = len(Sim_obj.coords[0])
    num_y_cells = len(Sim_obj.coords[1])
    ri = kwargs["ri"]
    num_t_points = len(Sim_obj.time[ri[0]:ri[1]])
    mt_l = len(vec_t[0][0][0])
    if (len(vec_t[0][0][0][ri[0]:mt_l-ri[1]]) % 2 != 0): 
        ri[1] = ri[1]+1   
    garbage, freq = fourierTransform(vec_t[0][0][0][ri[0]:mt_l-ri[1]], Sim_obj.time[ri[0]:mt_l-ri[1]], plot = False, returnFreq = True, windowing = windowing)

    fourierImage = np.zeros((num_x_cells, num_y_cells, dims, len(freq)))
    fourierPhaseImage = np.zeros_like(fourierImage)
    for x in range(0, num_x_cells): 
        for y in range(0, num_y_cells): 
            for dim in range(0, dims): 
                fourierImage[x][y][dim], fourierPhaseImage[x][y][dim]= fourierTransform(vec_t[x][y][dim][ri[0]:mt_l-ri[1]], Sim_obj.time[ri[0]:mt_l - ri[1]], returnFreq = False, returnPhase = True, windowing = windowing)
    if (dims > 1):
        fourierImage1D = np.zeros((num_x_cells, num_y_cells, len(freq)))
        for x in range(0, num_x_cells): 
            for y in range(0, num_y_cells): 
                fourierImage1D[x][y] = (np.square(fourierImage[x][y][0])+np.square(fourierImage[x][y][1])+np.square(fourierImage[x][y][2]))**(1/2)    
    else: 
        fourierImage1D = fourierImage[:,:,0,:]


    fourierAverage = np.zeros(len(freq))
    for freqN in range(0, len(freq)): 
        for x in range(0, num_x_cells): 
            for y in range(0, num_y_cells): 
                if (dims>1): 
                    fourierAverage[freqN] += fourierImage1D[x][y][freqN]
                else: 
                    fourierAverage[freqN] += fourierImage1D[x][y][freqN] 
    fourierAverage = fourierAverage/(num_x_cells*num_y_cells)
    if (dims > 1):
        dimFourierAverage = np.zeros((dims, len(freq)))
        for dim in range(0, dims): 
            for freqN in range(0, len(freq)):
                for x in range(0, num_x_cells): 
                    for y in range(0, num_y_cells):
                        dimFourierAverage[dim, freqN] += fourierImage[x][y][dim][freqN]
        dimFourierAverage = dimFourierAverage/(num_x_cells*num_y_cells)
    else: 
        dimFourierAverage = fourierAverage
    return (freq, fourierImage, fourierPhaseImage, fourierImage1D, fourierAverage, dimFourierAverage)

def createAFourierImage(Sim_obj, fieldNum, uniformTime = False, windowing = None, **kwargs): 
    """
    Performs a FFT on the simulation using the createVectorFourierImage function and stores the results in the simulation object as attributes. 
    See the documentation for the simulation object for more details about what each of the Fourier Transformed quantities is. 
    
    Paramaters: 
    -------------------------------------------------------------------------------------------------------------
    Sim_obj: faoommf Simulation object
        The simulation object to be used
    fieldNum: int
        The index of the field in the simulation you want to apply the FFT to
    uniformTime: boolean 
        Keep this as False, True case untested (this would change the FFT procedure used) 

    Returns: Nothing 
    
    """
    kwargs = {**defaultKwargs, **kwargs}
    ri = kwargs["ri"]
    Sim_obj.freq, Sim_obj.fourierImage[fieldNum], Sim_obj.fourierPhaseImage[fieldNum], Sim_obj.fourierImage1D[fieldNum], Sim_obj.fourierAverage[fieldNum], Sim_obj.dimFourierAverage[fieldNum] = createVectorFourierImage(Sim_obj, Sim_obj.v_t[fieldNum], ri = ri, uniformTime = uniformTime, dims = Sim_obj.dimsList[fieldNum], windowing = windowing)
    Sim_obj.s_avg_FT[fieldNum] = [None]*Sim_obj.dimsList[fieldNum]
    #for dim in range(0, Sim_obj.dimsList[fieldNum]): 
    #    Sim_obj.s_avg_FT[fieldNum][dim] = fourierTransform(Sim_obj.v_s_avg[fieldNum][dim], Sim_obj.time)

def dataFFT(Sim_obj, dataCat, plot = False, logPlot = False, ax = None, xlim = None, ylim = None): 
    """
    Performs the FFT of one of the OOMMF outputs stored in the Data file. 
    PRESENTLY DOES NOT WORK AS IMPORTING FROM DATA FILE NEEDS TO BE FIXED 
    
    Paramaters: 
    ---------------------------------------------------------------------------------------------
    Sim_obj: faoommf Simulation object
        The simulation object to be used
    dataCat: string
        The name of the data in the data file you want to FFT. Make sure the headers in the data file match the string here
        The headers in the data file should be seperated by spaces and should appear at the top of the file with no comment indication. 
    
    Returns: 
    -----------------------------------------------------------------------------------------------
    ft: array
        The FFT of the data
    freq: array
        The frequencies associated with the FFT
    """
    
    
    ft, freq = fourierTransform(Sim_obj.data[dataCat], Sim_obj.data['time'], returnFreq = True)
    if (plot == True): 
        if (ax == None): 
            fig1 = plt.figure(figsize=(10,8))
            ax = fig1.add_subplot(1,1,1)
        if (logPlot == True): 
            ax.plot(freq, np.log(ft))
        else: 
            ax.plot(freq, ft)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title("Time Fourier Transform of {}".format(dataCat))
        ax.set_xlabel("frequency in Hz")
        ax.set_ylabel("fourier amplitude")
    return ft, freq

def calcB(Sim, d):
    
    """
    
    Calculates the magnetic field a distance d above the film

    Returns an array like Sim.images
    
    """

    coordGrid = np.meshgrid(Sim.coords[0], Sim.coords[1])
    coordGrid = np.array([coordGrid[0], coordGrid[1], np.zeros_like(coordGrid[0])])
    coordGrid = np.einsum('ijk->jki', coordGrid) #[x][y][dim]

    bPlane = np.meshgrid(Sim.coords[0], Sim.coords[1])
    bPlane = np.array([bPlane[0], bPlane[1], np.full_like(bPlane[0], d)])
    bPlane = np.einsum('ijk->jki', bPlane) #[x][y][dim]

    B = np.zeros([len(Sim.time), len(Sim.coords[0]), len(Sim.coords[1]), 3])

    for t in range(0, len(Sim.time)): 

        m = np.array(Sim.images[0][t])
        if (t%10 ==0):
            print(t)

        for x in range(1, len(Sim.coords[0])-1): 
            for y in range(1, len(Sim.coords[1])-1): 

                rGrid = selectNN(coordGrid, [x, y]) - bPlane[x][y]

                r2_inv = (rGrid*rGrid).sum(2) #Checked, works 
                r2_inv = np.divide(1, r2_inv)
                r2_inv = np.einsum('ijk->jki', np.array([r2_inv,r2_inv,r2_inv]))

                rHat = rGrid*np.power(r2_inv, 1/2)

                #if (x == 200 and y == 20): 
                    #print(rHat)

                mNN = selectNN(m, [x,y])

                m_d_r = (mNN*rHat).sum(2)

                m_d_r = np.einsum('ijk->jki', np.array([m_d_r, m_d_r, m_d_r]))

                m_d_r_r = m_d_r*rHat

                Bi = (3*m_d_r_r-mNN)*np.power(r2_inv, 3/2)

                B[t][x][y] = -Bi.sum(0).sum(0)

    return(B)

def addField(Sim, dim = 3): 

    Sim.titles = Sim.titles + [None]
    Sim.dimsList.append(int(dim))
    Sim.images = Sim.images + [None]
    Sim.v_t = Sim.v_t + [None]
    Sim.v_s_avg = Sim.v_s_avg + [None]

    Sim.fourierImage = Sim.fourierImage + [None] #[field][x][y][dim][freq]
    Sim.fourierPhaseImage = Sim.fourierPhaseImage + [None] #[field][x][y][dim][freq]
    Sim.fourierImage1D = Sim.fourierImage1D + [None] #[field][x][y][freq] spatially averaged after FFT
    Sim.fourierAverage = Sim.fourierAverage + [None] #[field][freq]
    Sim.dimFourierAverage = Sim.dimFourierAverage + [None]
    Sim.s_avg_FT = Sim.s_avg_FT + [None]
    Sim.dm = Sim.dm + [None]
    Sim.sphImage = Sim.sphImage + [None]
    Sim.dm_sph = Sim.dm_sph + [None]

def makeBimage(Sim, d, fieldNum = None, title = None): 
    if (not fieldNum): 
        addField(Sim)
        fieldNum = len(Sim.images) - 1
    B = calcB(Sim, d)
    Sim.images[fieldNum] = B
    Sim.v_t[fieldNum] = np.einsum('txyd->xydt', B)
    Sim.v_s_avg[fieldNum] = B.sum(1).sum(1)
    if (not title): 
        title = "B field d = {}nm".format(np.round(d*1e9))
    Sim.titles[fieldNum] = title


def vizFourierModes(Sim_obj, fieldNum, s_avg_before = False, logPlot = False, ax = False, xlim = None, ylim = None, selectAxis = None, GHz = False, getPeaks = True): 
    """
    Plots the Fourier amplitudes of the entire system averaged spatially vs the frequency. 

    Parameters: 
    -----------------------------------------------------------------
    Sim_obj: faoommf Simulation object
        The simulation object to be used
    fieldNum: int
        The index of the field which you want to see the fourier modes
    ax: plotting axis (optional)
        The axis on which to plot the graph 
    xlim: array (2x1) (optional)
        The x limits on the graph 
    ylim: array (2x1) (optional) 
        The y limits on the graph 

    Returns: Nothing
    """

    if (type(Sim_obj.fourierAverage[fieldNum]) != np.ndarray): 
        createAFourierImage(Sim_obj, fieldNum)
    if (s_avg_before == False): 
        if (type(selectAxis) != int): 
            toPlot = Sim_obj.fourierAverage[fieldNum]
        else: 
            toPlot = Sim_obj.dimFourierAverage[fieldNum][selectAxis]


    elif (s_avg_before == True): 
        if (type(selectAxis) != int): 
            print("you should select an axis if the spatial summation is done before the FFT")
        else: 
            toPlot = Sim_obj.s_avg_FT[fieldNum][selectAxis]
    if (ax == False): 
        fig1 = plt.figure(figsize=(10,8))
        ax = fig1.add_subplot(1,1,1)
    if (GHz == True): 
        freq = np.array(Sim_obj.freq)*1e-9
        ax.set_xlabel("Frequency in GHz")
    else: 
        freq = Sim_obj.freq
        ax.set_xlabel("Frequency in Hz")
    if (logPlot == True): 
        ax.plot(freq, np.log(toPlot))
    else: 
        ax.plot(freq, toPlot)

    if (type(selectAxis) != int):
        ax.set_title("Fourier Amplitudes for {}".format(Sim_obj.titles[fieldNum]))
    else: 
        dimTitles = ['x', 'y', 'z']
        ax.set_title("Fourier Amplitudes for the {} component of {}".format(dimTitles[selectAxis], Sim_obj.titles[fieldNum]))
    if (logPlot == True): 
        ax.set_ylabel("Fourier Amplitudes (log scale)")
    else: 
        ax.set_ylabel("Fourier Amplitudes")
    if (xlim != None): 
        ax.set_xlim(xlim)
    if (ylim != None): 
        if (ylim == 0): 
            ax.set_ylim((0, max(toPlot[2::])))
        else: 
            ax.set_ylim(ylim)
    else: 
        ax.set_ylim((min(toPlot), max(toPlot[2::])))

    peaks = []
    if getPeaks == True: 
        for i in range(1, len(toPlot)-1): 

            if (toPlot[i]>toPlot[i-1] and toPlot[i]>toPlot[i+1]): 
                peaks.append([i, Sim_obj.freq[i], toPlot[i]])

        peaks.sort(key = lambda x: x[2], reverse = True)
        Sim_obj.peaks = peaks
        ind = 0
        for peak in peaks: 
            if (logPlot == True): 
                ax.plot(peak[1], np.log(peak[2]),"o", label = "{}, {}GHz".format(ind, peak[1]*10**(-9)))
            else: 
                ax.plot(peak[1], peak[2],"o", label = "{}, {}GHz".format(ind, peak[1]*10**(-9)))
            ind +=1
        ax.legend(title='Peaks', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        return peaks

def vizCellFourierModes(Sim_obj,fieldNum, x, y, ax = False, xlim = False, ylim = False, selectAxis = None): 
    """
    This method plots the Fourier amplitudes of an individual cell vs the frequency. 

    Parameters: 
    -----------------------------------------------------------------
    Sim_obj: faoommf Simulation object
        The simulation object to be used
    fieldNum: int
        The index of the field in the simulation you want vizualize the FFTs 
    x: int
        The x index of the desired cell
    y: int
        The y index of the desired cell
    ax: plotting axis (optional)
        The axis on which to plot the graph 
    xlim: array (2x1) (optional)
        The x limits on the graph 
    ylim: array (2x1) (optional) 
        The y limits on the graph 

    Returns: Nothing
    """
    if (type(Sim_obj.fourierImage1D[fieldNum]) != np.ndarray): 
        createAFourierImage(Sim_obj, fieldNum)
    if (type(selectAxis) != int): 
        toPlot = Sim_obj.fourierImage1D[fieldNum][x,y]
    else: 
        toPlot = Sim_obj.fourierImage[fieldNum][x,y][selectAxis]
    if (ax == False): 
        fig1 = plt.figure(figsize=(10,8))
        ax = fig1.add_subplot(1,1,1)
    ax.plot(Sim_obj.freq, toPlot)
    ax.set_title("Fourier Amplitudes")
    ax.set_xlabel("Frequency in Hz")
    if (type(selectAxis) != int):  
        ax.set_ylabel("Fourier Amplitude of cell {}, {} (x = {}, y = {})".format(x,y, Sim_obj.coords[0][x], Sim_obj.coords[1][y]))
    else: 
        dimTitles = ['x', 'y', 'z']
        ax.set_ylabel("Fourier Amplitude of the {} component of cell {}, {} (x = {}, y = {})".format(dimTitles[selectAxis],x,y, Sim_obj.coords[0][x], Sim_obj.coords[1][y]))
    if (xlim != False): 
        ax.set_xlim(xlim)
    if (ylim != False): 
        ax.set_ylim(ylim)

def vizFrequencyPlot(Sim_obj, fieldNum, frequency, selectAxis = None, ax = False, index = False, cmap = None, blackInactive = False, vmin = 0, vmax = None): 
    """
    Plots the spatial profile of a certain frequency's fourier amplitude as a colormap. As of now, the colors need to be adjusted 
    so that the base noise of fourier series doesn't bias the color gradient 

    Parameters: 
    ---------------------------------------------------------------------------------------------------------
    Sim_obj: faoommf Simulation object
        The simulation object to be used
    fieldNum: int
        The index of the field in the simulation you want vizualize the FFTs 
    frequency: float 
        The frequency that the fourier amplitude spatial profile will be plotted
    ax: plotting axis (optional)
        The axis which the plot will appear on
    index: boolean (optional)
        if True, the frequency will be read as the index of the frequency to plot from Sim_obj.freq
    cmap: string (optional)
        If given, the plot will use the matplotlib colormap indicated by the string. For a list of colormaps and their strings, 
        see https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
    blackInactive: boolean (optional)
        If true, inactive cells will be turned black on the display
    vmin: int (optional)
        If given, this will be the minimum fourier amplitude on the colormap bar 
    vmax: int (optional)
        If given, this will be the maximum fourier amplitude on the colormap bar 

    Returns: Nothing 
    """
    if (index == True): 
        freq_index = frequency
    else: 
        freq_index = 0
        while (frequency >= Sim_obj.freq[freq_index]): 
            freq_index +=1

    if (type(selectAxis) != int): 
        toPlot = Sim_obj.fourierImage1D[fieldNum][:,:,freq_index]
    else: 
        toPlot = Sim_obj.fourierImage[fieldNum][:,:,selectAxis,freq_index]
    if (ax == False): 
        fig1 = plt.figure(figsize=(10,8))
        ax = fig1.add_subplot(1,1,1)

    if (cmap == None): 
        cmap = "RdPu"
    cmp = cm.get_cmap(cmap, 256)
    newColors = cmp(np.linspace(0,1,256))
    if (blackInactive == True): 
        ind = 1
    else: 
        ind = 0 
    newColors[:ind, :] = [0,0,0,1]
    myCmp = ListedColormap(newColors)

    if vmin == None: 
        vmin =  np.min(toPlot[np.nonzero(toPlot)])

    psm = ax.imshow(toPlot[::1][::-1], extent=[Sim_obj.coords[0][0], Sim_obj.coords[0][-1], Sim_obj.coords[1][0], Sim_obj.coords[1][-1]], 
                    cmap = myCmp, vmin = vmin, vmax = vmax)

    plt.colorbar(psm, ax = ax)
    freqDisp = np.round(Sim_obj.freq[freq_index]*10**(-9), 2)
    if (type(selectAxis) != int):
        ax.set_title("Spatial profile of {} GHz".format(freqDisp))
    else: 
        dimTitles = ['x', 'y', 'z']
        ax.set_title("Spatial profile of {} component of {} GHz".format(dimTitles[selectAxis],freqDisp))
    ax.set_xlabel("x (meters)")
    ax.set_ylabel("y (meters)")

def vizFrequencyPhasePlot(Sim_obj, fieldNumPhase, frequency, selectAxisPhase = 2, fieldNumMag = None, selectAxisMag = None, ax = False, index = False, cmap = None, blackInactive = False, vmin = 0, vmax = None, avg_lr = [0], save_data = False, data_title = "Sim"): 
    """
    Plots the spatial profile of a certain frequency's fourier amplitude as a colormap. As of now, the colors need to be adjusted 
    so that the base noise of fourier series doesn't bias the color gradient 

    Parameters: 
    ---------------------------------------------------------------------------------------------------------
    Sim_obj: faoommf Simulation object
        The simulation object to be used
    fieldNumPhase: int
        The index of the field in the simulation you want vizualize the FFTs 
    frequency: float 
        The frequency that the fourier amplitude spatial profile will be plotted
    selectAxisMag: 
        axis, True if want dimentions averaged, int if you want to pick an axis, None to default to selectAxisPhase
    ax: plotting axis (optional)
        The axis which the plot will appear on
    index: boolean (optional)
        if True, the frequency will be read as the index of the frequency to plot from Sim_obj.freq
    cmap: string (optional)
        If given, the plot will use the matplotlib colormap indicated by the string. For a list of colormaps and their strings, 
        see https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
    blackInactive: boolean (optional)
        If true, inactive cells will be turned black on the display
    vmin: int (optional)
        If given, this will be the minimum fourier amplitude on the colormap bar 
    vmax: int (optional)
        If given, this will be the maximum fourier amplitude on the colormap bar 

    Returns: Nothing 
    """
    if (fieldNumMag == None): 
        fieldNumMag = fieldNumPhase
    if (selectAxisMag == None): 
        selectAxisMag = selectAxisPhase

    if (index == True): 
        freq_index = frequency
    else: 
        freq_index = 0
        while (frequency >= Sim_obj.freq[freq_index]): 
            freq_index +=1
            
    toPlotMag = np.zeros_like(Sim_obj.fourierImage1D[fieldNumMag][:,:,freq_index])
    Phase_x = np.zeros_like(Sim_obj.fourierPhaseImage[fieldNumPhase][:,:,selectAxisPhase, freq_index])
    Phase_y = np.zeros_like(Sim_obj.fourierPhaseImage[fieldNumPhase][:,:,selectAxisPhase, freq_index])
            
    for freq_avg in avg_lr:

        if (type(selectAxisMag) != int): 
            toPlotMag += Sim_obj.fourierImage1D[fieldNumMag][:,:,freq_index + freq_avg]
        else: 
            toPlotMag += Sim_obj.fourierImage[fieldNumMag][:,:,selectAxisMag,freq_index + freq_avg]
            
        Phase_x += np.cos(Sim_obj.fourierPhaseImage[fieldNumPhase][:,:,selectAxisPhase, freq_index + freq_avg])
        Phase_y += np.sin(Sim_obj.fourierPhaseImage[fieldNumPhase][:,:,selectAxisPhase, freq_index + freq_avg])
        
    Phase_x = Phase_x
    Phase_y = Phase_y

    toPlotPhase = np.arctan2(Phase_y, Phase_x)
    
    toPlotMag = toPlotMag / (len(avg_lr))

    num_x_cells = len(Sim_obj.coords[0])
    num_y_cells = len(Sim_obj.coords[1])
    if vmin == None: 
        vmin =  np.min(toPlotMag[np.nonzero(toPlotMag)])
    if (vmax == None): 
        vmax = np.max(toPlotMag)
    colorMap = np.zeros((num_x_cells, num_y_cells, 4))
    for x in range(0, num_x_cells): 
        for y in range(0, num_y_cells): 
            colorMap[x][y] = phaseToColor(toPlotPhase[x][y], max(min(1, (toPlotMag[x][y]-vmin)/(vmax-vmin)), 0))

    if (ax == False): 
        fig1 = plt.figure(figsize=(10,8))
        ax = fig1.add_subplot(1,1,1)


    psm = ax.imshow(colorMap[::1][::-1], extent=[Sim_obj.coords[0][0], Sim_obj.coords[0][-1], Sim_obj.coords[1][0], Sim_obj.coords[1][-1]])
    freqDisp = np.round(Sim_obj.freq[freq_index]*10**(-9), 2)
    dimTitles = ['x', 'y', 'z']
    if (type(selectAxisMag) != int):
        ax.set_title("Spatial profile of {} GHz, phase plot of {} component".format(freqDisp, dimTitles[selectAxisPhase]))
    else: 
        ax.set_title("Spatial profile of {} component of {} GHz, phase plot of {} component".format(dimTitles[selectAxisMag],freqDisp, dimTitles[selectAxisPhase]))
    ax.set_xlabel("x (meters)")
    ax.set_ylabel("y (meters)")

    if save_data == True: 
        np.savetxt(f"{data_title}_FFT_mag_freq={np.round(frequency*1e-9,2)}.csv", toPlotMag, delimiter = ",")
        np.savetxt(f"{data_title}_FFT_phase_freq={np.round(frequency*1e-9,2)}.csv", toPlotPhase, delimiter = ",")
    
    
def vortexCenter(Sim_obj, cellSize = 5*10**-9, numCells = 2, scaleFactor = 1, Ms = 8.5*10**5):  
    '''
    Finds the vortex core position by averaging the x and y components. Only works on simulations 
    with the same number of cells in the x and y direction and square cells . 
    
    Paramaters: 
    -------------------------------------------------------------------------------------------------
    Sim_obj: faoommf Simulation object
        The simulation object to be used
    Ms: float
        The saturation magnetization
    scaleFactor: float
        A scale factor you can adjust to find the true vortex center 
    
    Returns: 
    --------------------------------------------------------------------------------------------
    core_postion: array
        The position of the vortex center as a function of time
    '''
    assert (np.allclose(Sim_obj.coords[0][1]-Sim_obj.coords[0][0], Sim_obj.coords[1][1]-Sim_obj.coords[1][0])), "Simulation cells not square in xy plane "
    assert (len(Sim_obj.coords[0]) == len(Sim_obj.coords[1])), "Simulation has different number of cells in the x and y direction"
    cellSize = Sim_obj.coords[0][1]-Sim_obj.coords[0][0]
    numCells = len(Sim_obj.coords[0])
    images = Sim_obj.images[0]
    transform = (scaleFactor/Ms)*(np.pi/4)*np.array([[0, -cellSize/numCells],[cellSize/numCells, 0]])
    corePos = []
    for image in images: 
        r_avg = np.zeros(2)
        for i in range(0, 2): 
            r_avg[i] = np.sum(image[:,:,i])
        pos = transform.dot(r_avg)
        corePos.append(pos)
    corePos = np.array(corePos)
    Sim_obj.core_position = scaleFactor*corePos
    return scaleFactor*corePos

def plotVortexCenter(Sim_obj): 
    """
    Plots some stuff about the vortex core than can help estimate the gyrotropic frequency. Needs a lot of editing. 

    Paramaters: 
    -------------------------------------------------------------------------------------------------
    Sim_obj: faoommf Simulation object
        The simulation object to be used
    
    Returns: Nothing 
    """
    fig0 = plt.figure(figsize=(10,8))
    ax0 = fig0.add_subplot(1,1,1)
    fig1 = plt.figure(figsize=(10,8))
    ax1 = fig1.add_subplot(1,1,1)
    fig2 = plt.figure(figsize=(10,8))
    ax2 = fig2.add_subplot(1,1,1)
    ax0.plot(Sim_obj.core_position[:,0], Sim_obj.core_position[:,1], "+")
    r_t = Sim_obj.core_position
    rp_t = np.zeros_like(r_t)
    for t in range(0, len(r_t)):
        rp_t[t] = cart2polar(r_t[t])
    ax1.plot(Sim_obj.time, rp_t[:, 0])
    ax2.plot(Sim_obj.time, rp_t[:, 1])
    ax0.set_title("Postion of Vortex Core")
    ax1.set_title("r vs t ")
    ax2.set_title("theta vs t")

def TW_center_y(Sim_obj): 
    """
    Finds the TW center as a function of time. The TW should move along the x-axis. 
    
    Paramaters: 
    -------------------------------------------------------------------------------------------------
    Sim_obj: faoommf Simulation object
        The simulation object to be used

    Returns: 
    --------------------------------------------------------------------------------------------
    ys: array
        The position of the TW center as a function of time
    
    """
    num_x_cells = len(Sim_obj.coords[0])
    num_y_cells = len(Sim_obj.coords[1])
    dy = (Sim_obj.coords[1][1]-Sim_obj.coords[1][0])
    ys = []
    for image in Sim_obj.images[0]: 
        arr = image[int(num_x_cells/2), :, 1]
        index = 0
        for y in range(0, num_y_cells-1): 
            if (np.sign(arr[y]) != np.sign(arr[y+1])):
                index = y
        ys.append(index)
    ys = np.array(ys)*dy
    return ys

def get_gytrotropic_mode(Sim_obj, fieldNum): 
    """
    Estimates the frequency of the gyrotropic mode for a simulation where the vortex core 
    does not undergo a full orbit. It can aproximate the frequency of the gyrotropic mode 
    when the simulation time is not long enough to use the traditional FFT method

    Paramaters: 
    -------------------------------------------------------------------------------------------------
    Sim_obj: faoommf Simulation object
        The simulation object to be used
    FieldNum: int
        The field number for the magnetization 
    
    Returns: 
    --------------------------------------------------------------------------------------------
    core_postion: array
        The position of the vortex center as a function of time

    """
    r_t = Sim_obj.core_position
    rp_t = np.zeros_like(r_t)
    for t in range(0, len(r_t)):
        rp_t[t] = cart2polar(r_t[t])
    freqs = []
    for t in range(0, len(Sim_obj.time)-1): 
        freq = (rp_t[t+1]-rp_t[t])/(Sim_obj.time[t+1]-Sim_obj.time[t])/(2*np.pi)
        freqs.append(freq)
    gyrotropic_mode = np.median(freqs)
    return (freqs)

def vizField(Sim_obj, fieldNum, frame,start = 5, apl = 10, ax = None, vmin = None, vmax = None): 
    """
    Makes an arrow plot of the field at a specific time.

    Parameters: 
    ------------------------------------------------------------------------------------
    Sim_obj: faoommf Simulation object
        The simulation object to be used
    fieldNumPhase: int
        The index of the field in the simulation you want vizualize
    frame: int
        The index of the time frame that will be plotted
    ax: plotting axis (optional)
        The plotting axis where the plot will appear

    Returns: Nothing 
    """

    if (ax == None): 
        fig1 = plt.figure(figsize=(10,8))
        ax = fig1.add_subplot(1,1,1)
    xs, ys = Sim_obj.coords
    x,y = np.meshgrid(xs[start::apl], ys[start::apl])
    if (Sim_obj.dimsList[fieldNum] == 1): 
        c = Sim_obj.images[fieldNum][frame, start::apl, start::apl, 0]
        u = np.zeros_like(c)
        v = np.ones_like(c)
    else: 
        u = Sim_obj.images[fieldNum][frame, start::apl, start::apl, 0]
        v = Sim_obj.images[fieldNum][frame, start::apl, start::apl, 1]
        c = Sim_obj.images[fieldNum][frame, start::apl, start::apl, 2]
        if (vmax == None): 
            vmax = np.max(np.abs(Sim_obj.images[fieldNum][frame][start::apl, start::apl, 2]))
        if (vmin == None): 
            vmin = -vmax
    psm = ax.quiver(x,y,u,v,c, cmap = 'viridis', clim = [vmin, vmax])
    plt.colorbar(psm, ax = ax)

def makeSpherical(Sim_obj, fieldNum): 
    """
    Creates a field object in spherical coordinates from a field object in cartesian coordinates. 
    The field in spherical coordinates will be stored as an object attribute called Sim_obj.sphImages[fieldNum]. 

    Parameters: 
    ------------------------------------------------------------------------------------
    Sim_obj: faoommf Simulation object
        The simulation object to be used
    fieldNum: int
        The index of the field which the spherical coordinate field will be created 

    Returns: Nothing 
    """

    sphImage = []
    for t in range(0, len(Sim_obj.images[fieldNum])): 
        sphericalShot = np.zeros([len(Sim_obj.coords[0]), len(Sim_obj.coords[1]), 3])
        for x in range(0, len(Sim_obj.coords[0])): 
            for y in range(0, len(Sim_obj.coords[1])): 
                sphericalShot[x][y] = cart2spherical(Sim_obj.images[fieldNum][t][x][y])

        sphImage.append(sphericalShot)
    Sim_obj.sphImage[fieldNum] = sphImage

def make_delta_m(Sim_obj, fieldNum): 

    """
    Creates a field object which represents the difference between the magnetization at a given time and 
    the initial magnetization. For example, at any given time t, dm(t) = m(t)-m(0). dm is stored as a field.
    The field in spherical coordinates will be stored as an object attribute called Sim_obj.dm[fieldNum]. 

    Parameters: 
    ------------------------------------------------------------------------------------
    Sim_obj: faoommf Simulation object
        The simulation object to be used
    fieldNum: int
        The index of the field which the spherical coordinate field will be created 

    Returns: Nothing 
    """

    dm = []

    for t in range(0, len(Sim_obj.images[fieldNum])): 
        dm_shot = Sim_obj.images[fieldNum][t] - Sim_obj.images[fieldNum][0]
        dm.append(dm_shot)


    Sim_obj.dm[fieldNum] = dm

def make_delta_m_sph(Sim_obj, fieldNum): 

    """
    Creates a field object which represents the difference between the magnetization at a given time and 
    the initial magnetization in spherical coordinates. For example, at any given time t, dm(t) = m(t)-m(0). dm is stored as a field.
    The field in spherical coordinates will be stored as an object attribute called Sim_obj.dm[fieldNum]. 

    dm = [dr, dTheta, dPhi] where Theta is the polar angle and phi is the azimuthal angle of the magnetization 

    Parameters: 
    ------------------------------------------------------------------------------------
    Sim_obj: faoommf Simulation object
        The simulation object to be used
    fieldNum: int
        The index of the field which the spherical coordinate field will be created 

    Returns: Nothing 
    """

    dm_sph = []

    if not Sim_obj.sphImage[fieldNum]: 
        print("making spherical images")
        makeSpherical(Sim_obj, fieldNum)

    for t in range(0, len(Sim_obj.sphImage[fieldNum])): 
        dm_shot = Sim_obj.sphImage[fieldNum][t] - Sim_obj.sphImage[fieldNum][0]
        dm_shot[:,:,2] = (dm_shot[:,:,2] + np.pi)%(2*np.pi) - np.pi
        dm_sph.append(dm_shot)

    Sim_obj.dm_sph[fieldNum] = dm_sph

def plot_dm(Sim_obj, fieldNum, time, axis = [0,0,1], ax = None, cmap = "bwr", blackInactive = False, vmin = None, vmax = None, Ms = None):

    """
    Plots the spatial profile of the change in magnetization at a given time

    dm = [dr, dTheta, dPhi] where Theta is the polar angle and phi is the azimuthal angle of the magnetization 

    Parameters: 
    ------------------------------------------------------------------------------------
    Sim_obj: faoommf Simulation object
        The simulation object to be used
    fieldNum: int
        The index of the field to plot dm 
    time: float
        The time at which to plot dm(t) = m(t)-m(0)
    axis: str or arr
        If axis is an array of length 3, axis is the direction along which to plot dm. The dot product of 
        dm and axis will be plotted. For example, if axis = [1, 0, 0], the x component of dm will be 
        plotted. 

        Additionally, axis can also take the arguements "rho", "phi", or "theta". If one of these arguements 
        is given, the change in these coorinates will be plotted. For example, if axis = "phi",
        d_phi(t) = phi(t) - phi(0) will be plotted. 
    ax: matplotlib axis 
        if given, the plot will appear on the axis provided
    cmap: 
        The colormap to be used to plot dm 
    blackInactive: boolean
        If true, cells with 0 will appear black
    vmin: float
        The minimum value of dm on the colormap
    vmax: float
        The maximum value of dm on the colormap
    Ms: float
        If the field provided represents the magnetization field, and you wish to normalize to 
        unit magnetization, provide the saturation magnetization value here. Only works if axis 
        is a length 3 array and not a string arguement. 


    Returns: Nothing 
    """

    if (type(time) == int): 
        tindex = time
        
    else: 
        tindex = 0
        while (time>Sim_obj.time[tindex]): 
            tindex +=1

    if (type(axis) != str): 
        dm = Sim_obj.dm[fieldNum][tindex]

        if not Sim_obj.dm[fieldNum] : 
            make_delta_m(Sim_obj, fieldNum)

        if Ms: 
            dm = dm/Ms

        axis = axis/la.norm(axis)
        toPlot = dm[:,:,0]*axis[0] + dm[:,:,1]*axis[1] + dm[:,:,2]*axis[2]

    if (axis == "phi"): 
        if not Sim_obj.dm_sph[fieldNum]: 
            print("making dm")
            make_delta_m_sph(Sim_obj, fieldNum)
        
        toPlot = Sim_obj.dm_sph[fieldNum][tindex][:,:,2]

    if (axis == "theta"): 
        if not Sim_obj.dm_sph[fieldNum]: 
            make_delta_m_sph(Sim_obj, fieldNum)
        
        toPlot = Sim_obj.dm_sph[fieldNum][tindex][:,:,1]

    if (axis == "rho"): 
        if not Sim_obj.dm_sph[fieldNum]: 
            make_delta_m_sph(Sim_obj, fieldNum)
        
        toPlot = Sim_obj.dm_sph[fieldNum][tindex][:,:,0]

    if (not ax): 
        fig1 = plt.figure(figsize=(10,8))
        ax = fig1.add_subplot(1,1,1)

    cmp = cm.get_cmap(cmap, 256)
    newColors = cmp(np.linspace(0,1,256))
    if (blackInactive == True): 
        ind = 1
    else: 
        ind = 0 
    newColors[:ind, :] = [0,0,0,1]
    myCmp = ListedColormap(newColors)

    if vmax == None: 
        vmax =  np.max([np.abs(toPlot)])

    if vmin == None: 
        vmin =  - np.max([np.abs(toPlot)])
    psm = ax.imshow(toPlot[::1][::-1], extent=[Sim_obj.coords[0][0], Sim_obj.coords[0][-1], Sim_obj.coords[1][0], Sim_obj.coords[1][-1]], 
                    cmap = myCmp, vmin = vmin, vmax = vmax)

    plt.colorbar(psm, ax = ax)    

def plotV(Sim_obj, fieldNum, cell, ax = None): 
    """
    Plots the Field vs time for a cell 

    Paramaters: 
    ------------------------------------------------------------------------------------------
    Sim_obj: faoommf Simulation object
        The simulation object to be used
    fieldNumPhase: int
        The index of the field in the simulation you want vizualize the FFTs 
    cell: array (2x1)
        The index of the cell which the magnetization will be plotted
    ax: plotting axis (optional)
        If given, the plot will be shown on the given axis. 

    Returns: Nothing

    """
    dims = Sim_obj.dimsList[fieldNum]
    if (ax == None): 
        fig0 = plt.figure(figsize=(10,8))
        ax = fig0.add_subplot(1,1,1)
    x = cell[0]
    y = cell[1]

    if (dims>1):
        dimList = ["x", "y", "z"]
    else: 
        dimList = ["Scalar"]
    for dim in range(0, dims):
        ax.plot(Sim_obj.time, Sim_obj.v_t[fieldNum][x][y][dim], label = dimList[dim])


    ax.set_xlabel("time (s)")
    ax.set_ylabel(Sim_obj.titles[fieldNum])
    ax.set_title("{} plot for cell {}, {}".format(Sim_obj.titles[fieldNum], x,y))    
    ax.legend()

def getBarrierAngles(Sim, stage, r = None, index = True, basic = True):

    """
    Returns the angles Theta and Phi of the magnetization at a radius r along the x-axis. 

    Paramaters: 
    ------------------------------------------------------------------------------------------
    Sim: faoommf Simulation object
        The simulation object to be used
    stage: int 
        The index of the time at which to return Theta and Phi
    r: int
        The index of the radius at which to return Theta and Phi

    Returns: Nothing

    """

    num_x_cells = len(Sim.coords[0])
    num_y_cells = len(Sim.coords[1])

    centerInd = int(num_x_cells / 2) 

    if (num_x_cells % 2 == 0): 
        delta = 1
        centerInd -= 1
    else: 
        delta = 0
    if (r == None): 
        r = int(num_x_cells / 2) 
 
    targetCells = [[centerInd - r, centerInd], [centerInd + r + delta, centerInd + delta], [centerInd + delta, centerInd- r], [centerInd, centerInd + r + delta]]
    targetCellsBasic = [[centerInd,centerInd+r],[centerInd+delta,centerInd+r] ]

    theta = 0
    phi = 0 
    m = [0,0,0]

    if (basic == True): 
        for cell in targetCellsBasic: 
            m += Sim.images[0][stage][cell[0]][cell[1]]

        theta = np.arctan2(m[2], (m[0]**2+m[1]**2)**(1/2))
        phi = np.angle(m[0] + 1j*m[1])%(2*np.pi)


        return theta, phi

    else: 
        for cell in targetCells: 
            m = Sim.v_t[0][cell[0]][cell[1]][:,stage]
            r_ = [Sim.coords[0][cell[0]], Sim.coords[1][cell[1]], 0]
            theta += np.arctan(m[2]/(m[0]**2+m[1]**2)**(1/2))%(2*np.pi)

            m = np.array(m)
            r_ = np.array(r_)

            m[2] = 0
            m = np.array(m)
            r_ = np.array(r_)

            phi_abs = np.arccos(m.dot(r_)/(la.norm(m)*la.norm(r_)))
            sign = np.sign(np.cross(r_, m)[2])
            phi+=sign*phi_abs

        theta = theta/len(targetCells)
        phi = phi/len(targetCells)

        return theta, phi    

def fullFourierAnalysis(Sim_obj, fieldNum, 
                        s_avg_before = False, 
                        selectAxis = None, 
                        removeIndex = [0,0], 
                        savePlots = False, 
                        logPlot = False, 
                        xlim = None,
                        ylim = None,
                        blackInactive = False, 
                        cmap = None, 
                        fieldPlot = False,
                        negativePeaks = 1): 
    """
    Uses built-in methods to perform the typical steps in a fourier analysis of an OOMMF Simulation. 

    What it does: 
        1. Creates the Fourier Image of the Simulation if it has not been created yet using Sim_obj.createAFourierImage . 
        2. Plots Magnetization vs time for a cell at the center of the simulaiton and a cell 3/4th up the x and y axis. 
           This is done to check for the presence of precession. 
        3. Plots the spatially averaged time Fourier Transform of Magnetization vs Time.
        4. Asks which frequencies it should plot the spatial profile of. 
        5. Plots the spatial profiles of the given frequencies 
        6. Optionally, saves the plots

    Parameters: 
    ---------------------------------------------------------------------------------------------------
    savePlots: boolean (optional): 
        If True, saves a .pgn file for each plot created besides the Magnetization vs Time as this graph is rarely useful. 

    """

    if (type(Sim_obj.fourierImage[fieldNum]) != np.ndarray): 
        createAFourierImage(Sim_obj, fieldNum)
    num_x_cells = len(Sim_obj.coords[0])
    num_y_cells = len(Sim_obj.coords[1])

    fig0 = plt.figure(figsize=(10,8))
    ax0 = fig0.add_subplot(1,1,1)
    fig1 = plt.figure(figsize=(10,8))
    ax1 = fig1.add_subplot(1,1,1)
    cell0 = [int(num_x_cells/2), int(num_y_cells/2)]
    cell1 = [int(3*num_x_cells/4), int(3*num_y_cells/4)]
    plotV(Sim_obj, fieldNum, cell0, ax0)
    plotV(Sim_obj, fieldNum, cell1, ax1)
    ax0.set_title("{} plot for a center cell".format(Sim_obj.titles[fieldNum]))
    ax1.set_title("{} plot for a body cell".format(Sim_obj.titles[fieldNum]))
    ax0.legend()
    ax1.legend()

    fig2 = plt.figure(figsize=(10,8))
    ax2 = fig2.add_subplot(1,1,1)

    if (type(Sim_obj.fourierAverage[fieldNum]) != np.ndarray):
        createAFourierImage(Sim_obj, fieldNum)
    peaks = []
    if (s_avg_before == False): 
        if (type(selectAxis) != int): 
            toPlot = Sim_obj.fourierAverage[fieldNum]
        else: 
            toPlot = Sim_obj.dimFourierAverage[fieldNum][selectAxis]
    elif (s_avg_before == True): 
        if (type(selectAxis) != int): 
            print("you should select an axis if the spatial summation is done before the FFT")
        else: 
            toPlot = Sim_obj.s_avg_FT[fieldNum][selectAxis]
    for i in range(1, len(toPlot)-1): 

        if (negativePeaks*toPlot[i]>negativePeaks*toPlot[i-1] and negativePeaks*toPlot[i]>negativePeaks*toPlot[i+1]): 
            peaks.append([i, Sim_obj.freq[i], toPlot[i]])

    peaks.sort(key = lambda x: x[2], reverse = True)
    ind = 0
    for peak in peaks: 
        if (logPlot == True): 
            ax2.plot(peak[1], np.log(peak[2]),"o", label = "{}, {}GHz".format(ind, peak[1]*10**(-9)))
        else: 
            ax2.plot(peak[1], peak[2],"o", label = "{}, {}GHz".format(ind, peak[1]*10**(-9)))
        ind +=1
    if (ylim == None):
        if (logPlot == True): 
            ylim = (np.log(min(toPlot)), np.log(peaks[0][2]))
        else: 
            ylim = (min(toPlot), peaks[0][2])
    if (xlim == None): 
        xlim = (Sim_obj.freq[0], Sim_obj.freq[-1])
    vizFourierModes(Sim_obj, fieldNum, s_avg_before = s_avg_before, selectAxis = selectAxis, logPlot = logPlot, ax = ax2, xlim = xlim, ylim = ylim)
    ax2.legend(title='Peaks', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    if (savePlots == True): 
        fig2.savefig("{}_SpinWaveMode_Plot".format(Sim_obj.titles[fieldNum]))
    plt.show(block = False)
    PtoPlot = []
    term = "run"
    nums = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    while (term != "done"): 
        term = input("Which peaks would you like to plot the spatial profile? Enter an integer corosponding to the peaks on the graph. (done: done entering peaks; clear: clears list of peaks; rl: removes last peak entered)")
        if (term[0] in nums): 
            num = int(term)
            if ((num>=0) and (num<len(peaks))):
                PtoPlot.append(num)
        elif (term == "clear"): 
            PtoPlot = []
        elif (term == "rl"): 
            PtoPlot = PtoPlot[0:len(PtoPlot)-1:]
        elif (term != "done"): 
            print("invalid command")
    peaksToPlot = []
    for ind in PtoPlot:
        peaksToPlot.append(peaks[ind])
    peaksToPlot.sort(key = lambda x: x[1], reverse = False) 
    for i in range(0, len(peaksToPlot)): 
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(1,1,1)
        if (fieldPlot == True): 
            Sim_obj.vizField(fieldNum, 0, ax = ax)
        vizFrequencyPlot(Sim_obj, fieldNum, peaksToPlot[i][0], index = True, ax = ax, cmap = cmap, blackInactive = blackInactive)
        if (savePlots == True): 
            fig.savefig("{}_FrequencySpatialProfile_{}_MHz".format(Sim_obj.titles[fieldNum], int(peaksToPlot[i][1]*10**(-6))))

def SpinWaveMode_FourierAnalysis(Sim_obj, 
                                removeIndex = [0,0], 
                                savePlots = False, 
                                logPlot = False, 
                                xlim = None,
                                ylim = None,
                                blackInactive = False, 
                                cmap = None, 
                                fieldPlot = False,
                                negativePeaks = 1, 
                                phasePlots = True): 
    """
    Uses built-in methods to perform the typical steps in a fourier analysis of an OOMMF Simulation. 

    What it does: 
        1. Creates the Fourier Image of the Simulation if it has not been created yet using Sim_obj.createAFourierImage . 
        2. Plots Magnetization vs time for a cell at the center of the simulaiton and a cell 3/4th up the x and y axis. 
           This is done to check for the presence of precession. 
        3. Plots the spatially averaged time Fourier Transform of Magnetization vs Time.
        4. Asks which frequencies it should plot the spatial profile of. 
        5. Plots the spatial profiles of the given frequencies 
        6. Optionally, saves the plots

    Parameters: 
    ---------------------------------------------------------------------------------------------------
    savePlots: boolean (optional): 
        If True, saves a .pgn file for each plot created besides the Magnetization vs Time as this graph is rarely useful. 

    """

    if (type(Sim_obj.fourierImage[0]) != np.ndarray): 
        createAFourierImage(Sim_obj, 0)
    if (type(Sim_obj.fourierAverage[1]) != np.ndarray):
        createAFourierImage(Sim_obj, 1)
    num_x_cells = len(Sim_obj.coords[0])
    num_y_cells = len(Sim_obj.coords[1])

    fig0 = plt.figure(figsize=(10,8))
    ax0 = fig0.add_subplot(1,1,1)
    fig1 = plt.figure(figsize=(10,8))
    ax1 = fig1.add_subplot(1,1,1)
    cell0 = [int(num_x_cells/2), int(num_y_cells/2)]
    cell1 = [int(3*num_x_cells/4), int(3*num_y_cells/4)]
    plotV(Sim_obj, 0, cell0, ax0)
    plotV(Sim_obj,0, cell1, ax1)
    ax0.set_title("{} plot for a center cell".format(Sim_obj.titles[0]))
    ax1.set_title("{} plot for a body cell".format(Sim_obj.titles[0]))
    ax0.legend()
    ax1.legend()

    fig2 = plt.figure(figsize=(10,8))
    ax2 = fig2.add_subplot(1,1,1)

    vizFourierModes(Sim_obj, 0, ax = ax2)

    peaks = Sim_obj.peaks
    ind = 0
    
    
    ax2.legend(title='Peaks', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    if (savePlots == True): 
        fig2.savefig("{}_SpinWaveMode_Plot".format(Sim_obj.titles[1]))
    plt.show(block = False)
    PtoPlot = []
    term = "run"
    nums = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    while (term != "done"): 
        term = input("Which peaks would you like to plot the spatial profile? Enter an integer corosponding to the peaks on the graph. (done: done entering peaks; clear: clears list of peaks; rl: removes last peak entered)")
        if (term[0] in nums): 
            num = int(term)
            if ((num>=0) and (num<len(peaks))):
                PtoPlot.append(num)
        elif (term == "clear"): 
            PtoPlot = []
        elif (term == "rl"): 
            PtoPlot = PtoPlot[0:len(PtoPlot)-1:]
        elif (term != "done"): 
            print("invalid command")
    peaksToPlot = []
    for ind in PtoPlot:
        peaksToPlot.append(peaks[ind])
    peaksToPlot.sort(key = lambda x: x[1], reverse = False) 
    for i in range(0, len(peaksToPlot)): 
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(1,1,1)
        if (fieldPlot == True): 
            Sim_obj.vizField(0, 0, ax = ax)
        vizFrequencyPlot(Sim_obj, 0, peaksToPlot[i][0], index = True, ax = ax, cmap = cmap, blackInactive = blackInactive)
        if (phasePlots == True): 
            fig1 = plt.figure(figsize=(10,8))
            ax1 = fig1.add_subplot(1,1,1)
            vizFrequencyPhasePlot(Sim_obj, 0,peaksToPlot[i][0], selectAxisPhase = 2, fieldNumMag = 0, selectAxisMag = True, index = True, ax = ax1)
        if (savePlots == True): 
            fig.savefig("{}_FrequencySpatialProfile_{}_MHz".format(Sim_obj.titles[0], int(peaksToPlot[i][1]*10**(-6))))
            fig1.savefig("{}_FrequencyPhaseSpatialProfile_{}_MHz".format(Sim_obj.titles[0], int(peaksToPlot[i][1]*10**(-6))))




