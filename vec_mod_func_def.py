
# This cell is for writing all my functions to a python file
import scipy
from scipy.optimize import minimize
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib import cm
import math as m


def ASCII_cleaner(data_path):
    file = open(data_path, 'r')
    lines = file.readlines()

    for i in range(len(lines)):
        line = lines[i]
        if line.find("R2")!=-1 or line == "\n":
            lines[i] = None

    arr=np.array(lines)
    arr = arr[arr != None]

    def line_cleaner(line):

      out = line.replace(" ", "")
      out = out.replace("\n", "")
      out = re.split(r"/|\t", out)

      if len(out) > 3:
        out = np.array([float(i) for i in out])
        return out

    data = np.array([0,0,0,0])

    for line in arr:
        add = np.array(line_cleaner(line))
        if isinstance(add, np.ndarray):
            data =np.vstack((data,add))

    data = data[1:]
    
    return data

def degree_rectifier(inp):
    if inp < 0:
        return 360 + inp
    else:
        return inp
    
def degree_unrectifier(inp):
    if inp>180:
        return -(360-inp)
    else:
        return inp
degree_unrectifier_v = np.vectorize(degree_unrectifier)
    
def cart_to_polar(x,y):
    r = np.sqrt(x**2+y**2)
    t = np.arctan2(y,x)
    
    return r,t
    
def create_df(data):
    df = pd.DataFrame(data, columns=["R2", "R1", "S13", "deg"])
    df["rad"] = df["deg"] * (m.pi/180)
    df['deg'] = df['deg'].apply(degree_rectifier)
    df['dB'] = (20 * np.log10(df['S13'].values))    

    df['x'] = np.cos(df['rad']) * df["S13"]
    df['y'] = np.sin(df['rad']) * df["S13"]
    

    
    return df

def S13_to_V(x_init,y_init, df, max_voltage=1,morphed=True):
    
    print("Total transformation from S13 to Voltages")
    amp = df["S13"].values
    phase = df["deg"].values
    
    #resetting values for bugs
    midpoint = None
    topright, topleft, botleft, botright = None, None, None, None
    
    #find first peaks
    peaks  = find_peaks(amp, phase)
    topright, topleft, botleft, botright = peaks[0], peaks[1],peaks[2],peaks[3]
    midpoint = [(topright[0]+botleft[0])*0.5, (topright[1]+botleft[1])*0.5]
    
    #_______________________________________________________
    
    #move array to origin
    x_temp  = x_init - midpoint[0]
    y_temp = y_init - midpoint[1]
    
    print("Moved midpoint to origin: ", midpoint)

    #find origin-moved peaks
    topright[0], topright[1] = topright[0] -midpoint[0], topright[1] -midpoint[1]
    topleft[0], topleft[1] = topleft[0] -midpoint[0], topleft[1] -midpoint[1]
    botright[0], botright[1] = botright[0] -midpoint[0], botright[1] -midpoint[1]
    botleft[0], botleft[1] = botleft[0] -midpoint[0], botleft[1] -midpoint[1]
    midpoint = [(topright[0]+botleft[0])*0.5, (topright[1]+botleft[1])*0.5]
    
    #_______________________________________________________
    
    #rotate the array around the origin to straighten it out

    dy = botright[1] - botleft[1]
    dx= botright[0]- botleft[0]
    
    slope = ((dy / dx))
    print("Slope: ",slope)
    angle = np.arctan(slope)

    print("Rotation Angle: ",-angle, " radians")

    mat, rot_mat = matrix_rotation(angle, x_temp,y_temp)
    x_rot, y_rot = mat[:,0], mat[:,1]
    
    if morphed:
        #rotate 90 degrees clockwise
        mat, rot_mat = matrix_rotation(m.pi/2, x_rot,y_rot)
        x_rot, y_rot = mat[:,0], mat[:,1]
        print("Rotation Angle: ",90, " degrees")
    
    
    #_______________________________________________________
    
    #Scaling the array to fit within the boundaries at most
    x_limits, y_limits = np.array([-max_voltage/2, max_voltage/2]), np.array([-max_voltage/2, max_voltage/2])
    scaling_factor = find_correction(x_rot,y_rot,max_factor=3,resolution=0.01,x_limits=x_limits, y_limits=y_limits)

    x_scaled, y_scaled = x_rot*scaling_factor, y_rot*scaling_factor
    
    
    #_______________________________________________________
    
    #move the array to upper right quadrant
    boxlim = max_voltage/2
    x_pre_final, y_pre_final = x_scaled + boxlim, y_scaled+boxlim
    print("Moved array to upper right quadrant: ", [boxlim, boxlim])
    
    #_______________________________________________________
    
    #find box limits instead of corners
    topright, topleft, botleft, botright, peaks_x, peaks_y = cart_array_limit_finder(x_pre_final, y_pre_final)
    
    #find scaling factor to fill out domain
    box_width, box_height = abs(topright[0]-topleft[0]), abs(topright[1]-botright[1])
    final_scaling_factor = min(max_voltage/box_width, max_voltage/box_height)
    print("Second Scaling factor: ", final_scaling_factor)
    
    x_pre_final, y_pre_final = x_pre_final - boxlim, y_pre_final-boxlim
    x_scaled2, y_scaled2 = x_pre_final*final_scaling_factor, y_pre_final*final_scaling_factor
    x_scaled2, y_scaled2 = x_scaled2 + boxlim, y_scaled2+boxlim
    
    #find box limits instead of corners
    topright, topleft, botleft, botright, peaks_x, peaks_y = cart_array_limit_finder(x_scaled2, y_scaled2)
    
    #move array so that voltage limits are reached
    if x_scaled2.max() > max_voltage:
        x_delta = x_scaled2.max() - max_voltage
        x_final = x_scaled2 - x_delta
    else:
        x_final = x_scaled2
        x_delta=0
    if y_scaled2.max() > max_voltage:
        y_delta = y_scaled2.max() - max_voltage
        y_final = y_scaled2 - y_delta
    else:
        y_final = y_scaled2
        y_delta = 0
    
    print("X and Y Deltas for final adjustments: ",[x_delta, y_delta])
    topright, topleft, botleft, botright, peaks_x, peaks_y = cart_array_limit_finder(x_final, y_final)
    midpoint = [(topright[0]+botleft[0])*0.5, (topright[1]+botleft[1])*0.5]
    
    x_final, y_final = x_final+(boxlim-midpoint[0]), y_final+(boxlim-midpoint[1])
    print("Moved midpoint to domain center: ", [boxlim-midpoint[0], boxlim-midpoint[1]])
    
    x_final, y_final = x_final-(boxlim), y_final-(boxlim)
    
    horiz_scale = 1/((x_final.max()-x_final.min()))
    x_final = x_final * horiz_scale
    print("Scaled horizontally by factor: ", horiz_scale)
    x_final, y_final = x_final+(boxlim), y_final+(boxlim)
    
    #find box limits instead of corners
    topright, topleft, botleft, botright, peaks_x, peaks_y = cart_array_limit_finder(x_final, y_final)
    
    VI, VQ = x_final, y_final
    
    
    print("Max X: ",round(VI.max(),10))
    print("Max Y: ",round(VQ.max(),10))
    print("Min X: ",round(VI.min(),10))
    print("Min Y: ",round(VQ.min(),10))
    
    print("________________________")
    
    return VI, VQ, angle, scaling_factor, final_scaling_factor, peaks_x, peaks_y, rot_mat

def find_correction(x,y, max_factor, resolution,x_limits, y_limits):
    
    factors = np.arange(-max_factor, max_factor, resolution)

    def get_max(factor):
        x_max = max((x *factor).max(), abs((x*factor).min()))
        y_max = max((y *factor).max(), abs((y*factor).min()))
        
        return x_max, y_max
    
    get_max_v = np.vectorize(get_max)
    x_maxes, y_maxes = get_max_v(factors)
    tot_arr = np.vstack((x_maxes, y_maxes))
    tot_arr = np.vstack((tot_arr, factors))
    
    tot_arr = tot_arr
    
    maskx1 = np.where (tot_arr[0] <= x_limits[1], 1, 0)
    maskx2 = np.where (tot_arr[0] >= x_limits[0], 1, 0)
    masky1 = np.where (tot_arr[1] <= y_limits[1], 1, 0)
    masky2 = np.where (tot_arr[1] >= y_limits[0], 1, 0)
    
    maskx = maskx1 * maskx2
    masky = masky1 * masky2
    
    mask = maskx * masky
    factors = mask * tot_arr[2]
    factor = factors.max()
    
    print("Scaling factor: ", factor)
    
    return factor

def find_peaks(amp_in, phase_in):
    peaks = []
    phases = []
    
    if phase_in.mean() <= 2*m.pi: #convert to degrees if needed
        phase_in = np.degrees(phase_in)
        
    
    for quadrant in [(0,90), (90,180), (180,270), (270,360)]:            
        
        amp_temp = amp_in[(phase_in>=quadrant[0]) & (phase_in<quadrant[1])]
        phase_temp = phase_in[(phase_in>=quadrant[0]) & (phase_in<quadrant[1])]
        peak_amp = amp_temp[amp_temp.argmax()]
        peak_phase_deg = phase_temp[amp_temp.argmax()]
        
        if quadrant == (270,360):
            
            amp_temp = amp_in[(phase_in>=quadrant[0]) & (phase_in<0.9*quadrant[1])]
            phase_temp = phase_in[(phase_in>=quadrant[0]) & (phase_in<0.9*quadrant[1])]
            peak_amp = amp_temp[amp_temp.argmax()]
            peak_phase_deg = phase_temp[amp_temp.argmax()]
                    
        peak_phase_rad = np.radians(peak_phase_deg)
        
        peak_x = np.cos(np.radians(peak_phase_deg)) * peak_amp
        peak_y = np.sin(np.radians(peak_phase_deg)) * peak_amp
        
        peak = [peak_x, peak_y]
        peaks.append(peak)
        phases.append(peak_phase_deg)
    
    return peaks


def matrix_rotation(theta, x,y):
    
    mat = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    datamat = (np.vstack((x, y))).T
    outmat = np.matmul(datamat, mat)
    print("Rotation Matrix:\n", mat)
    return outmat, mat

def plot_corners(color):
    for corner_pair in [[topright, topleft], [topright, botright], [botleft, topleft], [botleft, botright]]:
        c1, c2 = corner_pair[0], corner_pair[1]
        plt.plot([c1[3], c2[3]] ,[c1[4], c2[4]], marker = 'o', color=color, linestyle="dashed")
        
def cart_array_limit_finder(x,y):
    topright = np.array([x.max(), y.max()])
    topleft = np.array([x.min(), y.max()])
    botleft = np.array([x.min(), y.min()])
    botright = np.array([x.max(), y.min()])

    peaks_x = np.array([topright[0], topleft[0], botright[0], botleft[0]])
    peaks_y = np.array([topright[1], topleft[1], botright[1], botleft[1]])

    return topright, topleft, botleft, botright, peaks_x, peaks_y

def plot_S2V_transformation(df,markersize=10, alpha=0.4, morphed=True):    
    x=df['x']
    y=df['y']
    amp = df["S13"].values
    phase = df["deg"].values

    #resetting values for bugs
    midpoint = None
    topright, topleft, botleft, botright = None, None, None, None

    #create figure
    fig, ax = plt.subplots(figsize=(12,12))
    ax.set_xlabel("S(1,3)")
    ax.tick_params(direction='in',which='both')
    ax.grid(True)

    #plot original data
    ax.scatter(x,y, alpha=alpha, linewidth=0.2, label="original", s=markersize)
    
    #find peaks
    peaks  = find_peaks(amp, phase)

    topright, topleft, botleft, botright = peaks[0], peaks[1],peaks[2],peaks[3]
    midpoint = [(topright[0]+botleft[0])*0.5, (topright[1]+botleft[1])*0.5]

    #plot corners
    

    #move array to origin
    x_temp  = x - midpoint[0]
    y_temp = y - midpoint[1]

    #find origin-moved peaks
    topright[0], topright[1] = topright[0] -midpoint[0], topright[1] -midpoint[1]
    topleft[0], topleft[1] = topleft[0] -midpoint[0], topleft[1] -midpoint[1]
    botright[0], botright[1] = botright[0] -midpoint[0], botright[1] -midpoint[1]
    botleft[0], botleft[1] = botleft[0] -midpoint[0], botleft[1] -midpoint[1]

    midpoint = [(topright[0]+botleft[0])*0.5, (topright[1]+botleft[1])*0.5]
    ax.scatter(midpoint[0], midpoint[1])
    ax.scatter(x_temp,y_temp, alpha=alpha, linewidth=0.2, label="moved",  s=markersize)

    #plot peaks or corners
    for corner_pair in [[topright, topleft], [topright, botright], [botleft, topleft], [botleft, botright]]:
        c1, c2 = corner_pair[0], corner_pair[1]
        plt.plot([c1[0], c2[0]] ,[c1[1], c2[1]], marker = 'o', color="red", linestyle="dashed")

    #rotate the array around the origin to straighten it out

    dy = botright[1] - botleft[1]
    dx= botright[0]- botleft[0]
    
    slope = ((dy / dx))
    print("Slope: ",slope)
    angle = np.arctan(slope)

    print("Rotation Angle: ",angle, " radians")

    mat, rotmat = matrix_rotation(angle, x_temp,y_temp)
    x_rot, y_rot = mat[:,0], mat[:,1]
    
    if morphed:
        #rotate 90 degrees counterclockwise
        mat, rot_mat = matrix_rotation(m.pi/2, x_rot,y_rot)
        x_rot, y_rot = mat[:,0], mat[:,1]
        print("Rotation Angle: ",90, " degrees")
    
    
    ax.scatter(x_rot,y_rot, alpha=alpha, linewidth=0.2, label="rotated", s=markersize)

    #find rotated peaks

    peaks_x = np.array([topright[0], topleft[0], botright[0], botleft[0]])
    peaks_y = np.array([topright[1], topleft[1], botright[1], botleft[1]])

    peaks_rotated, matrot2 = matrix_rotation(angle+m.pi/2, peaks_x,peaks_y)
    peaks_x_rot, peaks_y_rot = peaks_rotated[:,0], peaks_rotated[:,1]

    topright[0], topright[1] = peaks_x_rot[0], peaks_y_rot[0]
    topleft[0], topleft[1] = peaks_x_rot[1], peaks_y_rot[1]
    botright[0], botright[1] = peaks_x_rot[2], peaks_y_rot[2]
    botleft[0], botleft[1] = peaks_x_rot[3], peaks_y_rot[3]

    #plot rotated peaks or corners
    

    #Scaling the array to fit within the boundaries at most
    scaling_factor = find_correction(x_rot,y_rot,max_factor=3,resolution=0.01,x_limits=[-0.5,0.5],y_limits=[-0.5,0.5])

    x_scaled, y_scaled = x_rot*scaling_factor, y_rot*scaling_factor
    ax.scatter(x_scaled,y_scaled, alpha=alpha, linewidth=0.2, label="scaled", s=markersize)

    #find scaled peaks
    peaks_rotated *= scaling_factor
    peaks_x_rot, peaks_y_rot = peaks_rotated[:,0], peaks_rotated[:,1]

    topright[0], topright[1] = peaks_x_rot[0], peaks_y_rot[0]
    topleft[0], topleft[1] = peaks_x_rot[1], peaks_y_rot[1]
    botright[0], botright[1] = peaks_x_rot[2], peaks_y_rot[2]
    botleft[0], botleft[1] = peaks_x_rot[3], peaks_y_rot[3]
    
    for corner_pair in [[topright, topleft], [topright, botright], [botleft, topleft], [botleft, botright]]:
        c1, c2 = corner_pair[0], corner_pair[1]
        plt.plot([c1[0], c2[0]] ,[c1[1], c2[1]], marker = 'o', color="red", linestyle="dashed")
    

    ax.legend()
    plt.tight_layout()
    
def plot_cart_array(x, y, peaks_x=None, peaks_y=None,max_voltage=1, colormap=None, color_intensity=None, xlabel=None, ylabel=None, title=None):
    #create figure
    fig, ax = plt.subplots(figsize=(12,12))
    ax.set_xlabel(xlabel)
    ax.set_xlabel(ylabel)
    ax.tick_params(direction='in',which='both')
    ax.set_title(title)
    ax.grid(True)
    
    #plot array
    if colormap == None:
        ax.scatter(x,y, alpha=0.5, linewidth=0.2,s=20)
    else:
        im = ax.scatter(x,y, alpha=0.5, linewidth=0.2,s=20, c=color_intensity, cmap=colormap)
        cax = fig.add_axes([0.99, 0.1, 0.075, 0.8])
        fig.colorbar(im, cax = cax, orientation = 'vertical', label="Error at point")
    
    #plot corners
    
    if isinstance(peaks_y,np.ndarray) and isinstance(peaks_x,np.ndarray):
        topright, topleft, botleft, botright = [0,0],[0,0],[0,0],[0,0]

        topright[0], topright[1] = peaks_x[0], peaks_y[0]
        topleft[0], topleft[1] = peaks_x[1], peaks_y[1]
        botright[0], botright[1] = peaks_x[2], peaks_y[2]
        botleft[0], botleft[1] = peaks_x[3], peaks_y[3]

        for corner_pair in [[topright, topleft], [topright, botright], [botleft, topleft], [botleft, botright]]:
            c1, c2 = corner_pair[0], corner_pair[1]
            plt.plot([c1[0], c2[0]] ,[c1[1], c2[1]], marker = 'o', color="red", linestyle="dashed")
    
        #draw midpoint
        midpoint = [(topright[0]+botleft[0])*0.5, (topright[1]+botleft[1])*0.5]
        ax.scatter(midpoint[0], midpoint[1])
        ax.scatter(max_voltage/2, max_voltage/2)

def plot_S_data_polar(amp_arr, phase_arr, main_label):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(12,12))
    #plot points
    if phase_arr.mean() > 2*m.pi: #convert to radians if needed
        phase_arr = np.radians(phase_arr)
    
    ax.scatter(phase_arr, amp_arr, alpha=0.5, label=main_label)

        
    ax.set_xlabel(main_label)
    
def plot_error(df):

    VI_delta = abs(df['VI'].values - df['VI_unmorphed'].values)
    VQ_delta = abs(df['VQ'].values - df['VQ_unmorphed'].values)
    total_error = np.sqrt(VI_delta**2 + VQ_delta**2)
    
    df['error'] = total_error
