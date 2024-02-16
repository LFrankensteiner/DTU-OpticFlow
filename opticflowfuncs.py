import numpy as np
from scipy.ndimage import convolve1d
import cv2 as cv
import matplotlib.pyplot as plt
from skimage import color, io
import os


def linearLSQ(A,y):
    Q,R=np.linalg.qr(A,mode="reduced")
    x, _, rank, _ =np.linalg.lstsq(R,Q.T@y, rcond=-1)
    return x, rank

def read_img_folder(folder, gray = True):
    frame_names = os.listdir(folder) # List of files in folder of images
    n = len(frame_names) # Number of frames
    # Preallocates array to store frames
    frame = frame_names[0]
    im = io.imread(os.path.join(folder,frame))
    if gray:
        V = np.zeros([*im.shape[:2],n])
        for i,frame in enumerate(frame_names):
            im = io.imread(os.path.join(folder,frame))
            g = color.rgb2gray(im)
            V[:,:,i] = g
    else:
        V = np.zeros([*im.shape[:2],3,n], int)
        for i,frame in enumerate(frame_names):
            im = io.imread(os.path.join(folder,frame))
            V[:,:,:,i] = im
    return V


def read_video_cv(video_path, n_frames = None):
    cap = cv.VideoCapture(video_path)
    if n_frames is None:
        n_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    i = 0
    dims = cap.read()[1].shape
    allFrames = np.zeros([*dims[:2], n_frames])
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame[:, :, ::-1]
        cf = color.rgb2gray(frame)
        allFrames[:,:,i] = cf
        i += 1
    cap.release()
    return allFrames

def play_vid(vid):
    n = vid.shape[2]
    for i in range(n):
        cv.imshow("img",vid[:,:,i])
        if cv.waitKey(10) == ord('q'):
            break
    cv.destroyAllWindows()


def gauss(x, sigma):
    return 1/np.sqrt(2 * np.pi * sigma**2) * np.exp(-x**2/(2 * sigma**2))

def gauss_deriv(x, sigma):
    return -x/(sigma**3 * np.sqrt(2* np.pi)) * np.exp(-x**2/(2*sigma**2))


def dist_arr(s):
    """
    :param s: Parameter determining size of array.
    :return: 1d np.array length 2s+1: [-s, -s+1,..., -1, 0, 1,...,s-1, s]
    """
    s = int(s)
    return np.array([i for i in range(-s, s+1)]) 

def gauss_kernel1d(sigma, s = None):
    """
    Creates 1d gaussian kernel.

    :param sigma: Spread of distribution.
    :param s: Parameter determining size of kernel.
    :return: 1d np.array length 2s+1: gauss([-s, -s+1,..., -1, 0, 1,...,s-1, s])
    """
    if s is None:
        s = int(5 * sigma)
    x = dist_arr(s)
    return gauss(x, sigma)

def gauss_deriv_kernel1d(sigma, s = None):
    """
    Creates 1d gaussian derivate kernel.
    
    :param sigma: Spread of distribution.
    :param s: Parameter determining size of kernel.
    :return: 1d np.array length 2s+1: dgaus([-s, -s+1,..., -1, 0, 1,...,s-1, s])
    """
    if s is None:
        s = int(5 * sigma)
    x = dist_arr(s)
    return gauss_deriv(x, sigma)

def apply_gauss(img, sigma, s = None):
    """
    Applies a gaussian filter to an "image" of arbitrary dimension.
    
    :param img: Image to apply filter on.
    :param sigma: Spread of distribution.
    :param s: Parameter determining size of kernel.
    :return: np array with same shape as img. 
    """
    kernel = gauss_kernel1d(sigma, s)
    for i in range(len(img.shape)):
        img = convolve1d(img, kernel, axis=i)
    return img

def apply_gauss_deriv(img, axis, sigma, s = None):
    """
    Applies a gaussian derivative filter to an "image" of arbitrary dimension, with respect to a specified axis.
    
    :param img: Image to apply filter on.
    :param axis: Axis of derivative.
    :param sigma: Spread of distribution.
    :param s: Parameter determining size of kernel.
    :return: np array with same shape as img. 
    """
    kernel = gauss_kernel1d(sigma, s)
    dkernel = gauss_deriv_kernel1d(sigma, s)
    for i in range(len(img.shape)):
        if i == axis:
            img = convolve1d(img, dkernel, axis=i)
        else: 
            img = convolve1d(img, kernel, axis=i)
    return img


# Class to contain a optic flow object.
class OpticFlowVid:
    def __init__(self, V, sigma, s):
        self.V = V
        self.sigma = sigma
        self.s = s
        self.x_dim, self.y_dim, self.frames = V.shape
        self.gdx = apply_gauss_deriv(V, sigma=sigma, s=s, axis=1)
        self.gdy = apply_gauss_deriv(V, sigma=sigma, s=s, axis=0)
        self.gdt = apply_gauss_deriv(V, sigma=sigma, s=s, axis=2)
        self.vid_grid = np.concatenate([np.concatenate([self.V, self.gdx], axis=1), np.concatenate([self.gdy, self.gdt], axis=1)])


    
    def frame(self, framei):
        return self.V[:,:,framei]

    def optic_flow_voxel(self,N, xi, yi, frame):
        Vxi = self.gdx[xi-N:xi+N+1, yi-N:yi+N+1, frame].flatten()
        Vyi = self.gdy[xi-N:xi+N+1, yi-N:yi+N+1, frame].flatten()
        Vti = self.gdt[xi-N:xi+N+1, yi-N:yi+N+1, frame].flatten()
        A = np.array([Vxi, Vyi]).T
        xy, rank = linearLSQ(A, -Vti)
        return xy, rank
    

    def optic_flow_frame(self, N, xstride, ystride, frame): 
        flow = []
        for xi in range(max(N,xstride//2), self.x_dim-N, xstride):
            for yi in range(max(N,ystride//2), self.y_dim-N, ystride):
                dxdy, rank = self.optic_flow_voxel(N, xi, yi, frame)
                if rank == 2:
                    flow.append({"pixel" : np.array([xi,yi]), "displacement" : dxdy})
        return flow

    def optic_flow_grid(self, N, xstride, ystride):
        flow = dict.fromkeys(range(self.frames))
        for i in range(self.frames):
            flow[i] = self.optic_flow_frame(N, xstride, ystride, i)
  #          flow[i] = []
  #          for xi in range(max(N,xstride//2), self.x_dim-N, xstride):
   #             for yi in range(max(N,ystride//2), self.y_dim-N, ystride):
    #                dxdy, rank = self.optic_flow_voxel(N, xi, yi, i)
     #               if rank == 2:
      #                  flow[i].append({"pixel" : np.array([xi,yi]), "displacement" : dxdy})
        return flow

    def optic_flow_vid(self, N, xstride, ystride, min_len = 0,  interval = 1/1000):
        flow = self.optic_flow_grid(N, xstride, ystride)
        self.optic_flow_from_grid(flow, min_len, interval)

    def optic_flow_from_grid(self, flow, min_len = 0, interval = 1/1000):
        """
        Plays optic flow video from precomputed flow grid.

        """
        for framei in range(self.frames):
            plt.cla()
            plt.ylim(self.x_dim,0)
            plt.xlim(0,self.y_dim)
            io.imshow(self.frame(framei))
            flowi = flow[framei]
            for arr in flowi:
                if np.dot(*arr["displacement"]*2) > min_len:
                    x, y = arr["pixel"]
                    plt.arrow(y, x, *arr["displacement"].T, length_includes_head=True, head_width=5, head_length=2, color="cyan")
            plt.pause(interval) 