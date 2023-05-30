
import numpy as np
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
import torch
import scipy.interpolate as spi
from scipy.io import loadmat
tensor_type = torch.DoubleTensor

def _swap_colums(ar, i, j):
    aux = np.copy(ar[:, i])
    ar[:, i] = np.copy(ar[:, j])
    ar[:, j] = np.copy(aux)
    return np.copy(ar)

def load_landmarks_torch(m_reca=None,m_ref=None):
    if m_reca is None :
        m_reca_out = np.load('./data/PsrcAnchors2.npy').T 
    else :
        m_reca_out = m_reca.copy().T     
    if m_ref is None :
        m_ref_out = np.load('./data/PtarAnchors2.npy').T  
    else :
        m_ref_out = m_ref.copy().T
    return _swap_colums(m_reca_out, 0, 1), _swap_colums(m_ref_out, 0, 1)



def interpolate_image(intensities, deformed_pixels, padding_width=1):
    '''
    This function, given the original image in the intensities tensor, returns the final registered image
    deformed pixels : tensor of coordinates of registered pixels in original image referential
    ------- 
    intensities : (nr,nc,k)
    deformed_pixels : (nr*nc,d)
    -------
    returns a registered image in JregLD, of shape (nr,nc,k)
    
    '''
    nr,nc,_ = intensities.shape
    xim,yim = np.meshgrid(range(0,nc),range(0,nr))
    xim = xim.reshape(-1)
    yim = yim.reshape(-1)
    
    deformated_pixels_numpy = deformed_pixels.detach().numpy()

    
    pad = np.ones(intensities.shape)
    padded_image = np.concatenate((np.concatenate([pad,pad,pad],axis=1),np.concatenate([pad,intensities.numpy(),pad],axis=1),np.concatenate([pad,pad,pad],axis=1)),axis=0)

    
    JregLD = np.zeros_like(intensities)
    
    for i in range(len(xim)):
        value = padded_image[int(round(deformated_pixels_numpy[i,0]) + nr), int(round(deformated_pixels_numpy[i,1]) + nc),:]
        JregLD[yim[i],xim[i],:] = value

    return JregLD

def _differences(x, y):
    """ 
    x is of shape (n, 2)
    y is of shape (m, 2)
    --------
    returns the difference between each columns in x and y in a (2,n,m) tensor
    
    """
    x_col = x.t().unsqueeze(2)  # (n,2) -> (2,n,1)
    y_lin = y.t().unsqueeze(1)  # (m,2) -> (2,1,m)
    return x_col - y_lin

def _squared_distances(x, y):
    """ 
    x is of shape (n, 2)
    y is of shape (m, 2)
    
    --------
    returns the squared euclidean distance between each columns in x and y in a (n,m) tensor
    
    """
        
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    
    return dist

def gaussian_kernel(x, y, kernel_width):
    """ 
    x is of shape (n, 2)
    y is of shape (m, 2)
    kernel_width is a value
    
    --------
    returns the gaussian kernel value between each columns in x and y in a (n,m) tensor
    
    """
    squared_dist = _squared_distances(x, y)
    return torch.exp(- squared_dist / kernel_width **2 )


def h_gradx(cp, alpha, kernel_width):
    '''
    This function computes derivative of the kernel for each cp_i, with cp_i a control point(landmark).
    ---------
    
    cp is of shape (n_landmarks, 2)
    alpha is of shape (n_landmarks, 2)
    kernel_width is a value
    
    --------
    returns a tensor of shape (2,n_landmarks,n_landmarks)
    '''
    sq = _squared_distances(cp, cp)
    A = torch.exp(-sq / kernel_width **2)
    B = _differences(cp, cp) * A
    result = - 2 * B / (kernel_width ** 2)
    return result

    
def discretisation_step(cp, alpha, dt, kernel_width):
    
    '''
    TO DO
    ---------
    This function computes a step of discretized equations for both alpha and control points. 
    Compute here a displacement step of control points an alpha, from discretized system seen in class.
    ---------
    
    cp is of shape (n_landmarks, 2)
    alpha is of shape (n_landmarks, 2)
    dt is your time step 
    kernel_width is a value
    
    --------
    
    returns resulting control point and alpha displacements in tensors of size (n_landmarks,2) both.
    
    '''
    
    
    # mid_cp = cp  + dt / 2. * torch.matmul(gaussian_kernel(cp, cp, kernel_width), alpha)
    # mid_alpha = alpha - 1./2. * dt / 2. * torch.sum(alpha * (torch.matmul(h_gradx(cp, alpha, kernel_width), alpha)), 2).t()        
    result_cp = cp + dt * torch.matmul(gaussian_kernel(cp, cp, kernel_width), alpha)
    result_alpha_1 = alpha - dt/2   * torch.sum(alpha * (torch.matmul(h_gradx(cp, alpha, kernel_width), alpha)), 2).t() 
    
    # coordinates by coordinates
    result_alpha_2 = torch.zeros((len(cp),2)).double()
    result_alpha_2[:,0] = alpha[:,0] - dt * (torch.matmul(h_gradx(cp, alpha, kernel_width)[0],alpha)*alpha).sum(1)
    result_alpha_2[:,1] = alpha[:,1] - dt * (torch.matmul(h_gradx(cp, alpha, kernel_width)[1], alpha)*alpha).sum(1)
    
    return result_cp, result_alpha_2

def shoot(cp, alpha, kernel_width, n_steps=10):
    
    """
    TO DO
    ------------
    This is the trajectory of an Hamiltonian dynamic, with system seen in lecture notes. 
    Compute here trajectories of control points and alpha from t=0 to t=1.
    ------------
    cp is of shape (n_landmarks, 2)
    alpha is of shape (n_landmarks, 2)
    n_step : number of steps in your hamiltonian trajectory, use to define your time step
    kernel_width is a value
    --------
    returns traj_cp and traj_alpha trajectories of control points and alpha in lists. 
    The length of a list is equal to n_step. 
    In each element of the list, you have a tensor of size (n_landmarks,2) returned by rk2_step_with_dp() function.
    """

    traj_cp, traj_alpha =  [], []
    traj_cp.append(cp)
    traj_alpha.append(alpha)
    
    
    dt = 1. / float(n_steps-1)
    
    for _ in range(n_steps-1):
        new_cp, new_alpha = discretisation_step(traj_cp[-1], traj_alpha[-1], dt, kernel_width)
        traj_cp.append(new_cp)
        traj_alpha.append(new_alpha)
        
    return traj_cp, traj_alpha

def register_points(traj_cp, traj_alpha, y, kernel_width):
    """
    TO DO
    ------------
    This is the application of the computed trajectories on a set of points (landmarks or new points).
    ------------
    
    traj_cp is the list containing the trajectory of your landmarks 
    traj_alpha is is the list containing the trajectory of your alpha 
    y : points you want to register (landmarks or other points), size (n,2)
    kernel_width is a value
    
    --------
    
    returns traj_y, the trajectory of points y, in a list of lenght n_step. 
    In each element of the list, you should have a tensor of dimension (n,2) (same dimensions as y)
    
    """
    
    
    traj_y = [y]
    n_steps = len(traj_cp) - 1
    dt = 1. / float(n_steps)
    
    for i in range(len(traj_cp)-1):
        
        new_y = traj_y[-1] + dt * torch.matmul(gaussian_kernel(traj_y[-1], traj_cp[i], kernel_width), traj_alpha[i])
        traj_y.append(new_y)
        
    return traj_y

def inverse_register_points(traj_cp, traj_alpha, y, kernel_width):
    """
    TO DO
    ------------
    
    Compute inverse deformation of points y in a tensor named deformed_points, by using the previously defined register_points() function
    
    ------------
    
    traj_cp is the list containing the trajectory of your landmarks 
    traj_alpha is the list containing the trajectory of your alpha 
    y : The set of points to register with the inverse deformation
    kernel_width is a value
    
    --------
    
    returns the inverse registered points y (n, 2)
    
    """
    
    
    traj_cp_inverse = traj_cp[::-1]
    traj_alpha_inverse = [-1 * elt for elt in traj_alpha[::-1]]
    deformed_points = register_points(traj_cp_inverse, traj_alpha_inverse, y, kernel_width)[-1]
    
    return deformed_points

def register_image(traj_cp, traj_alpha, image, kernel_width):
    """
    TO DO
    ------------
    This is the application of the computed trajectories on an image, by computation of inversed phi_1.
    Compute inverse deformation of image points in a tensor named deformed_points, by using the previously defined register_points() function
    
    ------------
    
    traj_cp is the list containing the trajectory of your landmarks 
    traj_alpha is the list containing the trajectory of your alpha 
    image : image to register, of size (nr,nc,k), k is number of channels/ pixels' values, 3 for colored image, one for grey image
    kernel_width is a value
    
    --------
    
    returns the registered image, of same dimensions as image, (nr,nc,k)
    
    """
    
    nr, nc, k = image.shape
#     points = np.array(np.meshgrid(range(nr), range(nc)))
#     points = np.swapaxes(points, 0, 2).reshape(nr * nc, 2) 
#     points = torch.from_numpy(points).type(tensor_type)

    xim,yim = np.meshgrid(range(nc), range(nr))
    xim = xim.reshape(1,-1)
    yim = yim.reshape(1,-1)

    points = np.array([yim[0],xim[0]]).T
    points = torch.from_numpy(points).type(tensor_type)

    deformed_points = inverse_register_points(traj_cp, traj_alpha, points, kernel_width)
    
    #deformed_points should be of size (nr*nc,d)
    return interpolate_image(image, deformed_points)


def LDDMM(Ireca,Iref,landmarks_reca,landmarks_ref,niter,kernel_width,gamma,eps):
    
    '''
    This is the principal function, which computes gradient descent to minimize error and find optimal trajectories for control points, alpha.
    ------
    Ireca : image to register, in 3 dimensions, of size (nr,nc,k)
    Iref : image of reference (red anchor), that you want to reach, of size (nr,nc,k)
    landmarks_reca : array of size (n_landmarks,2)
    landmarks_ref : array of size (n_landmarks,2)
    niter: number of iterations of the algorithm to optimize trajectories
    kernel_width : value
    gamma : value
    eps : coefficient in step of gradient descent
    ------
    returns the registered image, registered landmarks, and also optimized control points and alpha trajectories

    '''
    
    cp = torch.from_numpy(landmarks_reca).type(tensor_type)
    cp_ref = torch.from_numpy(landmarks_ref).type(tensor_type)
    Im_reca = torch.from_numpy(Ireca.copy()).type(tensor_type)
    Im_ref = torch.from_numpy(Iref).type(tensor_type)
    
    alpha = torch.zeros(cp.size()).type(tensor_type)
    alpha.requires_grad_(True)

    for it in range(niter):
        
        #### Compute an estimation of control points and alpha trajectories
        traj_cp, traj_alpha = shoot(cp, alpha, kernel_width, n_steps=10)
        
        ##### Registration of the landmarks
        deformed_points = register_points(traj_cp, traj_alpha, cp, kernel_width)[-1]
        
        ##### Computation of the error, function to minimize
        error = torch.sum((deformed_points.contiguous().view(-1) - cp_ref.contiguous().view(-1)) ** 2) + gamma * torch.sum(torch.mm(alpha.T,torch.mm(gaussian_kernel(cp,cp,kernel_width), alpha)))
        
#         error = torch.mean((deformed_points-cp_ref)**2) + gamma * torch.sum(torch.mm(momenta.T,torch.mm(inverse_metric(cp,cp,kernel_width), momenta)))
       
        error.backward()
        
        eps_mom = eps/np.sqrt(np.sum(alpha.grad.numpy() ** 2))
        with torch.no_grad():
            alpha -=  eps_mom * alpha.grad  
        alpha.grad.zero_()

    #### Inversed of phi to register Im_reca
    registered_image = register_image(traj_cp, traj_alpha, Im_reca, kernel_width) 
    registered_cp = deformed_points.detach().numpy()
    
    

    return registered_image,registered_cp,traj_cp,traj_alpha