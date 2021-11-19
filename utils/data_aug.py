import mxnet.ndarray as nd 
import numpy as np 
from PIL import Image

def random_pca_lighting(src, alphastd=0.1, eigval=None, eigvec=None):
    """Apply random pca lighting noise to input image.

    Parameters
    ----------
    img : mxnet.nd.NDArray
        Input image with HWC format.
    alphastd : float
        Noise level [0, 1) for image with range [0, 255].
    eigval : list of floats.
        Eigen values, defaults to [55.46, 4.794, 1.148].
    eigvec : nested lists of floats
        Eigen vectors with shape (3, 3), defaults to
        [[-0.5675, 0.7192, 0.4009],
         [-0.5808, -0.0045, -0.8140],
         [-0.5836, -0.6948, 0.4203]].

    Returns
    -------
    mxnet.nd.NDArray
        Augmented image.

    """
    if alphastd <= 0:
        return src

    if eigval is None:
        eigval = np.array([55.46, 4.794, 1.148])
    if eigvec is None:
        eigvec = np.array([[-0.5675, 0.7192, 0.4009],
                           [-0.5808, -0.0045, -0.8140],
                           [-0.5836, -0.6948, 0.4203]])
    
    alpha = np.random.normal(0, alphastd, size=(3,))
    rgb = np.dot(eigvec * alpha, eigval)
    src = np.cast['float64'](src)
    src += rgb
    src = np.cast['uint8'](src)
    return src


if __name__ == '__main__':
    image = Image.open("/data/jiangmingchao/data/img-00005.jpeg")
    if image.mode != "RGB":
        image = image.convert("RGB")
    image_arr = np.array(image)
    image_nd = nd.array(image_arr)
    image_dst = random_pca_lighting(image_arr)
    
    # print(image_dst)
    image = Image.fromarray(image_dst)
    image.save("./pca.jpg")
    
    