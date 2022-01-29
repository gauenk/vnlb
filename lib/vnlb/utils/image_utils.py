
import numpy as np

def est_sigma(noisy):
    return 0.

def idx2coords(idx,color,height,width):

    # -- get shapes --
    whc = width*height*color
    wh = width*height

    # -- compute coords --
    t = (idx      ) // whc
    c = (idx % whc) // wh
    y = (idx % wh ) // width
    x = idx % width

    return t,c,y,x

def coords2idx(ti,hi,wi,color,height,width):
    pix = ti * width * height * color
    pix += hi * width
    pix += wi
    return pix

def yuv2rgb_cpp(burst):
    # burst.shape = (...,c,h,w) in YUV

    # -- n-dim -> 4-dim --
    shape = burst.shape
    shape_rs = (-1,shape[-3],shape[-2],shape[-1])
    burst = burst.reshape(shape_rs)

    # -- color convert --
    burst = apply_yuv2rgb(burst)

    # -- 4-dim -> n-dim --
    burst = burst.reshape(shape)
    return burst

def apply_yuv2rgb(burst):
    """
    rgb -> yuv [using the "cpp repo" logic]
    """
    burst_rgb = []
    # burst = rearrange(burst,'t c h w -> t h w c')
    t,h,w,c = burst.shape
    for ti in range(t):

        # -- init --
        image = burst[ti]
        image_rgb = np.zeros_like(image)
        weights = [1./np.sqrt(3),1./np.sqrt(2),np.sqrt(2.)/np.sqrt(3)]
        w = weights

        # -- rgb -> yuv --
        image_rgb[0] = w[0] * image[0] + w[1] * image[1] + w[2] * 0.5 * image[2]
        image_rgb[1] = w[0] * image[0] - w[2] * image[2]
        image_rgb[2] = w[0] * image[0] - w[1] * image[1] + w[2] * 0.5 * image[2]

        # -- append --
        burst_rgb.append(image_rgb)
    burst_rgb = np.stack(burst_rgb)

    return burst_rgb

def apply_color_xform_cpp(burst):
    """
    rgb -> yuv [using the "cpp repo" logic]
    """
    burst_yuv = []
    # burst = rearrange(burst,'t c h w -> t h w c')
    t,h,w,c = burst.shape
    for ti in range(t):

        # -- init --
        image = burst[ti]
        image_yuv = np.zeros_like(image)
        weights = [1./np.sqrt(3),1./np.sqrt(2),np.sqrt(2.)*2./np.sqrt(3)]

        # -- rgb -> yuv --
        image_yuv[0] = weights[0] * (image[0] + image[1] + image[2])
        image_yuv[1] = weights[1] * (image[0] - image[2])
        image_yuv[2] = weights[2] * (.25 * image[0] - 0.5 * image[1] + .25 * image[2])

        # -- append --
        burst_yuv.append(image_yuv)
    burst_yuv = np.stack(burst_yuv)

    return burst_yuv

def apply_color_xform(burst):
    """
    rgb -> yuv
    """
    burst_yuv = []
    burst = rearrange(burst,'t c h w -> t h w c')
    t,h,w,c = burst.shape
    for ti in range(t):
        image_yuv = cv2.cvtColor(burst[ti], cv2.COLOR_RGB2YUV)
        burst_yuv.append(image_yuv)
    burst_yuv = np.stack(burst_yuv)
    burst_yuv = rearrange(burst_yuv,'t h w c -> t c h w')
    return burst_yuv


def numpy_div0( a, b, fill=np.nan ):
    # https://stackoverflow.com/questions/26248654/how-to-return-0-with-divide-by-zero
    """ a / b, divide by 0 -> `fill`
        div0( [-1, 0, 1], 0, fill=np.nan) -> [nan nan nan]
        div0( 1, 0, fill=np.inf ) -> inf
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
    if np.isscalar( c ):
        return c if np.isfinite( c ) \
            else fill
    else:
        c[ ~ np.isfinite( c )] = fill
        return c

