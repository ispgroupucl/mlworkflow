import numpy as np


def palette(N, zero=[0,0,0], *, cmap=None, seed=None):
    if cmap is None:
        from matplotlib import pyplot as plt
        cmap = plt.cm.jet
    palette = cmap(np.linspace(0,1,N))[:,:3]
    palette = (palette * 255).astype(np.uint8)
    if seed is not None:
        np.random.RandomState(seed=abs(seed)%(1<<32)).shuffle(palette)
    return np.concatenate(([zero], palette), axis=0)


def arrays_to_rgba(r=None, g=None, b=None, alpha=None, scale=1):
    """Merge arrays to end up with 4 UINT8 channels from individual arrays"""
    _first = [np.asarray(x) for x in (r, g, b) if x is not None][0]
    if r is None: r = np.zeros_like(_first)
    if g is None: g = np.zeros_like(_first)
    if b is None: b = np.zeros_like(_first)
    if alpha is None:
        alpha = 255
    if isinstance(alpha, (float, int)):
        alpha = np.ones_like(_first) * alpha
    else:
        alpha = alpha * scale
    return np.stack((r*scale, g*scale, b*scale, alpha), axis=-1).astype(np.uint8)


def array_to_rgba(image, *, CHW=False, scale=1):
    """Transform an input array to end up with 4 UINT8 channels"""
    image = np.asarray(image)
    if len(image.shape) == 2:
        image = image[...,None]
    elif CHW:
        image = np.moveaxis(image, 0, 2)
    if scale != 1:
        image = image*scale
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    assert len(image.shape) == 3
    if image.shape[-1] == 1:
        image = np.tile(image, [1, 1, 3])
    if image.shape[-1] == 2:
        image = np.concatenate((image,
                                np.zeros([*image.shape[:2], 1], dtype=np.uint8)),
                               axis=-1)
    if image.shape[-1] == 3:
        image = np.concatenate((image, 255*np.ones(image.shape[:-1], dtype=np.uint8)[...,None]), axis=-1)
    assert image.shape[-1] == 4

    return image
