import numpy as np
import transforms3d as tf3

def theta2vec(theta):
    ''' Convert an angle (in radians) to a unit vector in that angle around Z '''
    #return np.array([np.cos(theta), np.sin(theta), 0.0])
    ai, aj, ak = tf3.euler.axangle2euler([0, 0, 1], theta, axes='sxyz')
    return tf3.euler.euler2mat(ai, aj, ak, axes='sxyz')


def quat2zalign(quat):
    ''' From quaternion, extract z_{ground} dot z_{body} '''
    # z_{body} from quaternion [a,b,c,d] in ground frame is:
    # [ 2bd + 2ac,
    #   2cd - 2ab,
    #   a**2 - b**2 - c**2 + d**2
    # ]
    # so inner product with z_{ground} = [0,0,1] is
    # z_{body} dot z_{ground} = a**2 - b**2 - c**2 + d**2
    a, b, c, d = quat
    return a**2 - b**2 - c**2 + d**2

def get_epsilon(cfg, step):
    return cfg['epsilon_final'] + cfg['epsilon_start'] - cfg['epsilon_final'] * np.exp(-1. * step / cfg['epsilon_decay'])