import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'True', 'true', 'T', 't', 'Y', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'false', 'F', 'f', 'N', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def megab(v):
    return v * 1024 * 1024

def gigab(v):
    return megab(v) * 1000



