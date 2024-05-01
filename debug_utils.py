
def min_max_norm(tmp):
    return (tmp-tmp.min())/(tmp.max()-tmp.min())

def depth2img(depth):
    """depth is [B,h,w], transfer it from CUDA to cpu, and convert it to normalized depth

    Args:
        depth (tensor): shape is [B,h,w]
    """
    tmp = depth.detach().cpu().numpy()[0]
    return min_max_norm(tmp)
