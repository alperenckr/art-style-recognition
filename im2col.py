def im2col_index(input_shape, HF, WF, padding, stride):
    # get input size
    H, W, D, N = input_shape
    # get output size
    out_h = 0
    out_w = 0
    if type(padding) is int:
        out_h = (H + 2 * padding - HF) / stride + 1
        out_w = (W + 2 * padding - WF) / stride + 1
    else:
        out_h = (H + padding[0] + padding[1] - HF) / stride + 1
        out_w = (W + padding[2] + padding[3] - WF) / stride + 1
    # for row index, compute the first index of the first HF * WF block
    r0 = np.repeat(np.arange(HF), WF)
    r0 = np.tile(r0, D)
    # then compute the bias of each block
    r_bias = stride * np.repeat(np.arange(out_h), out_w)
    # then the row index is the r0 + r_bias
    r = r0.reshape(-1, 1) + r_bias.reshape(1, -1)

    # the same to the col index
    c0 = np.tile(np.arange(WF), HF * D)
    c_bias = stride * np.tile(np.arange(out_w), out_h)
    c = c0.reshape(-1, 1) + c_bias.reshape(1, -1)

    # then the dimension index
    d = np.repeat(np.arange(D), HF * WF).reshape(-1, 1)

    return (r, c, d)

def im2col(x, HF, WF, padding, stride):
    # paddingding
    x_paddingded = None
    if type(padding) is int:
        x_paddingded = np.padding(x, ((padding, padding), (padding, padding), (0, 0), (0, 0)), mode='constant')
    else:
        x_paddingded = np.padding(x, ((padding[0], padding[1]), (padding[2], padding[3]), (0, 0), (0, 0)), mode='constant')
    r, c, d = im2col_index(x.shape, HF, WF, padding, stride)
    cols = x_paddingded[r, c, d, :]
    cols = cols.reshape(HF * WF * x.shape[2], -1)
    return cols


def col2im(cols, input_shape, HF, WF, padding, stride):
    # get input size
    H, W, D, N = input_shape
    H_paddingded = 0
    W_paddingded = 0
    if type(padding) is int:
        H_paddingded, W_paddingded = H + 2 * padding, W + 2 * padding
    else:
        H_paddingded, W_paddingded = H + padding[0] + padding[1], W + padding[2] + padding[3]
    x_paddingded = np.zeros((H_paddingded, W_paddingded, D, N), dtype=cols.dtype)
    r, c, d = im2col_index(input_shape, HF, WF, padding, stride)
    cols_reshaped = cols.reshape((HF * WF * D, -1, N))
    np.add.at(x_paddingded, (r, c, d, slice(None)), cols_reshaped)
    if padding == 0:
        return x_paddingded
    elif type(padding) is int:
        return x_paddingded[padding:-padding, padding:-padding, :, :]
    else:
        return x_paddingded[padding[0]:-padding[1], padding[2]:-padding[3], :, :]