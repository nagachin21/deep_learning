import numpy as np

def smooth_curve(x):
    """損失関数のグラフを滑らかにするために用いる

    参考：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """

    Parameter
    ---------
    input_data: （データ数,チャンネル数,高さ,幅）の4次元配列からなる入力データ
    filter_h  : フィルターの高さ
    filter_w  : フィルターの幅
    stride    : ストライド
    pad       : パディング

    Returns
    -------
    col : 2次元配列
    """
    N, C, H, W = input_data.shape
    out_h = int(1 + (H + pad * 2 - filter_h) / stride)
    out_w = int(1 + (W + pad * 2 - filter_w) / stride)

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad)], 'constant')
    col = np.zeros(N, C, filter_h, filter_w, out_h, out_w)

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transponse(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col 
