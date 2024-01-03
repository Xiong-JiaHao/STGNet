from scipy.signal import butter, filtfilt

FrameSize = 128
ClipFramesLen = 160
ClipStep = 60


def bandpass_filter(data, lowcut=0.8, highcut=3.0, fps=30):
    # fs = fps  # 采样频率
    # nyq = 0.5 * fs  # Частота Найквиста
    # low = float(lowcut) / float(nyq)
    # high = float(highcut) / float(nyq)
    order = 8.0
    b, a = butter(order, [lowcut, highcut], fs=fps, btype='band')
    bandpass = filtfilt(b, a, data)
    return bandpass
