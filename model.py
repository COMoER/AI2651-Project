import numpy as np
from vad_utils import read_label_from_file,prediction_to_vad_label
from evaluate import get_metrics
import os
import matplotlib.pyplot as plt
import librosa
from tqdm import tqdm

#过零率
def sgn(x:np.ndarray):
    y = np.ones(x.shape,np.int32)
    y[x < 0] = 0
    return y

def ZSR(frame:np.ndarray):

    sub_frame = sgn(frame[:,1:])-sgn(frame[:,:-1])
    y = np.mean(np.abs(sub_frame),axis = 1)

    return y

#门函数
def rectangle(x:np.ndarray):
    assert(len(x.shape) == 1)
    return x

def hamming(x:np.ndarray):
    assert(len(x.shape) == 1)
    n = x.shape[0]
    w = 0.54-0.46*np.cos(2*np.pi*(n-1-np.arange(n))/(n-1))
    return x*w

#
def frame(x:np.ndarray,sr,length:int,hop_length:int = None,gate = rectangle):
    '''
    to divide the data into frames

    Args:
        x: input_data (N,)
        length: frame_size
        hop_length: frame_shift
        gate: gate_function

    Returns:
        the frame group required,the time respect to frames

    '''
    if hop_length == None:
        hop_length = length//4

    start = 0
    stop = length
    frame_group = []
    frame_time = []

    while(start < x.shape[0]):
        if stop < x.shape[0]:
            frame_group.append(gate(x[start:stop]))
        else:
            pad_width = (0, stop - x.shape[0])
            frame_group.append(gate(np.pad(x[start:], pad_width, 'constant', constant_values=(0, 0))))
        frame_time.append(start + length//2)
        start = start + hop_length
        stop = start + length


    return np.float32(frame_group),np.float32(frame_time)/sr


#短时能量
def short_time_energy(frame_group):
    return np.sum(frame_group**2,axis = 1)


def spectral_centroid(frames,length,sample_rate):

    fft_data = np.fft.rfft(frames, axis=1)
    # 频谱质心
    freq = np.fft.rfftfreq(length, 1 / sample_rate)
    fft_data_abs = np.abs(fft_data)
    return np.sum(fft_data_abs * freq, axis=1)

def sigmoid(x:np.ndarray,k):
    # x>= 0
    return 2.0/(1.0+np.exp(-k*x))-1

def VAD(feature:dict,weight:dict,k:dict,threshold):
    '''
        main function for VAD

    Args:
        feature: dict for the names of the features
        weight: weight dict for the features
        k: sigmoid k argument for the features
        threshold: threshold for the confidence

    Returns:
        0,1 prediction according to threshold, float confidence
    '''
    feature_names = list(feature.keys())

    state_rate = 1

    N = feature[feature_names[0]].shape[0]


    weights = np.zeros(N)

    for f in feature_names:
        w = (sigmoid(feature[f],k[f]) if k[f] != 0 else feature[f])
        weights += weight[f]*w
        state_rate -= weight[f]
    # state_rate is the weight of the state machine
    # 简单状态机
    pre_weights = weights.copy()
    for i in range(1,N):
        #判断在原confidence中其前一帧是否大于阈值
        if(pre_weights[i-1]>threshold):
            #若是则加上状态机的阈值
            weights[i] += 1*state_rate
        else:
            #若不是则施加一定的惩罚
            weights[i] = max(weights[i]-0.4*state_rate,0)


    result = np.zeros(N)
    result[weights > threshold] = 1

    # 平滑滤波
    result = mean_pooling(result, 30, 0.4)

    return result,weights

#计算Naive准确率
def precise(pred,label_y):

    return np.sum(pred == label_y)/pred.shape[0]

#计算在不同阈值下的P和R，最后没用到
def ROC(pred,label_y):
    recall = []
    fake_positive = []
    label_y = np.array(label_y)
    label_true = np.sum(label_y == 1)
    label_false = np.sum(label_y == 0)
    for threshold in np.arange(100)*0.01:
        pred_true = np.sum(np.logical_and((pred>=threshold)==label_y,label_y == 1))
        pred_false = np.sum(np.logical_and(pred>=threshold,label_y == 0))
        recall.append(pred_true/label_true)
        fake_positive.append(pred_false/label_false)
    return fake_positive,recall
#平滑滤波中的中值滤波函数
def mean_pooling(pred,window_size = 10,threshold:float = 0.5):

    return np.array([(np.sum(pred[i:i+window_size]) > int(np.round(threshold*window_size)))
                      if i+window_size < pred.shape[0] else (np.sum(pred[i:]) > int(np.round(threshold*(pred.shape[0]-i))))
                      for i in range(pred.shape[0]) ],np.int)

def pipeline(data,sample_rate,length,hop_length):
    '''
    提取特征，进行VAD 整个项目的pipeline

    Args:
        data: data (N,1) normalized

    Returns:
        pred: (N,1) 0,1 predictions
        scores: (N,1) float scores
        frame_time: 帧时
    '''
    data_frame, frame_time = frame(data, sample_rate, length, hop_length, gate=rectangle)
    data_frame_hamming, _ = frame(data, sample_rate, length, hop_length, gate=hamming)

    e = short_time_energy(data_frame)

    zsr = ZSR(data_frame)

    sc = spectral_centroid(data_frame_hamming,length,sample_rate)

    plt.plot(np.arange(data.shape[0])/sample_rate,data,'b')
    #
    # plt.plot(frame_time,sklearn.preprocessing.minmax_scale(zsr),'g')
    #
    # plt.plot(frame_time,sklearn.preprocessing.minmax_scale(sc),'r')

    feature = {'e': e, 'zsr': zsr, 'sc': sc}

    weight = {'e': 0.5, 'zsr': 0, 'sc': 0.3}

    k = {'e': 10, 'zsr': 0, 'sc': 0.001}

    pred, scores = VAD(feature, weight, k,threshold=0.3)

    return pred,scores,frame_time

def prediction():
    '''
    对于测试集的预测
    '''

    save_path = "test_label_task1.txt"

    dataset = "wavs/test"

    # default arguments
    sr = 22050
    length = 256
    hop_length = 64
    with open(save_path,'w') as f:
        for path in tqdm(os.listdir(dataset)):
            wav_path = os.path.join(dataset, path)
            data, sample_rate = librosa.load(wav_path, sr=sr)

            pred, score,frame_time = pipeline(data, sample_rate, length, hop_length)

            pred_time = prediction_to_vad_label(pred, length / sr, hop_length / sr)

            f.write(f"{path.split('.')[0]:s} {pred_time:s}\n")


def dev_prediction():
    '''
    对于开发集的预测
    '''
    dataset = "wavs/dev"

    # default arguments
    sr = 22050
    length = 256
    hop_length = 64

    label: dict = read_label_from_file(frame_size=length / sr,
                                       frame_shift=hop_length / sr)  # a dict keys are file_path items are sequence of VAD

    aucs = []
    eers = []
    prec = []


    pbar = tqdm(enumerate(list(label.keys())))
    for i, path in pbar:

        label_y = label[path]

        wav_path = os.path.join(dataset, path + ".wav")
        data, sample_rate = librosa.load(wav_path, sr=sr)

        pred, weights,frame_time = pipeline(data,sample_rate,length,hop_length)

        # label未对最后一个有声时间段后进行标注，故补充

        n = len(label_y)

        while (n < frame_time.shape[0]):
            label_y.append(0)
            n += 1

        x, y = ROC(weights, label_y)

        pres = precise(pred,label_y)
        prec.append(pres)

        plt.plot(frame_time,pred,'r')
        plt.legend(["signal", "pred"])
        #
        plt.show()
        plt.waitforbuttonpress(10)
        plt.close()

        # m = get_metrics(pred,label_y)

        auc, eer = get_metrics(weights, label_y)

        aucs.append(auc)
        eers.append(eer)
        pbar.set_postfix({f'auc of {path}': f"{auc:0.4f}", f"eer of {path}": f"{eer * 100:0.4f}%",f'precision of {path}': f"{pres*100:0.4f}%"})

    #显示各个指标
    print(
        f"average auc is {float(np.float32(aucs).mean()):0.4f} , average eer is {float(np.float32(eers).mean()) * 100:0.4f}% and average of precision is {float(np.float32(prec).mean())*100:.4f}%")
    plt.subplot(1, 3, 1)
    plt.plot(range(len(list(label.keys()))), aucs, '-')
    plt.title("auc")
    plt.subplot(1, 3, 2)
    plt.plot(range(len(list(label.keys()))), eers, '-')
    plt.title("eer")
    plt.subplot(1, 3, 3)
    plt.plot(range(len(list(label.keys()))), prec, '-')
    plt.title("precision")
    plt.show()
    plt.waitforbuttonpress(10)
    plt.close()

if __name__ == "__main__":
    #本代码使用了相对路径，确保wavs和data文件夹与代码同根目标时再跑
    dev_prediction()
    # prediction()







