import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import wavfile
import numpy as np
import torch
from umap import UMAP
from scipy.signal import stft
 

def get_chunks(audio,targets,trues,chunkLen):

    chunks = []
    for on,off in zip(targets[:-1],targets[1:]):
        onInd,offInd = np.searchsorted(trues,on),np.searchsorted(trues,off)
        tmpSeg = audio[onInd:offInd]
        if offInd - onInd >= chunkLen:
            tmpSeg = tmpSeg[:chunkLen]
            pt_aud = torch.from_numpy(tmpSeg).type(torch.FloatTensor)
            #print(pt_aud.shape)
            chunks.append(pt_aud)

    if len(chunks)> 0:
        return torch.vstack(chunks)[:,None,:]
    else:
        print(f"segment too short? {targets}")
        return torch.empty([])

def embed_audio_file(af,model,device,roif=''):

    model_dt = 0.01
    fs, aud = wavfile.read(af)
    if np.amax(aud) > 1:
        aud = aud/32768 # if audio is coded with ints, divide by max int value to convert to float
  
    n = len(aud)
    len_s = n/fs

    audioTimes = np.linspace(0,len_s,n)
    targetTimes = np.linspace(0,len_s, int(round(len_s/model_dt)))
    chunkLen = int(np.round(fs * 0.01))

    #allLatents = []
    #segLatents = []

    allAudioChunks = get_chunks(aud,targetTimes,audioTimes,chunkLen).cuda(device)
    
    allAudioLatents = model.encoder(allAudioChunks).detach().cpu().numpy()

    if roif != '':
        onsets,offsets = np.loadtxt(roif,unpack=True)
        allSegLatents = []
        allSegChunks = []
        #print(1/model_dt)
        for on,off in zip(onsets,offsets):
            targetTimes = np.linspace(on,off,int(round((off - on)/model_dt)))

            segChunks = get_chunks(aud,targetTimes,audioTimes,chunkLen).cuda(device)
            if len(segChunks.shape) != 0:
                segLatents = model.encoder(segChunks).detach().cpu().numpy()
                allSegChunks.append(segChunks.detach().cpu().numpy().squeeze())
                allSegLatents.append(segLatents)
            #for cOn,cOff in zip(targetTimes[:-1],targetTimes[1:]):
            #    onInd,offInd = np.searchsorted

        return allAudioLatents,allSegLatents,allSegChunks
    else:
        return allAudioLatents
    
def plot_pc_embeddings(audioLatents,len_s,roiLatents=[],roiAudio=[]):


    audMu = np.nanmean(audioLatents,axis=0)
    centeredAud = audioLatents - audMu
    cov = centeredAud.T @ centeredAud 
    _,vh = np.linalg.eigh(cov)
    audPCs = audioLatents @ vh.T

    cAx = np.linspace(0,len_s,centeredAud.shape[0])
    print(cAx.shape)
    print(centeredAud.shape)

    ax = plt.gca()
    ax.scatter(audPCs[:,0],audPCs[:,1],c=cAx,cmap='flare_r')

    if len(roiLatents) > 0:
        assert len(roiLatents) == len(roiAudio),print("must have audio for each ROI!")
    
        for lat,aud in zip(roiLatents,roiAudio):

            pcLat = (lat - audMu) @ vh.T    
            ax.plot(pcLat[:,0],pcLat[:,1])

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()
    plt.close()
        
def plot_umap_embeddings(audioLatents,len_s,roiLatents=[],roiAudio=[],fs=44100):


    audMu = np.nanmean(audioLatents,axis=0)
    centeredAud = audioLatents - audMu
    transform = UMAP(n_neighbors=5,n_components=2,min_dist=0.05)
    audLD = transform.fit_transform(centeredAud)
    
    #cov = centeredAud.T @ centeredAud 
    #_,vh = np.linalg.eigh(cov)
    #audPCs = audioLatents @ vh.T

    cAx = np.linspace(0,len_s,centeredAud.shape[0])
    print(cAx.shape)
    print(centeredAud.shape)

    fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
    ax1.scatter(audLD[:,0],audLD[:,1],c=cAx,cmap='flare_r')

    if len(roiLatents) > 0:
        assert len(roiLatents) == len(roiAudio),print("must have audio for each ROI!")
    
        choice = np.random.choice(len(roiLatents),1)[0]
        lat,aud = roiLatents[choice],roiAudio[choice]
        #for lat,aud in zip(roiLatents,roiAudio):

        latLD = transform.transform(lat - audMu)    
        ax1.plot(latLD[:,0],latLD[:,1])
        

        fullClip = aud.flatten()
        f,t,sxx = stft(fullClip,fs=fs,nperseg=256,noverlap=128,nfft=256)

        sxx = np.log(np.abs(sxx) + 1e-10)
        ax2.imshow(sxx,origin='lower',extent=[t[0],t[-1],f[0]/1e3,f[-1]/1e3],aspect='auto')
        #print(t[0])
        #print(t[1])
    ax1.set_xlabel("UMAP 1")
    ax1.set_ylabel("UMAP 2")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    plt.show()
    plt.close()