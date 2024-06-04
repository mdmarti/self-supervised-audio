
from torch.utils.data import DataLoader, Dataset
from scipy.io import wavfile
import numpy as np
from tqdm import tqdm
from typing import List
import torch

def waveformSet(Dataset):


    def __init__(self,audioFiles: List[str],audioROIs: List[str],maxSegs: int =1000,seed=1738):


        self.chunks = []
        totalFiles = len(audioFiles)
        order = np.random.choice(totalFiles,maxSegs,replace=False,seed=seed)
        for ii in tqdm(order,desc='Pre-processing audio'):
            aud = audioFiles[ii]
            onoffs = audioROIs[ii]
            fs,audio = wavfile.read(aud)
            if ii == 0:
                self.fs = fs
            audio_times = np.linspace(0,len(audio)/fs,len(audio))
            onset,offset = np.loadtxt(onoffs)
            for on,off in zip(onset,offset):
                segOn,segOff = np.searchsorted(audio_times,on),np.searchsorted(audio_times,off)

                self.chunks += self._separate_into_chunks(audio[segOn:segOff])

        self.eligible_inds = self._crosscorr_chunks()

    def __len__(self):

        return len(self.chunks)
    
    def __getitem__(self,index):

        audioSample = self.chunks[index]
        validInds = self.eligible_inds[index]
        ind2 = np.random.choice(np.argwhere(validInds==1).squeeze(),1)
        sample2 = self.chunks[ind2]

        return (torch.from_numpy(audioSample).type(torch.FloatTensor),
                torch.from_numpy(sample2).type(torch.FloatTensor))
        
    def _separate_into_chunks(self,segment,chunklen_s=0.01):
        
        currTimes = np.arange(0,len(segment)/self.fs,1/self.fs)
        segOns = np.arange(0,len(segment)/self.fs,chunklen_s)
        newChunks = []
        for on in segOns:
            onInd,offInd = np.searchsorted(currTimes,on),np.searchsorted(currTimes,on+chunklen_s)
            newChunks.append(segment[onInd:offInd])
          
        return newChunks
    
    def _crosscorr_chunks(self):
        """
        this will be VERY inefficient! after this works, see if you can vectorize
        -- maybe can vectorize based on how many chunks we have?
        """

        allInds = []
        for ii,c1 in enumerate(self.chunks):
            chunkCorrs = []
            centered1 = c1 - np.nanmean(c1)
            sd1 = np.nanstd(centered1)
            for jj,c2 in enumerate(self.chunks):
                if ii != jj:
                    centered2 = c2 - np.nanmean(c2)
                    sd2 = np.nanstd(centered2)
                    chunkCorrs.append((centered1.T @ centered2)/(sd1  * sd2 * len(centered2)))
                else:
                    chunkCorrs.append(0)

            assert np.all(chunkCorrs <= 1) and np.all(chunkCorrs >= -1), print("corrs outside of valid range")
            
            chunkCorrs = np.array(chunkCorrs)
            corrsSorted = np.sort(chunkCorrs,axis=None)
            cutoff = corrsSorted[int(round(0.9*len(chunkCorrs)))]
            allInds.append( chunkCorrs >= cutoff)
        
        return allInds
                
def specSet(Dataset):

    def __init__(self):

        pass 