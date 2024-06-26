
from torch.utils.data import DataLoader, Dataset
from scipy.io import wavfile
import numpy as np
from tqdm import tqdm
from typing import List
import torch

EPS = 1e-6

class waveformSet(Dataset):


    def __init__(self,audioFiles: List[str],audioROIs: List[str],overlap=0.5,maxSegs: int =1000,seg_lim=20000,seed=1738):


        self.chunks = []
        self.overlap = overlap
        self.seg_lim=seg_lim
        totalFiles = len(audioFiles)
        generator = np.random.default_rng(seed=seed)
        order = generator.choice(totalFiles,totalFiles,replace=False)

        def getChunks(onset,offset,currentCount):
            """
            gets all chunks for current onsets and offsets. 
            raises a flag if we've hit our maximum number of segments
            """
            tmpChunks = []
            for on,off in zip(onset,offset):
                    
                if currentCount >= maxSegs:
                    return tmpChunks,currentCount,True
                segOn,segOff = np.searchsorted(audio_times,on),np.searchsorted(audio_times,off)

                tmpChunks += self._separate_into_chunks(audio[segOn:segOff])
                currentCount += 1
            return tmpChunks,currentCount,False
        
        totalSegs = 0

        for ii,ind in tqdm(enumerate(order),desc='Pre-processing audio'):
            aud = audioFiles[ind]
            onoffs = audioROIs[ind]
            #print(onoffs)

            fs,audio = wavfile.read(aud)
            if np.amax(audio) > 1:
                audio = audio/32768 # if audio is coded with ints, divide by max int value to convert to float
            if ii == 0:
                self.fs = fs
            audio_times = np.linspace(0,len(audio)/fs,len(audio))
            #print(np.loadtxt(onoffs))
            #onset,offset = np.loadtxt(onoffs)
            onset,offset = np.loadtxt(onoffs,delimiter=' ',unpack=True)
            chunks,totalSegs,flag = getChunks(onset,offset,totalSegs)
            self.chunks += chunks
            if flag:
                print("max number of segments reached!")
                break

        print("Getting positive pair indices")  
        self.eligible_inds = self._crosscorr_chunks()

    def __len__(self):

        return len(self.chunks)
    
    def __getitem__(self,index):

        audioSample = self.chunks[index]
        validInds = self.eligible_inds[index]
        ind2 = np.random.choice(np.argwhere(validInds==1).squeeze(),1)[0]
        sample2 = self.chunks[ind2]

        return (torch.from_numpy(audioSample).type(torch.FloatTensor),
                torch.from_numpy(sample2).type(torch.FloatTensor))
        
    def _separate_into_chunks(self,segment,chunklen_s=0.01):
        
        currTimes = np.arange(0,len(segment)/self.fs,1/self.fs)
        chunkLen = int(np.round(self.fs * chunklen_s))
        segOns = np.arange(0,len(segment)/self.fs,chunklen_s*(1-self.overlap))
        newChunks = []
        #print(chunkLen)
        for on in segOns:
            onInd,offInd = np.searchsorted(currTimes,on),np.searchsorted(currTimes,on+chunklen_s)
            
            if offInd - onInd >= chunkLen:
                newChunks.append(segment[onInd:offInd][:chunkLen])
          
        return newChunks
    
    def _crosscorr_chunks(self):
        """
        this will be VERY inefficient! after this works, see if you can vectorize
        -- maybe can vectorize based on how many chunks we have?
        """
        allInds = []
        if len(self.chunks) >= self.seg_lim:

            tmpChunks = np.array(self.chunks)
            tmpChunks += EPS * np.random.normal(size=tmpChunks.shape)
            sdFull = np.nanstd(tmpChunks,axis=-1)
            scale = tmpChunks.shape[-1]
            centeredFull = tmpChunks - np.nanmean(tmpChunks,axis=-1,keepdims=True)
            for ii,c1 in tqdm(enumerate(centeredFull),desc=f'being inefficient: more thank {self.seg_lim} segs'):
                chunkCorrs = []
                #centered1 = c1 - np.nanmean(c1)
                sd1 = sdFull[ii]#np.nanstd(centered1)
                cov = centeredFull @ c1 /scale
                chunkCorrs = cov/(sd1 * sdFull)
                #print(chunkCorrs.shape)
                chunkCorrs[ii] = -1
                assert np.all(chunkCorrs <= 1) and np.all(chunkCorrs >= -1), print(f"corrs outside of valid range: {np.amin(chunkCorrs),np.amax(chunkCorrs)}")
                corrsSorted = np.sort(chunkCorrs,axis=None)
                cutoff = corrsSorted[int(round(0.9*len(chunkCorrs)))]
                allInds.append( chunkCorrs >= cutoff)
                """
                for jj,c2 in enumerate(self.chunks):
                    if ii != jj:
                        centered2 = c2 - np.nanmean(c2)
                        sd2 = np.nanstd(centered2)
                        chunkCorrs.append((centered1.T @ centered2)/(EPS+ sd1  * sd2 * len(centered2)))
                    else:
                        chunkCorrs.append(-1)

                
                
                chunkCorrs = np.array(chunkCorrs)
                assert np.all(chunkCorrs <= 1) and np.all(chunkCorrs >= -1), print("corrs outside of valid range")
                corrsSorted = np.sort(chunkCorrs,axis=None)
                cutoff = corrsSorted[int(round(0.9*len(chunkCorrs)))]
                allInds.append( chunkCorrs >= cutoff)
                """
        else:
            print("being efficient")

            tmpChunks = np.array(self.chunks)
            tmpChunks += EPS * np.random.normal(size=tmpChunks.shape)
            #tmpChunks += EPS * np.random.randn(*tmpChunks.shape) # for numerical stability
            corrs = np.corrcoef(tmpChunks)
            np.fill_diagonal(corrs,-1)
            corrs = np.nan_to_num(corrs,-1) # replace nans with lowest val -- need better fix for periods of pure silence (sd = 0)
            for row in tqdm(corrs,desc='Finding valid inds'):
                assert np.all(row <= 1) and np.all(row >= -1), print(f"corrs outside of valid range: {row}")
                corrsSorted = np.sort(row,axis=None)
                cutoff = corrsSorted[int(round(0.9*len(row)))]
                allInds.append( row >= cutoff)



        
        return allInds
                
def specSet(Dataset):

    def __init__(self):

        pass 