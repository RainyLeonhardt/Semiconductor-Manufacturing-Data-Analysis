import warnings
import imblearn
from sklearn.utils import resample
from imblearn.over_sampling import * 
import numpy as np
import pandas as pd
import torch
from packaging import version
from torch import optim
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional

from CSR_Synthesizer.data_sampler import DataSampler
from CSR_Synthesizer.data_transformer import DataTransformer
from CSR_Synthesizer.synthesizers.base import BaseSynthesizer
from CSR_Synthesizer.synthesizers.ctgan import CTGANSynthesizer
from CSR_Synthesizer.synthesizers.smote import SMOTE

class SRE_Synthesizer():
  # def __init__(self, target_data, sample_numbers=None, epochs=None):
  #   self.target_data = target_data
  #   self.sample_numbers = sample_numbers
  #   self.epochs= epochs
  def fit(self, target_data, sample_numbers=None, k_neighbors = None):
    self.target_data = target_data
    self.sample_numbers = sample_numbers          
    self.k_neighbors= k_neighbors    
    resampled_total_rows = int(len(target_data.index))
    discrete_columns = list(target_data.columns)
    split_sample_numbers = int(sample_numbers/2)
    SMOTE_Model = SMOTE()
    smote = SMOTE_Model.fit(target_data, sample_numbers=split_sample_numbers, k_neighbors=k_neighbors) 
    
    resampled = resample(target_data,
                  replace=True, # sample with replacement
                  n_samples=split_sample_numbers, # match number in majority class
                  random_state=27) # reproducible results
          #resampled to resampled 生成
    resampled_smote = resample(smote,
                        replace=True, # sample with replacement
                        n_samples=split_sample_numbers, # match number in majority class
                        random_state=27) # reproducible results
          #resampled to resampled 生成
    resampledforsmote = resampled[:resampled_total_rows]      
          #SMOTE to resampled 生成
    SMOTE_Model = SMOTE()
    smote_resampled = SMOTE_Model.fit(resampledforsmote, sample_numbers=split_sample_numbers, k_neighbors=k_neighbors)
          #resampled_resampled 合成
    smoteandresampled = [smote, resampled]
    smoteandresampled = pd.concat(smoteandresampled )
    smoteandresampled_resampled_smote = [smoteandresampled, resampled_smote]
    smoteandresampled_resampled_smote = pd.concat(smoteandresampled_resampled_smote)
    smoteandresampled_resampled_smote_smote_resampled = [smoteandresampled_resampled_smote, smote_resampled]
    smoteandresampled_resampled_smote_smote_resampled = pd.concat(smoteandresampled_resampled_smote_smote_resampled)

    smote_resampled_data = smoteandresampled_resampled_smote_smote_resampled.sample(split_sample_numbers*4)
    smote_resampled_data = smote_resampled_data.drop_duplicates(subset=None, keep='first', inplace=False)
    smote_resampled_data = smote_resampled_data[:sample_numbers]
    return smote_resampled_data

class SR_Synthesizer():
  # def __init__(self, target_data, sample_numbers=None, epochs=None):
  #   self.target_data = target_data
  #   self.sample_numbers = sample_numbers
  #   self.epochs= epochs
  def fit(self, target_data, sample_numbers=None, k_neighbors = None):
    self.target_data = target_data
    self.sample_numbers = sample_numbers
    self.k_neighbors= k_neighbors  
    resampled_total_rows = int(len(target_data.index))
    discrete_columns = list(target_data.columns)
    SMOTE_Model = SMOTE()
    smote = SMOTE_Model.fit(target_data, sample_numbers=sample_numbers, k_neighbors=k_neighbors) 
    
    resampled = resample(target_data,
                  replace=True, # sample with replacement
                  n_samples=sample_numbers, # match number in majority class
                  random_state=27) # reproducible results
          #resampled to resampled 生成
    resampled_smote = resample(smote,
                        replace=True, # sample with replacement
                        n_samples=sample_numbers, # match number in majority class
                        random_state=27) # reproducible results
          #resampled to resampled 生成
    resampledforsmote = resampled[:resampled_total_rows]      
          #SMOTE to resampled 生成
    SMOTE_Model = SMOTE()
    smote_resampled = SMOTE_Model.fit(resampledforsmote, sample_numbers=sample_numbers, k_neighbors=k_neighbors)
          #resampled_resampled 合成
    smoteandresampled = [smote, resampled]
    smoteandresampled = pd.concat(smoteandresampled )
    smoteandresampled_resampled_smote = [smoteandresampled, resampled_smote]
    smoteandresampled_resampled_smote = pd.concat(smoteandresampled_resampled_smote)
    smoteandresampled_resampled_smote_smote_resampled = [smoteandresampled_resampled_smote, smote_resampled]
    smoteandresampled_resampled_smote_smote_resampled = pd.concat(smoteandresampled_resampled_smote_smote_resampled)

    smote_resampled_data = smoteandresampled_resampled_smote_smote_resampled.sample(sample_numbers*4)
    smote_resampled_data = smote_resampled_data.drop_duplicates(subset=None, keep='first', inplace=False)
    smote_resampled_data = smote_resampled_data[:sample_numbers]
    return smote_resampled_data

class CRE_Synthesizer():
  # def __init__(self, target_data, sample_numbers=None, epochs=None):
  #   self.target_data = target_data
  #   self.sample_numbers = sample_numbers
  #   self.epochs= epochs
  def fit(self, target_data, sample_numbers=None, epochs=None):
    self.target_data = target_data
    self.sample_numbers = sample_numbers
    self.epochs= epochs              
    resampled_total_rows = int(len(target_data.index))
    discrete_columns = list(target_data.columns)
    split_sample_numbers = int(sample_numbers/2)
    ctgan = CTGANSynthesizer()
    ctgan.fit(target_data, discrete_columns, epochs=epochs)
    samples = ctgan.sample(split_sample_numbers)
    samples 
    resampled = resample(target_data,
                  replace=True, # sample with replacement
                  n_samples=split_sample_numbers, # match number in majority class
                  random_state=27) # reproducible results
          #resampled to CTGAN 生成
    resampled_ctgan = resample(samples,
                        replace=True, # sample with replacement
                        n_samples=split_sample_numbers, # match number in majority class
                        random_state=27) # reproducible results
          #CTGAN to resampled 生成
    resampledforctgan = resampled[:resampled_total_rows]      
    ctgan_resampled = CTGANSynthesizer()
    ctgan_resampled.fit(resampledforctgan, discrete_columns, epochs=epochs)
    ctgan_resampled = ctgan_resampled.sample(split_sample_numbers)
          #CTGAN_resampled 合成
    resampledandctgan = [samples, resampled]
    resampledandctgan = pd.concat(resampledandctgan)
    resampledandctgan_resampled_ctgan = [resampledandctgan, resampled_ctgan]
    resampledandctgan_resampled_ctgan = pd.concat(resampledandctgan_resampled_ctgan)
    resampledandctgan_resampled_ctgan_ctgan_resampled = [resampledandctgan_resampled_ctgan, ctgan_resampled]
    resampledandctgan_resampled_ctgan_ctgan_resampled = pd.concat(resampledandctgan_resampled_ctgan_ctgan_resampled)

    ctgan_resampled_data = resampledandctgan_resampled_ctgan_ctgan_resampled.sample(split_sample_numbers*4)
    ctgan_resampled_data = ctgan_resampled_data.drop_duplicates(subset=None, keep='first', inplace=False)
    ctgan_resampled_data = ctgan_resampled_data[:sample_numbers]
    return ctgan_resampled_data

class CR_Synthesizer():
  # def __init__(self, target_data, sample_numbers=None, epochs=None):
  #   self.target_data = target_data
  #   self.sample_numbers = sample_numbers
  #   self.epochs= epochs
  def fit(self, target_data, sample_numbers=None, epochs=None):
    self.target_data = target_data
    self.sample_numbers = sample_numbers
    self.epochs= epochs              
    resampled_total_rows = int(len(target_data.index))
    discrete_columns = list(target_data.columns)
    ctgan = CTGANSynthesizer()
    ctgan.fit(target_data, discrete_columns, epochs=epochs)
    samples = ctgan.sample(sample_numbers)
    samples 
    resampled = resample(target_data,
                  replace=True, # sample with replacement
                  n_samples=sample_numbers, # match number in majority class
                  random_state=27) # reproducible results
          #resampled to CTGAN 生成
    resampled_ctgan = resample(samples,
                        replace=True, # sample with replacement
                        n_samples=sample_numbers, # match number in majority class
                        random_state=27) # reproducible results
          #CTGAN to resampled 生成
    resampledforctgan = resampled[:resampled_total_rows]      
    ctgan_resampled = CTGANSynthesizer()
    ctgan_resampled.fit(resampledforctgan, discrete_columns, epochs=epochs)
    ctgan_resampled = ctgan_resampled.sample(sample_numbers)
          #CTGAN_resampled 合成
    resampledandctgan = [samples, resampled]
    resampledandctgan = pd.concat(resampledandctgan)
    resampledandctgan_resampled_ctgan = [resampledandctgan, resampled_ctgan]
    resampledandctgan_resampled_ctgan = pd.concat(resampledandctgan_resampled_ctgan)
    resampledandctgan_resampled_ctgan_ctgan_resampled = [resampledandctgan_resampled_ctgan, ctgan_resampled]
    resampledandctgan_resampled_ctgan_ctgan_resampled = pd.concat(resampledandctgan_resampled_ctgan_ctgan_resampled)

    ctgan_resampled_data = resampledandctgan_resampled_ctgan_ctgan_resampled.sample(sample_numbers*4)
    ctgan_resampled_data = ctgan_resampled_data.drop_duplicates(subset=None, keep='first', inplace=False)
    ctgan_resampled_data = ctgan_resampled_data[:sample_numbers]
    return ctgan_resampled_data

    
class CSE_Synthesizer():
  # def __init__(self, target_data, sample_numbers=None, epochs=None):
  #   self.target_data = target_data
  #   self.sample_numbers = sample_numbers
  #   self.epochs= epochs
  def fit(self, target_data, sample_numbers=None, epochs=None, k_neighbors = None):
    self.target_data = target_data
    self.sample_numbers = sample_numbers
    self.epochs= epochs              
    self.k_neighbors= k_neighbors  
    smote_total_rows = int(len(target_data.index))
    discrete_columns = list(target_data.columns)
    split_sample_numbers = int(sample_numbers/2)
    ctgan = CTGANSynthesizer()
    ctgan.fit(target_data, discrete_columns, epochs=epochs)
    samples = ctgan.sample(split_sample_numbers)
    samples 
    SMOTE_Model = SMOTE()
    smote = SMOTE_Model.fit(target_data, sample_numbers=split_sample_numbers, k_neighbors=k_neighbors) 
          #SMOTE to CTGAN 生成
    SMOTE_Model = SMOTE()
    smote_ctgan = SMOTE_Model.fit(samples, sample_numbers=split_sample_numbers, k_neighbors=k_neighbors)
          #CTGAN to SMOTE 生成
    smoteforctgan = smote[:smote_total_rows]      
    ctgan_smote = CTGANSynthesizer()
    ctgan_smote.fit(smoteforctgan, discrete_columns, epochs=epochs)
    ctgan_smote = ctgan_smote.sample(split_sample_numbers)
          #CTGAN_SMOTE 合成
    smoteandctgan = [samples, smote]
    smoteandctgan = pd.concat(smoteandctgan)
    smoteandctgan_smote_ctgan = [smoteandctgan, smote_ctgan]
    smoteandctgan_smote_ctgan = pd.concat(smoteandctgan_smote_ctgan)
    smoteandctgan_smote_ctgan_ctgan_smote = [smoteandctgan_smote_ctgan, ctgan_smote]
    smoteandctgan_smote_ctgan_ctgan_smote = pd.concat(smoteandctgan_smote_ctgan_ctgan_smote)

    ctgan_smote_data = smoteandctgan_smote_ctgan_ctgan_smote.sample(split_sample_numbers*4)
    ctgan_smote_data = ctgan_smote_data.drop_duplicates(subset=None, keep='first', inplace=False)
    ctgan_smote_data = ctgan_smote_data[:sample_numbers]
    return ctgan_smote_data

class CS_Synthesizer():
  # def __init__(self, target_data, sample_numbers=None, epochs=None):
  #   self.target_data = target_data
  #   self.sample_numbers = sample_numbers
  #   self.epochs= epochs
  def fit(self, target_data, sample_numbers=None, epochs=None, k_neighbors = None):
    self.target_data = target_data
    self.sample_numbers = sample_numbers
    self.epochs= epochs              
    self.k_neighbors= k_neighbors  
    smote_total_rows = int(len(target_data.index))
    discrete_columns = list(target_data.columns)
    ctgan = CTGANSynthesizer()
    ctgan.fit(target_data, discrete_columns, epochs=epochs)
    samples = ctgan.sample(sample_numbers)
    samples 
    SMOTE_Model = SMOTE()
    smote = SMOTE_Model.fit(target_data, sample_numbers=sample_numbers, k_neighbors=k_neighbors) 
          #SMOTE to CTGAN 生成
    SMOTE_Model = SMOTE()
    smote_ctgan = SMOTE_Model.fit(samples, sample_numbers=sample_numbers, k_neighbors=k_neighbors)
          #CTGAN to SMOTE 生成
    smoteforctgan = smote[:smote_total_rows]      
    ctgan_smote = CTGANSynthesizer()
    ctgan_smote.fit(smoteforctgan, discrete_columns, epochs=epochs)
    ctgan_smote = ctgan_smote.sample(sample_numbers)
          #CTGAN_SMOTE 合成
    smoteandctgan = [samples, smote]
    smoteandctgan = pd.concat(smoteandctgan)
    smoteandctgan_smote_ctgan = [smoteandctgan, smote_ctgan]
    smoteandctgan_smote_ctgan = pd.concat(smoteandctgan_smote_ctgan)
    smoteandctgan_smote_ctgan_ctgan_smote = [smoteandctgan_smote_ctgan, ctgan_smote]
    smoteandctgan_smote_ctgan_ctgan_smote = pd.concat(smoteandctgan_smote_ctgan_ctgan_smote)

    ctgan_smote_data = smoteandctgan_smote_ctgan_ctgan_smote.sample(sample_numbers*4)
    ctgan_smote_data = ctgan_smote_data.drop_duplicates(subset=None, keep='first', inplace=False)
    ctgan_smote_data = ctgan_smote_data[:sample_numbers]
    return ctgan_smote_data

class CR_SynthesizerNrows():#把所有資料以n個rows分成各個fold並一次運算一個fold
  # def __init__(self, target_data, sample_numbers=None, epochs=None):
  #   self.target_data = target_data
  #   self.sample_numbers = sample_numbers
  #   self.epochs= epochs
  def fit(self, target_data, rows_per_folds=None, sample_numbers=None, epochs=None):
    self.target_data = target_data
    self.rows_per_folds = rows_per_folds
    self.sample_numbers = sample_numbers
    self.epochs= epochs
      
    discrete_columns = list(target_data.columns)

    if rows_per_folds >= int(len(target_data.index)):
      resampled_total_rows = int(len(target_data.index))
      discrete_columns = list(target_data.columns)
      ctgan = CTGANSynthesizer()
      ctgan.fit(target_data, discrete_columns, epochs=epochs)
      samples = ctgan.sample(sample_numbers)
      samples 
      resampled = resample(target_data,
                    replace=True, # sample with replacement
                    n_samples=sample_numbers, # match number in majority class
                    random_state=27) # reproducible results
            #resampled to CTGAN 生成
      resampled_ctgan = resample(samples,
                          replace=True, # sample with replacement
                          n_samples=sample_numbers, # match number in majority class
                          random_state=27) # reproducible results
            #CTGAN to resampled 生成
      resampledforctgan = resampled[:resampled_total_rows]      
      ctgan_resampled = CTGANSynthesizer()
      ctgan_resampled.fit(resampledforctgan, discrete_columns, epochs=epochs)
      ctgan_resampled = ctgan_resampled.sample(sample_numbers)
            #CTGAN_resampled 合成
      resampledandctgan = [samples, resampled]
      resampledandctgan = pd.concat(resampledandctgan)
      resampledandctgan_resampled_ctgan = [resampledandctgan, resampled_ctgan]
      resampledandctgan_resampled_ctgan = pd.concat(resampledandctgan_resampled_ctgan)
      resampledandctgan_resampled_ctgan_ctgan_resampled = [resampledandctgan_resampled_ctgan, ctgan_resampled]
      resampledandctgan_resampled_ctgan_ctgan_resampled = pd.concat(resampledandctgan_resampled_ctgan_ctgan_resampled)

      ctgan_resampled_data = resampledandctgan_resampled_ctgan_ctgan_resampled.sample(sample_numbers*4)
      ctgan_resampled_data = ctgan_resampled_data.drop_duplicates(subset=None, keep='first', inplace=False)
      ctgan_resampled_data = ctgan_resampled_data[:sample_numbers]
      return ctgan_resampled_data

    else:

      # length = int(len(target_data)/split_folds) #length of each fold
      folds = []
      data_length = int(len(target_data.index))
      length_numbers = rows_per_folds
      CS_iterations = int(data_length/length_numbers)
      for i in range(CS_iterations):
        folds += [target_data[i*length_numbers:(i+1)*length_numbers]]
      folds += [target_data[(CS_iterations)*length_numbers:len(target_data)]]

      previous = []
      count = 0   #計數器
      while count < (CS_iterations):

        resampled_total_rows = int(len(folds[count].index))
        # split_sample_numbers = int(sample_numbers/2)
        # split_sample_numbers = int(sample_numbers)
        # split_sample_numbers = int((len(folds[count].index)*1.3))
        split_sample_numbers = int((sample_numbers/(CS_iterations+1))*1.5)
        discrete_columns = list(target_data.columns)
        ctgan = CTGANSynthesizer()
        ctgan.fit(folds[count], discrete_columns, epochs=epochs)
        samples = ctgan.sample(split_sample_numbers)
        samples 
        resampled = resample(folds[count],
                      replace=True, # sample with replacement
                      n_samples=split_sample_numbers, # match number in majority class
                      random_state=27) # reproducible results
              #resampled to CTGAN 生成
        resampled_ctgan = resample(samples,
                            replace=True, # sample with replacement
                            n_samples=split_sample_numbers, # match number in majority class
                            random_state=27) # reproducible results
              #CTGAN to resampled 生成
        resampledforctgan = resampled[:resampled_total_rows]      
        ctgan_resampled = CTGANSynthesizer()
        ctgan_resampled.fit(resampledforctgan, discrete_columns, epochs=epochs)
        ctgan_resampled = ctgan_resampled.sample(split_sample_numbers)
              #CTGAN_resampled 合成
        resampledandctgan = [samples, resampled]
        resampledandctgan = pd.concat(resampledandctgan)
        resampledandctgan_resampled_ctgan = [resampledandctgan, resampled_ctgan]
        resampledandctgan_resampled_ctgan = pd.concat(resampledandctgan_resampled_ctgan)
        resampledandctgan_resampled_ctgan_ctgan_resampled = [resampledandctgan_resampled_ctgan, ctgan_resampled]
        resampledandctgan_resampled_ctgan_ctgan_resampled = pd.concat(resampledandctgan_resampled_ctgan_ctgan_resampled)

        ctgan_resampled_data = resampledandctgan_resampled_ctgan_ctgan_resampled.sample(split_sample_numbers*4)
        ctgan_resampled_data = ctgan_resampled_data.drop_duplicates(subset=None, keep='first', inplace=False)
        ctgan_resampled_data = ctgan_resampled_data[:split_sample_numbers]
        data = ctgan_resampled_data
        previous.append(data)
        ctgan_resampled_data = pd.concat(previous)

        count = count+1

      ctgan_resampled_data = ctgan_resampled_data.drop_duplicates(subset=None, keep='first', inplace=False)
      CS_sample_numbers = int(len(ctgan_resampled_data))
      ctgan_resampled_data = ctgan_resampled_data.sample(CS_sample_numbers)
      ctgan_resampled_data = ctgan_resampled_data[:sample_numbers]
      
      return ctgan_resampled_data

class CRE_SynthesizerNrows():#把所有資料以n個rows分成各個fold並一次運算一個fold
  # def __init__(self, target_data, sample_numbers=None, epochs=None):
  #   self.target_data = target_data
  #   self.sample_numbers = sample_numbers
  #   self.epochs= epochs
  def fit(self, target_data, rows_per_folds=None, sample_numbers=None, epochs=None):
    self.target_data = target_data
    self.rows_per_folds = rows_per_folds
    self.sample_numbers = sample_numbers
    self.epochs= epochs
      
    discrete_columns = list(target_data.columns)

    if rows_per_folds >= int(len(target_data.index)):
      resampled_total_rows = int(len(target_data.index))
      discrete_columns = list(target_data.columns)
      ctgan = CTGANSynthesizer()
      ctgan.fit(target_data, discrete_columns, epochs=epochs)
      samples = ctgan.sample(sample_numbers)
      samples 
      resampled = resample(target_data,
                    replace=True, # sample with replacement
                    n_samples=sample_numbers, # match number in majority class
                    random_state=27) # reproducible results
            #resampled to CTGAN 生成
      resampled_ctgan = resample(samples,
                          replace=True, # sample with replacement
                          n_samples=sample_numbers, # match number in majority class
                          random_state=27) # reproducible results
            #CTGAN to resampled 生成
      resampledforctgan = resampled[:resampled_total_rows]      
      ctgan_resampled = CTGANSynthesizer()
      ctgan_resampled.fit(resampledforctgan, discrete_columns, epochs=epochs)
      ctgan_resampled = ctgan_resampled.sample(sample_numbers)
            #CTGAN_resampled 合成
      resampledandctgan = [samples, resampled]
      resampledandctgan = pd.concat(resampledandctgan)
      resampledandctgan_resampled_ctgan = [resampledandctgan, resampled_ctgan]
      resampledandctgan_resampled_ctgan = pd.concat(resampledandctgan_resampled_ctgan)
      resampledandctgan_resampled_ctgan_ctgan_resampled = [resampledandctgan_resampled_ctgan, ctgan_resampled]
      resampledandctgan_resampled_ctgan_ctgan_resampled = pd.concat(resampledandctgan_resampled_ctgan_ctgan_resampled)

      ctgan_resampled_data = resampledandctgan_resampled_ctgan_ctgan_resampled.sample(sample_numbers*4)
      ctgan_resampled_data = ctgan_resampled_data.drop_duplicates(subset=None, keep='first', inplace=False)
      ctgan_resampled_data = ctgan_resampled_data[:sample_numbers]
      return ctgan_resampled_data

    else:

      # length = int(len(target_data)/split_folds) #length of each fold
      folds = []
      data_length = int(len(target_data.index))
      length_numbers = rows_per_folds
      CS_iterations = int(data_length/length_numbers)
      for i in range(CS_iterations):
        folds += [target_data[i*length_numbers:(i+1)*length_numbers]]
      folds += [target_data[(CS_iterations)*length_numbers:len(target_data)]]

      previous = []
      ctgan_previous = []
      resampled_previous = []
      ctgan_resampled_previous = []
      resampled_ctgan_previous = []
      count = 0   #計數器
      while count < (CS_iterations):

        resampled_total_rows = int(len(folds[count].index))
        # split_sample_numbers = int(sample_numbers/2)
        # split_sample_numbers = int(sample_numbers)
        # split_sample_numbers = int((len(folds[count].index)*1.3))
        split_sample_numbers = int((sample_numbers/(CS_iterations+1))*1.5)
        discrete_columns = list(target_data.columns)
        ctgan = CTGANSynthesizer()
        ctgan.fit(folds[count], discrete_columns, epochs=epochs)
        samples = ctgan.sample(split_sample_numbers)
        samples 
        resampled = resample(folds[count],
                      replace=True, # sample with replacement
                      n_samples=split_sample_numbers, # match number in majority class
                      random_state=27) # reproducible results
              #resampled to CTGAN 生成
        resampled_ctgan = resample(samples,
                            replace=True, # sample with replacement
                            n_samples=split_sample_numbers, # match number in majority class
                            random_state=27) # reproducible results
              #CTGAN to resampled 生成
        resampledforctgan = resampled[:resampled_total_rows]      
        ctgan_resampled = CTGANSynthesizer()
        ctgan_resampled.fit(resampledforctgan, discrete_columns, epochs=epochs)
        ctgan_resampled = ctgan_resampled.sample(split_sample_numbers)
              #CTGAN_resampled 合成
        # resampledandctgan = [samples, resampled]
        # resampledandctgan = pd.concat(resampledandctgan)
        # resampledandctgan_resampled_ctgan = [resampledandctgan, resampled_ctgan]
        # resampledandctgan_resampled_ctgan = pd.concat(resampledandctgan_resampled_ctgan)
        # resampledandctgan_resampled_ctgan_ctgan_resampled = [resampledandctgan_resampled_ctgan, ctgan_resampled]
        # resampledandctgan_resampled_ctgan_ctgan_resampled = pd.concat(resampledandctgan_resampled_ctgan_ctgan_resampled)

        #CTGAN
        samples = samples.sample(int(len(samples)))
        samples = samples.drop_duplicates(subset=None, keep='first', inplace=False)
        data_ctgan = samples
        ctgan_previous.append(data_ctgan)
        samples = pd.concat(ctgan_previous)

        #resampled
        resampled = resampled.sample(int(len(resampled)))
        resampled = resampled.drop_duplicates(subset=None, keep='first', inplace=False)
        data_resampled = resampled
        resampled_previous.append(data_resampled)
        resampled = pd.concat(resampled_previous)

        #resampled-CTGAN
        resampled_ctgan = resampled_ctgan.sample(int(len(resampled_ctgan)))
        resampled_ctgan = resampled_ctgan.drop_duplicates(subset=None, keep='first', inplace=False)
        data_resampled_ctgan = resampled_ctgan
        resampled_ctgan_previous.append(data_resampled_ctgan)
        resampled_ctgan = pd.concat(resampled_ctgan_previous)

        #CTGAN-resampled
        ctgan_resampled = ctgan_resampled.sample(int(len(ctgan_resampled)))
        ctgan_resampled = ctgan_resampled.drop_duplicates(subset=None, keep='first', inplace=False)
        data_ctgan_resampled = ctgan_resampled
        ctgan_resampled_previous.append(data_ctgan_resampled)
        ctgan_resampled = pd.concat(ctgan_resampled_previous)

        # ctgan_resampled_data = resampledandctgan_resampled_ctgan_ctgan_resampled.sample(split_sample_numbers*4)
        # ctgan_resampled_data = ctgan_resampled_data.drop_duplicates(subset=None, keep='first', inplace=False)
        # ctgan_resampled_data = ctgan_resampled_data[:split_sample_numbers]
        # data = ctgan_resampled_data
        # previous.append(data)
        # ctgan_resampled_data = pd.concat(previous)

        count = count+1
                #CTGAN_resampled 合成
      resampledandctgan = [samples, resampled]
      resampledandctgan = pd.concat(resampledandctgan)
      resampledandctgan_resampled_ctgan = [resampledandctgan, resampled_ctgan]
      resampledandctgan_resampled_ctgan = pd.concat(resampledandctgan_resampled_ctgan)
      resampledandctgan_resampled_ctgan_ctgan_resampled = [resampledandctgan_resampled_ctgan, ctgan_resampled]
      resampledandctgan_resampled_ctgan_ctgan_resampled = pd.concat(resampledandctgan_resampled_ctgan_ctgan_resampled)

      ctgan_resampled_data = resampledandctgan_resampled_ctgan_ctgan_resampled.sample(int(len(resampledandctgan_resampled_ctgan_ctgan_resampled)))
      ctgan_resampled_data = ctgan_resampled_data.drop_duplicates(subset=None, keep='first', inplace=False)
      CS_sample_numbers = int(len(ctgan_resampled_data))
      ctgan_resampled_data = ctgan_resampled_data.sample(CS_sample_numbers)
      ctgan_resampled_data = ctgan_resampled_data[:sample_numbers]
      
      return ctgan_resampled_data

class CS_SynthesizerNrows():#把所有資料以n個rows分成各個fold並一次運算一個fold
  # def __init__(self, target_data, sample_numbers=None, epochs=None):
  #   self.target_data = target_data
  #   self.sample_numbers = sample_numbers
  #   self.epochs= epochs
  def fit(self, target_data, rows_per_folds=None, sample_numbers=None, epochs=None, k_neighbors = None):
    self.target_data = target_data
    self.rows_per_folds = rows_per_folds
    self.sample_numbers = sample_numbers
    self.epochs= epochs
    self.k_neighbors= k_neighbors        
    discrete_columns = list(target_data.columns)

    if rows_per_folds >= int(len(target_data.index)):
      smote_total_rows = int(len(target_data.index))
      discrete_columns = list(target_data.columns)
      ctgan = CTGANSynthesizer()
      ctgan.fit(target_data, discrete_columns, epochs=epochs)
      samples = ctgan.sample(sample_numbers)
      samples 
      SMOTE_Model = SMOTE()
      smote = SMOTE_Model.fit(target_data, sample_numbers=sample_numbers, k_neighbors=k_neighbors) 
            #smote to CTGAN 生成
      SMOTE_Model = SMOTE()
      smote_ctgan = SMOTE_Model.fit(samples, sample_numbers=sample_numbers, k_neighbors=k_neighbors)
            #CTGAN to smote 生成
      smoteforctgan = smote[:smote_total_rows]      
      ctgan_smote = CTGANSynthesizer()
      ctgan_smote.fit(smote_ctgan, discrete_columns, epochs=epochs)
      ctgan_smote = ctgan_smote.sample(sample_numbers)
            #CTGAN_smote 合成
      smoteandctgan = [samples, smote]
      smoteandctgan = pd.concat(smoteandctgan)
      smoteandctgan_smote_ctgan = [smoteandctgan, smote_ctgan]
      smoteandctgan_smote_ctgan = pd.concat(smoteandctgan_smote_ctgan)
      smoteandctgan_smote_ctgan_ctgan_smote = [smoteandctgan_smote_ctgan, ctgan_smote]
      smoteandctgan_smote_ctgan_ctgan_smote = pd.concat(smoteandctgan_smote_ctgan_ctgan_smote)

      ctgan_smote_data = smoteandctgan_smote_ctgan_ctgan_smote.sample(sample_numbers*4)
      ctgan_smote_data = ctgan_smote_data.drop_duplicates(subset=None, keep='first', inplace=False)
      ctgan_smote_data = ctgan_smote_data[:sample_numbers]
      return ctgan_smote_data

    else:

      # length = int(len(target_data)/split_folds) #length of each fold
      folds = []
      data_length = int(len(target_data.index))
      length_numbers = rows_per_folds
      CS_iterations = int(data_length/length_numbers)
      for i in range(CS_iterations):
        folds += [target_data[i*length_numbers:(i+1)*length_numbers]]
      folds += [target_data[(CS_iterations)*length_numbers:len(target_data)]]

      previous = []
      count = 0   #計數器
      while count < (CS_iterations):

        smote_total_rows = int(len(folds[count].index))
        # split_sample_numbers = int(sample_numbers/2)
        # split_sample_numbers = int(sample_numbers)
        # split_sample_numbers = int((len(folds[count].index)*1.3))
        split_sample_numbers = int((sample_numbers/(CS_iterations+1))*1.5)
        discrete_columns = list(target_data.columns)
        ctgan = CTGANSynthesizer()
        ctgan.fit(folds[count], discrete_columns, epochs=epochs)
        samples = ctgan.sample(split_sample_numbers)
        samples 
        SMOTE_Model = SMOTE()
        smote = SMOTE_Model.fit(target_data, sample_numbers=split_sample_numbers, k_neighbors=k_neighbors) 
              #smote to CTGAN 生成
        SMOTE_Model = SMOTE()
        smote_ctgan = SMOTE_Model.fit(samples, sample_numbers=split_sample_numbers, k_neighbors=k_neighbors)
              #CTGAN to smote 生成
        smoteforctgan = smote[:smote_total_rows]      
        ctgan_smote = CTGANSynthesizer()
        ctgan_smote.fit(smoteforctgan, discrete_columns, epochs=epochs)
        ctgan_smote = ctgan_smote.sample(split_sample_numbers)
              #CTGAN_smote 合成
        smoteandctgan = [samples, smote]
        smoteandctgan = pd.concat(smoteandctgan)
        smoteandctgan_smote_ctgan = [smoteandctgan, smote_ctgan]
        smoteandctgan_smote_ctgan = pd.concat(smoteandctgan_smote_ctgan)
        smoteandctgan_smote_ctgan_ctgan_smote = [smoteandctgan_smote_ctgan, ctgan_smote]
        smoteandctgan_smote_ctgan_ctgan_smote = pd.concat(smoteandctgan_smote_ctgan_ctgan_smote)

        ctgan_smote_data = smoteandctgan_smote_ctgan_ctgan_smote.sample(split_sample_numbers*4)
        ctgan_smote_data = ctgan_smote_data.drop_duplicates(subset=None, keep='first', inplace=False)
        ctgan_smote_data = ctgan_smote_data[:split_sample_numbers]
        data = ctgan_smote_data
        previous.append(data)
        ctgan_smote_data = pd.concat(previous)

        count = count+1

      ctgan_smote_data = ctgan_smote_data.drop_duplicates(subset=None, keep='first', inplace=False)
      CS_sample_numbers = int(len(ctgan_smote_data))
      ctgan_smote_data = ctgan_smote_data.sample(CS_sample_numbers)
      ctgan_smote_data = ctgan_smote_data[:sample_numbers]
      
      return ctgan_smote_data

class CSE_SynthesizerNrows():#把所有資料以n個rows分成各個fold並一次運算一個fold
  # def __init__(self, target_data, sample_numbers=None, epochs=None):
  #   self.target_data = target_data
  #   self.sample_numbers = sample_numbers
  #   self.epochs= epochs
  def fit(self, target_data, rows_per_folds=None, sample_numbers=None, epochs=None, k_neighbors = None):
    self.target_data = target_data
    self.rows_per_folds = rows_per_folds
    self.sample_numbers = sample_numbers
    self.epochs= epochs
    self.k_neighbors= k_neighbors 
      
    discrete_columns = list(target_data.columns)

    if rows_per_folds >= int(len(target_data.index)):
      smote_total_rows = int(len(target_data.index))
      discrete_columns = list(target_data.columns)
      ctgan = CTGANSynthesizer()
      ctgan.fit(target_data, discrete_columns, epochs=epochs)
      samples = ctgan.sample(sample_numbers)
      samples 
      SMOTE_Model = SMOTE()
      smote = SMOTE_Model.fit(target_data, sample_numbers=sample_numbers, k_neighbors=k_neighbors) 
            #smote to CTGAN 生成
      SMOTE_Model = SMOTE()
      smote_ctgan = SMOTE_Model.fit(samples, sample_numbers=sample_numbers, k_neighbors=k_neighbors)
            #CTGAN to smote 生成
      smoteforctgan = smote[:smote_total_rows]      
      ctgan_smote = CTGANSynthesizer()
      ctgan_smote.fit(smote_ctgan, discrete_columns, epochs=epochs)
      ctgan_smote = ctgan_smote.sample(sample_numbers)
            #CTGAN_smote 合成
      smoteandctgan = [samples, smote]
      smoteandctgan = pd.concat(smoteandctgan)
      smoteandctgan_smote_ctgan = [smoteandctgan, smote_ctgan]
      smoteandctgan_smote_ctgan = pd.concat(smoteandctgan_smote_ctgan)
      smoteandctgan_smote_ctgan_ctgan_smote = [smoteandctgan_smote_ctgan, ctgan_smote]
      smoteandctgan_smote_ctgan_ctgan_smote = pd.concat(smoteandctgan_smote_ctgan_ctgan_smote)

      ctgan_smote_data = smoteandctgan_smote_ctgan_ctgan_smote.sample(sample_numbers*4)
      ctgan_smote_data = ctgan_smote_data.drop_duplicates(subset=None, keep='first', inplace=False)
      ctgan_smote_data = ctgan_smote_data[:sample_numbers]
      return ctgan_smote_data

    else:

      # length = int(len(target_data)/split_folds) #length of each fold
      folds = []
      data_length = int(len(target_data.index))
      length_numbers = rows_per_folds
      CS_iterations = int(data_length/length_numbers)
      for i in range(CS_iterations):
        folds += [target_data[i*length_numbers:(i+1)*length_numbers]]
      folds += [target_data[(CS_iterations)*length_numbers:len(target_data)]]

      previous = []
      ctgan_previous = []
      smote_previous = []
      ctgan_smote_previous = []
      smote_ctgan_previous = []
      count = 0   #計數器
      while count < (CS_iterations):

        smote_total_rows = int(len(folds[count].index))
        # split_sample_numbers = int(sample_numbers/2)
        # split_sample_numbers = int(sample_numbers)
        # split_sample_numbers = int((len(folds[count].index)*1.3))
        split_sample_numbers = int((sample_numbers/(CS_iterations+1))*1.5)
        discrete_columns = list(target_data.columns)
        ctgan = CTGANSynthesizer()
        ctgan.fit(folds[count], discrete_columns, epochs=epochs)
        samples = ctgan.sample(split_sample_numbers)
        samples 
        SMOTE_Model = SMOTE()
        smote = SMOTE_Model.fit(target_data, sample_numbers=sample_numbers, k_neighbors=k_neighbors) 
              #smote to CTGAN 生成
        SMOTE_Model = SMOTE()
        smote_ctgan = SMOTE_Model.fit(samples, sample_numbers=sample_numbers, k_neighbors=k_neighbors)
              #CTGAN to smote 生成
        smoteforctgan = smote[:smote_total_rows]      
        ctgan_smote = CTGANSynthesizer()
        ctgan_smote.fit(smoteforctgan, discrete_columns, epochs=epochs)
        ctgan_smote = ctgan_smote.sample(split_sample_numbers)
              #CTGAN_smote 合成
        # smoteandctgan = [samples, smote]
        # smoteandctgan = pd.concat(smoteandctgan)
        # smoteandctgan_smote_ctgan = [smoteandctgan, smote_ctgan]
        # smoteandctgan_smote_ctgan = pd.concat(smoteandctgan_smote_ctgan)
        # smoteandctgan_smote_ctgan_ctgan_smote = [smoteandctgan_smote_ctgan, ctgan_smote]
        # smoteandctgan_smote_ctgan_ctgan_smote = pd.concat(smoteandctgan_smote_ctgan_ctgan_smote)

        #CTGAN
        samples = samples.sample(int(len(samples)))
        samples = samples.drop_duplicates(subset=None, keep='first', inplace=False)
        data_ctgan = samples
        ctgan_previous.append(data_ctgan)
        samples = pd.concat(ctgan_previous)

        #smote
        smote = smote.sample(int(len(smote)))
        smote = smote.drop_duplicates(subset=None, keep='first', inplace=False)
        data_smote = smote
        smote_previous.append(data_smote)
        smote = pd.concat(smote_previous)

        #smote-CTGAN
        smote_ctgan = smote_ctgan.sample(int(len(smote_ctgan)))
        smote_ctgan = smote_ctgan.drop_duplicates(subset=None, keep='first', inplace=False)
        data_smote_ctgan = smote_ctgan
        smote_ctgan_previous.append(data_smote_ctgan)
        smote_ctgan = pd.concat(smote_ctgan_previous)

        #CTGAN-smote
        ctgan_smote = ctgan_smote.sample(int(len(ctgan_smote)))
        ctgan_smote = ctgan_smote.drop_duplicates(subset=None, keep='first', inplace=False)
        data_ctgan_smote = ctgan_smote
        ctgan_smote_previous.append(data_ctgan_smote)
        ctgan_smote = pd.concat(ctgan_smote_previous)

        # ctgan_smote_data = smoteandctgan_smote_ctgan_ctgan_smote.sample(split_sample_numbers*4)
        # ctgan_smote_data = ctgan_smote_data.drop_duplicates(subset=None, keep='first', inplace=False)
        # ctgan_smote_data = ctgan_smote_data[:split_sample_numbers]
        # data = ctgan_smote_data
        # previous.append(data)
        # ctgan_smote_data = pd.concat(previous)

        count = count+1
                #CTGAN_smote 合成
      smoteandctgan = [samples, smote]
      smoteandctgan = pd.concat(smoteandctgan)
      smoteandctgan_smote_ctgan = [smoteandctgan, smote_ctgan]
      smoteandctgan_smote_ctgan = pd.concat(smoteandctgan_smote_ctgan)
      smoteandctgan_smote_ctgan_ctgan_smote = [smoteandctgan_smote_ctgan, ctgan_smote]
      smoteandctgan_smote_ctgan_ctgan_smote = pd.concat(smoteandctgan_smote_ctgan_ctgan_smote)

      ctgan_smote_data = smoteandctgan_smote_ctgan_ctgan_smote.sample(int(len(smoteandctgan_smote_ctgan_ctgan_smote)))
      ctgan_smote_data = ctgan_smote_data.drop_duplicates(subset=None, keep='first', inplace=False)
      CS_sample_numbers = int(len(ctgan_smote_data))
      ctgan_smote_data = ctgan_smote_data.sample(CS_sample_numbers)
      ctgan_smote_data = ctgan_smote_data[:sample_numbers]
      
      return ctgan_smote_data


class CSR_Synthesizer():
  # def __init__(self, target_data, sample_numbers=None, epochs=None):
  #   self.target_data = target_data
  #   self.sample_numbers = sample_numbers
  #   self.epochs= epochs
  def fit(self, target_data, sample_numbers=None, epochs=None, k_neighbors = None):
    self.target_data = target_data
    self.sample_numbers = sample_numbers
    self.epochs= epochs
    self.k_neighbors= k_neighbors              
    original_total_rows = int(len(target_data.index))
    discrete_columns = list(target_data.columns)
    split_sample_numbers_Origin = int(sample_numbers*2/3)
    split_sample_numbers_After = int(sample_numbers/6)

          #SMOTE 生成
    smote = SMOTE()
    sdata = smote.fit(target_data, sample_numbers=split_sample_numbers_Origin, k_neighbors=k_neighbors)    
          #CTGAN 生成
    ctgan = CTGANSynthesizer()
    ctgan.fit(target_data, discrete_columns, epochs=epochs)
    samples = ctgan.sample(split_sample_numbers_Origin)
    samples
          #resampled 生成     
    resampled = resample(target_data,
                  replace=True, # sample with replacement
                  n_samples=split_sample_numbers_Origin, # match number in majority class
                  random_state=27) # reproducible results


          #resampled to CTGAN 生成
    resampled_ctgan = resample(samples,
                        replace=True, # sample with replacement
                        n_samples=split_sample_numbers_After, # match number in majority class
                        random_state=27) # reproducible results
          #resampled to SMOTE 生成
    resampled_smote = resample(sdata,
                        replace=True, # sample with replacement
                        n_samples=split_sample_numbers_After, # match number in majority class
                        random_state=27) # reproducible results

          #CTGAN to resampled 生成
    resampledforctgan = resampled[:original_total_rows]      
    ctgan_resampled = CTGANSynthesizer()
    ctgan_resampled.fit(resampledforctgan, discrete_columns, epochs=epochs)
    ctgan_resampled = ctgan_resampled.sample(split_sample_numbers_After)

          #CTGAN to SMOTE 生成
    smoteforctgan = sdata[:original_total_rows]      
    ctgan_smote = CTGANSynthesizer()
    ctgan_smote.fit(smoteforctgan, discrete_columns, epochs=epochs)
    ctgan_smote = ctgan_smote.sample(split_sample_numbers_After)

          #SMOTE to CTGAN 生成
    smote = SMOTE()
    smote_ctgan = smote.fit(samples, sample_numbers=split_sample_numbers_After, k_neighbors=k_neighbors)
          #SMOTE to resampled 生成
    smote = SMOTE()
    smote_resampled = smote.fit(resampled, sample_numbers=split_sample_numbers_After, k_neighbors=k_neighbors)

          #CTGAN_resampled 合成
    ctgan_smote_resampled = [samples, resampled, sdata, resampled_ctgan, resampled_smote, ctgan_smote, smote_resampled, smote_ctgan, smote_resampled]
    ctgan_smote_resampled = pd.concat(ctgan_smote_resampled)
    
    ctgan_smote_resampled_data = ctgan_smote_resampled.sample(int(len(ctgan_smote_resampled.index)))
    ctgan_smote_resampled_data = ctgan_smote_resampled_data.drop_duplicates(subset=None, keep='first', inplace=False)
    ctgan_smote_resampled_data = ctgan_smote_resampled_data[:sample_numbers]
    return ctgan_smote_resampled_data


class CSRE_SynthesizerNrows():#把所有資料以n個rows分成各個fold並一次運算一個fold
  # def __init__(self, target_data, sample_numbers=None, epochs=None):
  #   self.target_data = target_data
  #   self.sample_numbers = sample_numbers
  #   self.epochs= epochs
  def fit(self, target_data, rows_per_folds=None, sample_numbers=None, epochs=None, k_neighbors = None):
    self.target_data = target_data
    self.rows_per_folds = rows_per_folds
    self.sample_numbers = sample_numbers
    self.epochs= epochs
    self.k_neighbors= k_neighbors 
      
    discrete_columns = list(target_data.columns)

    if rows_per_folds >= int(len(target_data.index)):
      original_total_rows = int(len(target_data.index))
      discrete_columns = list(target_data.columns)
      split_sample_numbers_Origin = int(sample_numbers*2/3)
      split_sample_numbers_After = int(sample_numbers/6)

            #SMOTE 生成
      smote = SMOTE()
      sdata = smote.fit(target_data, sample_numbers=split_sample_numbers_Origin, k_neighbors=k_neighbors)    
            #CTGAN 生成
      ctgan = CTGANSynthesizer()
      ctgan.fit(target_data, discrete_columns, epochs=epochs)
      samples = ctgan.sample(split_sample_numbers_Origin)
      samples
            #resampled 生成     
      resampled = resample(target_data,
                    replace=True, # sample with replacement
                    n_samples=split_sample_numbers_Origin, # match number in majority class
                    random_state=27) # reproducible results


            #resampled to CTGAN 生成
      resampled_ctgan = resample(samples,
                          replace=True, # sample with replacement
                          n_samples=split_sample_numbers_After, # match number in majority class
                          random_state=27) # reproducible results
            #resampled to SMOTE 生成
      resampled_smote = resample(sdata,
                          replace=True, # sample with replacement
                          n_samples=split_sample_numbers_After, # match number in majority class
                          random_state=27) # reproducible results

            #CTGAN to resampled 生成
      resampledforctgan = resampled[:original_total_rows]      
      ctgan_resampled = CTGANSynthesizer()
      ctgan_resampled.fit(resampledforctgan, discrete_columns, epochs=epochs)
      ctgan_resampled = ctgan_resampled.sample(split_sample_numbers_After)

            #CTGAN to SMOTE 生成
      smoteforctgan = sdata[:original_total_rows]      
      ctgan_smote = CTGANSynthesizer()
      ctgan_smote.fit(smoteforctgan, discrete_columns, epochs=epochs)
      ctgan_smote = ctgan_smote.sample(split_sample_numbers_After)

            #SMOTE to CTGAN 生成
      smote = SMOTE()
      smote_ctgan = smote.fit(samples, sample_numbers=split_sample_numbers_After, k_neighbors=k_neighbors)
            #SMOTE to resampled 生成
      smote = SMOTE()
      smote_resampled = smote.fit(resampled, sample_numbers=split_sample_numbers_After, k_neighbors=k_neighbors)

            #CTGAN_resampled 合成
      ctgan_smote_resampled = [samples, resampled, sdata, resampled_ctgan, resampled_smote, ctgan_smote, smote_resampled, smote_ctgan, smote_resampled]
      ctgan_smote_resampled = pd.concat(ctgan_smote_resampled)
      
      ctgan_smote_resampled_data = ctgan_smote_resampled.sample(int(len(ctgan_smote_resampled.index)))
      ctgan_smote_resampled_data = ctgan_smote_resampled_data.drop_duplicates(subset=None, keep='first', inplace=False)
      ctgan_smote_resampled_data = ctgan_smote_resampled_data[:sample_numbers]
      return ctgan_smote_resampled_data

    else:
      # length = int(len(target_data)/split_folds) #length of each fold
      folds = []
      data_length = int(len(target_data.index))
      length_numbers = rows_per_folds
      CS_iterations = int(data_length/length_numbers)
      for i in range(CS_iterations):
        folds += [target_data[i*length_numbers:(i+1)*length_numbers]]
      folds += [target_data[(CS_iterations)*length_numbers:len(target_data)]]

      previous = []
      ctgan_previous = []
      smote_previous = []
      resampled_previous = []
      ctgan_smote_previous = []
      ctgan_resampled_previous = []
      smote_ctgan_previous = []
      smote_resampled_previous = []
      resampled_ctgan_previous = []
      resampled_smote_previous = []
      count = 0   #計數器
      while count < (CS_iterations):

        original_total_rows = int(len(folds[count].index))
        # split_sample_numbers = int(sample_numbers/2)
        # split_sample_numbers = int(sample_numbers)
        # split_sample_numbers = int((len(folds[count].index)*1.3))
        split_sample_numbers_Origin = int((sample_numbers/(CS_iterations+1))*1.5)
        split_sample_numbers_After = int(sample_numbers/(CS_iterations+1)/2.667)
        discrete_columns = list(target_data.columns)


        #SMOTE 生成
        smote = SMOTE()
        sdata = smote.fit(folds[count], sample_numbers=split_sample_numbers_Origin, k_neighbors=k_neighbors) 
        #CTGAN 生成
        ctgan = CTGANSynthesizer()
        ctgan.fit(folds[count], discrete_columns, epochs=epochs)
        samples = ctgan.sample(split_sample_numbers_Origin)
        samples
        #resampled 生成 
        resampled = resample(folds[count],
                      replace=True, # sample with replacement
                      n_samples=split_sample_numbers_Origin, # match number in majority class
                      random_state=27) # reproducible results
        #resampled to CTGAN 生成
        resampled_ctgan = resample(samples,
                            replace=True, # sample with replacement
                            n_samples=split_sample_numbers_After, # match number in majority class
                            random_state=27) # reproducible results
        #resampled to SMOTE 生成
        resampled_smote = resample(sdata,
                            replace=True, # sample with replacement
                            n_samples=split_sample_numbers_After, # match number in majority class
                            random_state=27) # reproducible results
        #CTGAN to resampled 生成
        resampledforctgan = resampled[:original_total_rows]      
        ctgan_resampled = CTGANSynthesizer()
        ctgan_resampled.fit(resampledforctgan, discrete_columns, epochs=epochs)
        ctgan_resampled = ctgan_resampled.sample(split_sample_numbers_After)
        #CTGAN to SMOTE 生成
        smoteforctgan = sdata[:original_total_rows]      
        ctgan_smote = CTGANSynthesizer()
        ctgan_smote.fit(smoteforctgan, discrete_columns, epochs=epochs)
        ctgan_smote = ctgan_smote.sample(split_sample_numbers_After)
        #SMOTE to CTGAN 生成
        smote = SMOTE()
        smote_ctgan = smote.fit(samples, sample_numbers=split_sample_numbers_After, k_neighbors=k_neighbors)
        #SMOTE to resampled 生成
        smote = SMOTE()
        smote_resampled = smote.fit(resampled, sample_numbers=split_sample_numbers_After, k_neighbors=k_neighbors)

        #CTGAN_SMOTE 合成
        # smoteandctgan = [samples, smote]
        # smoteandctgan = pd.concat(smoteandctgan)
        # smoteandctgan_smote_ctgan = [smoteandctgan, smote_ctgan]
        # smoteandctgan_smote_ctgan = pd.concat(smoteandctgan_smote_ctgan)
        # smoteandctgan_smote_ctgan_ctgan_smote = [smoteandctgan_smote_ctgan, ctgan_smote]
        # smoteandctgan_smote_ctgan_ctgan_smote = pd.concat(smoteandctgan_smote_ctgan_ctgan_smote)

        #CTGAN
        samples = samples.sample(int(len(samples)))
        samples = samples.drop_duplicates(subset=None, keep='first', inplace=False)
        data_ctgan = samples
        ctgan_previous.append(data_ctgan)
        samples = pd.concat(ctgan_previous)
        #CTGAN-resampled
        ctgan_resampled = ctgan_resampled.sample(int(len(ctgan_resampled)))
        ctgan_resampled = ctgan_resampled.drop_duplicates(subset=None, keep='first', inplace=False)
        data_ctgan_resampled = ctgan_resampled
        ctgan_resampled_previous.append(data_ctgan_resampled)
        ctgan_resampled = pd.concat(ctgan_resampled_previous)
        #CTGAN-SMOTE
        ctgan_smote = ctgan_smote.sample(int(len(ctgan_smote)))
        ctgan_smote = ctgan_smote.drop_duplicates(subset=None, keep='first', inplace=False)
        data_ctgan_smote = ctgan_smote
        ctgan_smote_previous.append(data_ctgan_smote)
        ctgan_smote = pd.concat(ctgan_smote_previous)
        #SMOTE
        sdata = sdata.sample(int(len(sdata)))
        sdata = sdata.drop_duplicates(subset=None, keep='first', inplace=False)
        data_smote = sdata
        smote_previous.append(data_smote)
        sdata = pd.concat(smote_previous)
        #SMOTE-CTGAN
        smote_ctgan = smote_ctgan.sample(int(len(smote_ctgan)))
        smote_ctgan = smote_ctgan.drop_duplicates(subset=None, keep='first', inplace=False)
        data_smote_ctgan = smote_ctgan
        smote_ctgan_previous.append(data_smote_ctgan)
        smote_ctgan = pd.concat(smote_ctgan_previous)
        #SMOTE-resampled
        smote_resampled = smote_resampled.sample(int(len(smote_resampled)))
        smote_resampled = smote_resampled.drop_duplicates(subset=None, keep='first', inplace=False)
        data_smote_resampled = smote_resampled
        smote_resampled_previous.append(data_smote_resampled)
        smote_resampled = pd.concat(smote_resampled_previous)
        #resampled
        resampled = resampled.sample(int(len(resampled)))
        resampled = resampled.drop_duplicates(subset=None, keep='first', inplace=False)
        data_resampled = resampled
        resampled_previous.append(data_resampled)
        resampled = pd.concat(resampled_previous)
        #resampled-CTGAN
        resampled_ctgan = resampled_ctgan.sample(int(len(resampled_ctgan)))
        resampled_ctgan = resampled_ctgan.drop_duplicates(subset=None, keep='first', inplace=False)
        data_resampled_ctgan = resampled_ctgan
        resampled_ctgan_previous.append(data_resampled_ctgan)
        resampled_ctgan = pd.concat(resampled_ctgan_previous)
        #resampled-SMOTE
        resampled_smote = resampled_smote.sample(int(len(resampled_smote)))
        resampled_smote = resampled_smote.drop_duplicates(subset=None, keep='first', inplace=False)
        data_resampled_smote = resampled_smote
        resampled_smote_previous.append(data_resampled_smote)
        resampled_smote = pd.concat(resampled_smote_previous)

        # ctgan_smote_data = smoteandctgan_smote_ctgan_ctgan_smote.sample(split_sample_numbers*4)
        # ctgan_smote_data = ctgan_smote_data.drop_duplicates(subset=None, keep='first', inplace=False)
        # ctgan_smote_data = ctgan_smote_data[:split_sample_numbers]
        # data = ctgan_smote_data
        # previous.append(data)
        # ctgan_smote_data = pd.concat(previous)

        count = count+1
                #CTGAN_SMOTE 合成
    ctgan_smote_resampled = [samples, resampled, sdata, resampled_ctgan, resampled_smote, ctgan_smote, smote_resampled, smote_ctgan, smote_resampled]
    ctgan_smote_resampled = pd.concat(ctgan_smote_resampled)
    
    ctgan_smote_resampled_data = ctgan_smote_resampled.sample(int(len(ctgan_smote_resampled.index)))
    ctgan_smote_resampled_data = ctgan_smote_resampled_data.drop_duplicates(subset=None, keep='first', inplace=False)
    ctgan_smote_resampled_data = ctgan_smote_resampled_data[:sample_numbers]
    return ctgan_smote_resampled_data