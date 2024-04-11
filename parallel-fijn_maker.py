#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 11:07:51 2019

@author: iG
"""
import Wyckoff_finder as wf
import pandas as pd
import numpy as np
import itertools as it
import time
import copy
import multiprocessing as mp
import os


def inout_creator(df = pd.DataFrame(), features='datosrahm.csv',
                  site_normalization=False):
    
    """
    Parameters: 
        df:  A pandas DataFrame which contains information about the spacegroups and
                    the occupied symmetry sites. This must be specified with extension.
        features: A csv - file which contains the features to be use for each present 
                element in the sites of the structure.
        site_normalization: if the vacancies in a Wyckoff site are not considered. 
                            If False, vacancies are considered
    Returns:
        X: A matrix of samples x sites x features.
        fracsum: A matrix of samples x sites x occupation
        df: A pandas DataFrame with True/False values.
    """
    
    df = df
   
    start=time.time()
    
    datos=pd.read_csv(features)
    datos=datos.fillna(-1)

    dicc=dict(datos[['Symbol','Z']].values)

    dicc['D']=1
    dicc['Bk']=97
    dicc['Cf']=98
    dicc['Es']=99
    dicc['Fm']=100
    dicc['Md']=101
    dicc['No']=102
    dicc['Lr']=103
    
    max_sitios = max(df['sites'].values)
    
    X=np.zeros((len(df),max_sitios,104))

    mult=np.zeros((len(df),max_sitios))
    wyckmul=np.load('WyckoffSG_dict.npy', allow_pickle=True).item()['wyckmul']
    
    todelete = list()
    
    for row in range(len(df)):
        item=df['WyckOcc'][row]
        sitios=list(item.values()) #Diccionario de elementos con fracciones de ocupación en ese sitio     
        sitocc=np.zeros((len(sitios),104)) #Vector para 104 elementos de la tabla periódica
        spacegroup = str(df['sgnum'][row]).zfill(3)
        
        try:
        
            s=[int(wyckmul[spacegroup][i]) for j in [list(item.keys()) for item in \
               sitios] for i in j]
        
        except:
            print(row)
            print('There exists an error concerning with the space group of CIF ', df['name'][row],'\n')
            print('Please check in www.crystallography.net to provide the correct space group number of that CIF',
                  '\n','\n')
            spacegroup=input('Give me the correct spacegroup:'+'\n'+'\n')
            s=[int(wyckmul[spacegroup][i]) for j in [list(item.keys()) for item in \
               list(df['WyckOcc'][row].values())] for i in j]
        
        occs=[]
        for i in range(len(sitios)):

            for j in list(sitios[i].values()):
                
                ocupacion=np.array(list(j.values()))
                llaves=[llave.replace('+','').replace('-','').replace('1',
                        '').replace('2','').replace('3','').replace('4',
                                   '') for llave in np.array(list(j.keys()))]
                llaves=[llave.replace('.','') for llave in llaves]
                llaves=[llave.replace('5','').replace('6','').replace('7',
                        '').replace('8','').replace('9','').replace('0',
                                   '') for llave in llaves]
                vector=np.zeros((1,104))
                occs=[sum(ocupacion)]+occs
                
                try:
                    
                    idx=[dicc[k] for k in llaves]
                    for k in idx:
                        vector[0][k-1] = ocupacion[idx.index(k)]
                
                except:
                    print('The compound with the cif ', df['name'][row], ' will be deleted')
                    print('The database will be updated')
                    todelete += [row]
                    
            sitocc[i]=vector
            
        while sitocc.shape[0] != max_sitios:
            sitocc=np.concatenate((np.zeros((1,104)),sitocc))
            s=[0]+s
        
        X[row,:,:]=sitocc
        mult[row]=s
    
    features=datos.iloc[:,2:].values
    x=X[:,:,:96]
    
    fracsum = np.expand_dims(np.sum(x,axis=2), axis=2)
    
    if site_normalization == True:
        x = np.nan_to_num(x/fracsum)
        
    x=np.dot(x,features)    
    
  
    x = np.delete(x, todelete,axis=0)
    fracsum = np.delete(fracsum, todelete,axis=0)
    df = df.drop(df.index[todelete]).reset_index(drop=True)
    

    print('inout_creator lasted ',round(time.time()-start,2),' s')    
    return x, fracsum, df

def positions(archivo=1010902, dist = 100):
    
    pos, _, angles, abc = wf.wyckoff_positions(archivo = archivo)
    
    mult = [np.asarray(list(pos[i].values())).shape[1] for i in range(len(pos))]
    
    pos = np.concatenate([np.asarray(list(pos[item].values())) \
                          for item in range(len(pos))],axis=1)
        
    mot = pos.reshape((pos.shape[1],pos.shape[2]))
    
    volumen=abc[0]*abc[1]*abc[2]*np.sqrt(1-(np.cos(np.deg2rad(angles[0])))**2 - \
               (np.cos(np.deg2rad(angles[1])))**2 -\
               (np.cos(np.deg2rad(angles[2])))**2 + \
               2*np.cos(np.deg2rad(angles[0]))*np.cos(np.deg2rad(angles[1]))*np.cos(np.deg2rad(angles[2])))

    #La variable matrix convierte las coordenadas relativas a coordenadas absolutas en un sistema cartesiano
    matrix=np.array([[abc[0],abc[1]*np.cos(np.deg2rad(angles[2])),abc[2]*np.cos(np.deg2rad(angles[1]))],
                      [0,abc[1]*np.sin(np.deg2rad(angles[2])),abc[2]*(np.cos(np.deg2rad(angles[0]))-np.cos(np.deg2rad(angles[1]))*np.cos(np.deg2rad(angles[2])))/np.sin(np.deg2rad(angles[2]))],
                      [0,0,volumen/(abc[0]*abc[1]*np.sin(np.deg2rad(angles[2])))]])

    mt = np.round(matrix,5)

    n = int(np.ceil((dist+10)/np.min(mt[mt > 0])))

    if n > 30:
        print('Number of unit cell for each half - dimension is ',n,'\n')
        n = 30

    tras = list(it.product(np.arange(-n,n+1),repeat=3))

    zero = tras.index((0,0,0))
    tras = np.asarray(tras)

    h,w = mot.shape
    d = tras.shape[0]

    tras = tras.T
    
    mot = mot[:,:,np.newaxis]
    tras = tras[np.newaxis, :, :]
    
    
    mot = np.repeat(mot, d, axis=2)
    tras = np.repeat(tras, h, axis=0)

    mot = tras + mot
    mot = np.swapaxes(mot,1,2)
    mot = mot.astype(float)
    mot = np.matmul(mot,matrix)

    return mot, zero, n, mult

def exponential(x,n = 1, coef = 1):
    return np.exp(-np.power(x/coef,n))

def potential(x, n = 1, coef = 1):
    return np.power(coef*x,-n)

def angcos(x, dist = 5):
    return np.multiply(0.5*(np.cos(np.pi*x/dist) + 1), x <= dist)

def angtanh(x, dist = 5):
    return np.multiply(np.power(np.tahn(1-x/dist),3), x <= dist)

def rij(mult=[1,1,3], p = np.zeros((1,1,1)), zero = 1, dist=100, radii = [1,1,1], exponent = 1):
    
    radii = [item for item in radii if item != 0]
    l = [sum(mult[:i]) for i in range(len(mult)+1)]
    rij = list()

    for i,atrad_i in zip(range(1,len(l)),radii):
        r = p - p[l[i-1],zero,:]
        r = np.linalg.norm(r,axis=2)
    
        for j,atrad_j in zip(range(1,len(l)), radii):
            coef = (atrad_i + atrad_j)
            init = l[j-1]
            fin = l[j]
            rj = r[init:fin,:]
            rj = rj[rj <= dist]
            rj = rj[rj != 0]
            rj = np.sum(exponential(x = rj, n=exponent, coef=coef)*angcos(x=rj,dist=dist))
            rij += [rj]
    
    lon = int((len(rij))**(1/2))
    rij = np.asarray(rij).reshape((lon,lon))
        
    return rij
'''
df_name = input('Provide the name of the pickle database with extension:'+'\n')
cutoff_radius = input('Provide an integer cutoff radius in angstroms [default is 25]:'+'\n')
power = input('Provide the exponent value of the local function[default is 1]:'+'\n')
diffelec = input('Do you want to incorporate the electronegativity difference to the local function [default False, else type True]:'+'\n')

if not cutoff_radius:
    Rc = 25
else:
    Rc = int(cutoff_radius)

if not power:
    exponente = 1
else:
    exponente = float(power)
'''
import sys

df_name, cutoff_radius, power, diffelec = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

if not cutoff_radius:
    Rc = 25
else:
    Rc = int(cutoff_radius)

if not power:
    exponente = 1
else:
    exponente = float(power)

start = time.time()
df = pd.read_pickle(df_name)
X, S, df = inout_creator(df=df)

number = df['sites'].max()

new_directory = 'Rc_' + str(Rc) + '_pot_' + str(np.round(exponente,2))
if diffelec != 'False': new_directory = new_directory + '_diffelec'

try:
    os.mkdir(new_directory)
except:
    pass

def create_rij(row = 0):#, directorio ='./', diffelec = False):
    try:
        p,z,n,m = positions(archivo = df['name'][row], dist=100)
        radii = np.nan_to_num(X[row,:,1]/(S[row,:,0]+1e-16))

        r = rij(mult=m, p=p, zero=z, dist=Rc, radii = radii, exponent = exponente) 
        numSites = len(m)

        if diffelec != 'False':
            elecs = X[row, -numSites:,0]
            deltaElecs = elecs[None,:,None]-elecs[None,None,:]

            r = np.multiply(deltaElecs[0], r)
            #r = r[~np.eye(numSites, dtype=bool)].reshape(numSites,-1)

        newFile_route = new_directory + '/' + str(df['name'][row])
        np.save(newFile_route, r)
    except:
        pass
    return
'''
for row in range(df.shape[0]):
    print(row)
    create_rij(row = row)
'''
#args = [(row, new_directory, bool(diffelec)) for row in range(df.shape[0])]

if __name__ == "__main__":

    mp.Pool(14).map(create_rij, range(df.shape[0]))
