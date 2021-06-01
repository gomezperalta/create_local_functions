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
    
    max_sitios = max(df['sitios'].values)
    
    X=np.zeros((len(df),max_sitios,104))

    mult=np.zeros((len(df),max_sitios))
    wyckmul=np.load('WyckoffSG_dict.npy').item()['wyckmul']
    
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
            print('There exists an error concerning with the space group of CIF ', df['cif'][row],'\n')
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
                    print('The compound with the cif ', df['cif'][row], ' will be deleted')
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

def rij(mult=[1,1,3], p = np.zeros((1,1,1)), zero = 1, dist=100, 
        sites = 4, radii = [1,1,1], exponent = 1):
    
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

    s = sites

    if lon != s:
        rij = np.concatenate((np.zeros((rij.shape[0],s-lon)),rij),axis=1)
        rij = np.concatenate((np.zeros((s-lon,s)),rij), axis=0)    
        
    return rij

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

start = time.time()

df = pd.read_pickle(df_name)

X, S, df = inout_creator(df=df)

number = (np.max(df['sitios']))

print('Radial part of local function is computed now. This may take hours')

for row in range(len(df)):
    
    try:
        p,z,n,m = positions(archivo = df['cif'][row], dist=50)

        radii = np.nan_to_num(X[row,:,1]/S[row,:,0])

        if row == 0:
            r = rij(mult=m,p=p,zero=z,dist=Rc, radii = radii, sites = int(number), exponent = exponente)    
            r = np.expand_dims(r,axis=0)
        
        else:
            try:
                r_temp = rij(mult=m,p=p,zero=z,dist=Rc, radii = radii, sites = int(number), exponent = exponente)    
                r_temp = np.expand_dims(r_temp,axis=0)
            except:
                with open('problematic_samples_rij1.txt','a') as f:
                    f.write('Check the cif ' + str(df['cif'][row]) + ' in database ')
                    f.write('\n')
                    f.close()
                r_temp=np.zeros((1,r.shape[1],r.shape[2]))
                print('There was a problem with the cif ' + str(df['cif'][row]))
            r = np.concatenate((r,r_temp))
    except:
        with open('problematic_samples_rij1.txt','a') as f:
            f.write('The cif ' + str(df['cif'][row]) + ' in database cannot be treated with Wyckoff_finder.py')
            f.write('\n')
            f.close()
        print('There was a problem with the cif ' + str(df['cif'][row]))
        r_temp=np.zeros((1,r.shape[1],r.shape[2]))
        r = np.concatenate((r,r_temp))
            
    if row%1000 == 0:
        print(row)
        np.save('Xrij' + str(exponente) + '_' + cutoff_radius , r) 

np.save('Xrij' + str(exponente) + '_' + cutoff_radius , r)      

print('Radial part computed in ', time.time()-start, ' s')
print('Radial part saved as ' + str(exponente) + '_' + cutoff_radius)

if diffelec:
	print('Local function will be computed now')
	start = time.time()

	rij = copy.deepcopy(r)
	z = X[:,:,0]
	zeff = np.zeros(rij.shape)
	        
	for item in range(z.shape[0]):
	    t = z[item][:,np.newaxis]
	    delec = np.repeat(t[:,np.newaxis],int(number),axis=2) - \
	                        np.repeat(t[:,np.newaxis],int(number),axis=2).T
	    delec = delec.reshape((delec.shape[0],delec.shape[2]))
	        
	    zeff[item] = delec
	f = np.multiply(zeff,rij)
	        
	np.save('frdelec_rij'+ str(exponente) + '_' + str(cutoff_radius) + '-or.npy', f)
	print('Local function computed in ', time.time()-start, ' s')
	print('Local function was saved as '+ 'frdelec_rij'+ str(exponente) + '_' + \
	      str(cutoff_radius) + '-or.npy')

	print('Deleting the diagonal elements of matrices...')
	fn = np.zeros((f.shape[0], f.shape[1], f.shape[2] - 1))
	        
	for item in range(f.shape[0]):
	    delec = f[item]
	    delec = delec[~np.eye(delec.shape[0], dtype=bool)].reshape(delec.shape[0],-1)
	            
	    fn[item] = delec
	            
	np.save('frdelec_rij' + str(exponente) + '_' + str(cutoff_radius) + '.npy', fn)
	print('Deletion completed. Creating dictionary....')
else:
	f = copy.deepcopy(r)

diccio = dict()
for row in range(df.shape[0]):
    cif = df['cif'][row]
    matriz = f[row]
    sitios = df['sitios'][row]
    matriz = matriz[-sitios:,-sitios:]
    diccio[cif] = matriz
    
np.save('fij_' + str(exponente) + '_' + cutoff_radius + '_diccio', diccio)
print('Dictionary  created. EXITING PROGRAM..............................')
quit()
