#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 13:11:29 2020

@author: root
"""

import numpy as np

# X is the concate of resting data, with shape (N*250 x channel#)
def zca_whitening_matrix(X):
    #calculate noise covariance matrix, sigma = (X-mu).T * (X-mu) / (N * 250)
    sigma = np.cov(X, rowvar=False)
    
    #eigenvalue decomposition of the covariance matrix (X = U * np.diag(S) * V)
    U,S,V = np.linalg.svd(sigma)
        #U : [MxM] eigenvectors
        #S : [Mx1] eigenvalues
        #V : [MxM] transpose of U
    
    epsilon = 1e-6
    #ZCA whitening matrix : U x Lanbda x U.T
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S+epsilon)), U.T)) # [MxM]
    
    return ZCAMatrix 
    