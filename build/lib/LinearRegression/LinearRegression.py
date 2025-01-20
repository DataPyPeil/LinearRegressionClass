# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 10:44:13 2024

@author: peill
          
"""

import numpy as np
import matplotlib.pyplot as plt

class LinearRegression():
    
    def __init__(self, X, y_gt, lr=0.003, eps=0.001, reg=None, std=False):

        self.X = X
        self.y_gt = y_gt
        self.lr = lr
        self.eps = eps
        self.n, self.p = X.shape[0], X.shape[1]+1
        self.params = np.random.rand(self.p, 1)
        self.best_params = None
        self.reg = reg
        
        # Add intercept
        col = np.ones((self.n, 1))
        self.X = np.concatenate((col, X), axis=1)
        
        # Standardisation
        if std:
            self.X = self._normalisationMinMax()
            
        # Initialisation
        self.y_pred = self._evaluate()
        self.J = [self._costFunction()]
        self.J_grad = [self._gradients()]
        
        
    def _evaluate(self):
        """
        Compute the model LINEAR REGRESSION

        Returns
        -------
        ndarray
            prediction

        """
        return np.dot(self.X, self.params)
    
    def _costFunction(self):
        """Compute cost function"""
        return np.square(self.y_pred - self.y_gt).sum() / (2*self.y_gt.shape[0])
    
    def _regularizedCostFunction(self, reg='Lasso', penalty=0.1, penalty2=None):
        """Compute cost function in case of regularized LinearRegression"""
        reg_cap = reg.capitalize()
        if reg_cap=='LASSO':
            return self._costFunction() + penalty*np.abs(self.params).sum()
        elif reg_cap=='RIDGE':
            return self._costFunction() + penalty*np.square(self.params).sum()
        elif reg_cap=='ELASTIC':
            if penalty2 is None:
                ValueError('Invalid value for second penalty term')
            else:
                return self._costFunction() + penalty*np.square(self.params).sum() + penalty2*np.abs(self.params).sum()
        else:
            ValueError(f'Invalid penalty name: {reg}\nPossibilities: "Ridge" or "Lasso"')
        
    def _gradients(self):
        """Calculate gradients"""
        gradient = (((self.y_pred - self.y_gt) * self.X).sum(axis=0) / self.X.shape[0]).reshape(self.X.shape[1],1)
        return gradient
    
    def _regularizedGradients(self, reg='Lasso', penalty=0.1, penalty2=0.1):
        """Calculate gradients in case of regularized LinearRegression"""
        
        reg_cap = reg.capitalize()
        if reg_cap=='LASSO':
            return self._gradients() + penalty*self.params.shape[0]
        elif reg_cap=='RIDGE':
            return self._gradients() + 2*penalty*self.params.sum()
        elif reg_cap=='ELASTIC':
            if penalty2 is None:
                ValueError('Invalid value for second penalty term')
            else:
                return self._gradients() + 2*penalty*self.params.sum + penalty2*self.params.shape[0]
        else:
            ValueError(f'Invalid penalty name: {reg}\nPossibilities: "Ridge" or "Lasso" or "Elastic"')
        
    def _evaluateUpdate(self):
        """Compute the change in gradient"""
        return np.abs((self.J[-1] - self.J[-2])/self.J[-2])
    
    def _updateParams(self):
        """Update parameters"""
        return self.params - self.lr*self.J_grad[-1]
    
    def _standardisation(self):
        "Standardisation of input values"
        return (self.X - np.mean(self.X)) / np.std(self.X)
    
    def _normalisationMinMax(self):
        """Normalisation of input values"""
        return ((self.X - np.min(self.X))/np.max(self.X)-np.min(self.X))
        
    def fit(self):
        """
        Model fit

        Returns
        -------
        None.

        """
        diff = [self.eps + 1]
        count = 0
        
        if self.reg is None:
            while diff[-1] > self.eps:
                if count%1000==0:
                    print(f'\n--- Itération {count} ---\nMSE = {self.J[-1]:.7E}')
                count += 1
                self.params = self._updateParams()
                self.y_pred = self._evaluate()
                self.J.append(self._costFunction())
                self.J_grad.append(self._gradients())
                diff.append(self._evaluateUpdate())
        
        else:
            while diff[-1] > self.eps:
                if count%1000==0:
                    print(f'\n--- Itération {count} ---\nMSE = {self.J[-1]:.7E}')
                count += 1
                self.params = self._updateParams()
                self.y_pred = self._evaluate()
                self.J.append(self._regularizedCostFunction())
                self.J_grad.append(self._gradients())
                diff.append(self._evaluateUpdate())
        
        self.best_params = self.params.copy()
        print(f'\nFit process ended with success\nMSE = {self.J[-1]}\n')
    
    def predict(self):
        """
        Make predictions

        Returns
        -------
        y-value
            Prediction of the model as is.
        """
        return np.dot(self.X, self.params)
    
    def score(self):
        """
        Compute coefficient of determination R-squared

        Parameters
        ----------
        y_gt : ndarray
            Target to predict = Ground Truth.

        Returns
        -------
        float
            R-squared.
        """
        return 1-(np.sum((self.y_gt-self.y_pred)**2)/np.sum((self.y_gt-np.mean(self.y_gt))**2))
        
    def plot(self):
        """
        Plot the evolution of the cost function during model fitting

        Returns
        -------
        None.

        """
        plt.figure(figsize=(6,6))
        plt.plot(self.J, label='J($\\theta$)', color='navy', lw=1)
        plt.title('Evolution de la fonction coût J($\\theta$)')
        plt.yscale('log')
        plt.ylabel('LogLoss')
        plt.xlabel('Itération')
        plt.legend()
        
    def plot_featureimportance(self, names=None):
        """
        Plot the features by their importance in the model

        Parameters
        ----------
        names : list, optional
            Names of the features if exists. The default is None.

        Returns
        -------
        None.

        """
        p = np.abs(self.params.reshape(-1))
        
        if names is None:
            names = np.array([f'Feature {k}' for k in range (p.shape[0])])
        else:
            names = np.concat((np.array(['Bias']), names))
        
        # Sort by importance
        index = np.argsort(p)
        names_modified = np.take_along_axis(names, index, axis=0)
        p_modif = np.take_along_axis(p*100/np.sum(p), index, axis=0)

        # Plot
        plt.figure(figsize=(5,5))
        plt.title('Feature importance')
        plt.xlabel('% of importance')
        plt.barh(names_modified, p_modif, color=['navy', 'darkorange'])
        plt.show()
        
    
 