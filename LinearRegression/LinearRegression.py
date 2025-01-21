# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 10:44:13 2024

@author: peill
          
"""

import numpy as np
import matplotlib.pyplot as plt

class LinearRegression():
    
    def __init__(self, lr=0.003, eps=1e-5):
        if lr<=0:
            raise ValueError(f'Learning rate must be strictly positive. You entered lr={lr}')
        if eps<=0:
            raise ValueError(f'Epsilon must be strictly positive. You entered eps={eps}')
            
        self.lr = lr
        self.eps = eps

    def _evaluate(self, X):
        """
        Compute the model LINEAR REGRESSION

        Returns
        -------
        ndarray
            prediction

        """
        return np.dot(X, self.params)
    
    def _costFunction(self, y_gt, y_pred):
        """Compute cost function"""
        return np.square(y_pred - y_gt).sum() / (2*y_gt.shape[0])
    
    def _regularizedCostFunction(self, y_gt, y_pred):
        """Compute cost function in case of regularized LinearRegression"""
        reg_cap = self.regularization.capitalize()
        if reg_cap=='LASSO':
            return self._costFunction(y_gt, y_pred) + self.penalty*np.abs(self.params).sum()
        elif reg_cap=='RIDGE':
            return self._costFunction(y_gt, y_pred) + self.penalty*np.square(self.params).sum()
        elif reg_cap=='ELASTIC':
            if self.penalty2 is None:
                ValueError('Invalid value for second penalty term')
            else:
                return self._costFunction(y_gt, y_pred) + self.penalty*np.square(self.params).sum() + self.penalty2*np.abs(self.params).sum()
        else:
            ValueError(f'Invalid penalty name: {self.regularization}\nPossibilities: "Ridge" or "Lasso"')
        
    def _gradients(self, X, y_gt, y_pred):
        """Calculate gradients"""
        gradient = (((y_pred - y_gt) * X).sum(axis=0) / X.shape[0]).reshape(X.shape[1],1)
        return gradient
    
    def _regularizedGradients(self, X, y_gt, y_pred):
        """Calculate gradients in case of regularized LinearRegression"""
        
        reg_cap = self.regularization.capitalize()
        if reg_cap=='LASSO':
            return self._gradients(X, y_gt, y_pred) + self.penalty*self.params.shape[0]
        elif reg_cap=='RIDGE':
            return self._gradients(X, y_gt, y_pred) + 2*self.penalty*self.params.sum()
        elif reg_cap=='ELASTIC':
            return self._gradients(X, y_gt, y_pred) + 2*self.penalty*self.params.sum() + self.penalty2*self.params.shape[0]
        else:
            ValueError(f'Invalid regularization name: {self.regularization}\nPossibilities: "Ridge" or "Lasso" or "Elastic"')
        
    def _evaluateUpdate(self):
        """Compute the change in gradient"""
        return np.abs((self.J[-1] - self.J[-2])/self.J[-2])
    
    def _updateParams(self):
        """Update parameters"""
        return self.params - self.lr*self.J_grad[-1]
    
    def _standardisation(X):
        "Standardisation of input values"
        return (X - np.mean(X)) / np.std(X)
    
    # def _normalisationMinMax(X):
    #     """Normalisation of input values"""
    #     return ((X - np.min(X))/np.max(X)-np.min(X))
        
    def fit(self, X, y_gt, y_intercept:bool=True, standardize:bool=False, regularization:str=None, penalty:float=0.1, penalty2:float=0.2):
        """
        Fit the model to X and y_gt data

        Parameters
        ----------
        X : ndarray
            Features.
        y_gt : ndarray
            Targets.
        y_intercept : bool, optional
            Add a bias to the linear regression model.
            The default is True.
        standardize : bool, optional
            Standardize X matrix before fitting.
            The default is False.
        regularization : str, optional
            Add a regularization technique. The default is None.
        penalty : float, optional
            Penalty term for Ridge or Lasso. The default is 0.1.
        penalty2 : float, optional
            Second penalty term for ELASTIC regularization. The default is 0.2.

        Returns
        -------
        None.n
        """
        # Initialize model
        self.n, self.p = X.shape[0], X.shape[1]+1
        self.params = np.random.rand(self.p, 1)
        self.regularization = regularization
        
        # Add intercept
        if y_intercept:
            col = np.ones((self.n, 1))
            X = np.concatenate((col, X), axis=1)
        
        # Standardize if necessary
        if standardize:
            X = _standardisation(X)
            
        diff = [self.eps + 1]
        count = 0
        
        if self.regularization is None:
            # Initialisation
            y_pred = self._evaluate(X)
            self.J = [self._costFunction(y_gt, y_pred)]
            self.J_grad = [self._gradients(X, y_gt, y_pred)]
            
            # Update weights
            while diff[-1] > self.eps:
                if count%1000==0:
                    print(f'\n--- Itération {count} ---\nMSE = {self.J[-1]:.7E}')
                count += 1
                self.params = self._updateParams()
                y_pred = self._evaluate(X)
                self.J.append(self._costFunction(y_gt, y_pred))
                self.J_grad.append(self._gradients(X, y_gt, y_pred))
                diff.append(self._evaluateUpdate())
        
        else:
            # Initialisation
            self.penalty = penalty
            self.penalty2 = penalty2
            y_pred = self._evaluate(X)
            self.J = [self._regularizedCostFunction(y_gt, y_pred)]
            self.J_grad = [self._regularizedGradients(X, y_gt, y_pred)]
            
            while diff[-1] > self.eps:
                if count%1000==0:
                    print(f'\n--- Itération {count} ---\nMSE = {self.J[-1]:.7E}')
                count += 1
                self.params = self._updateParams()
                y_pred = self._evaluate(X)
                self.J.append(self._regularizedCostFunction(y_gt, y_pred))
                self.J_grad.append(self._regularizedGradients(X, y_gt, y_pred))
                diff.append(self._evaluateUpdate())
        
        self.best_params = self.params.copy()
        print(f'\nFit process ended with success\nMSE = {self.J[-1]}\n')
    
    def predict(self, X, y_intercept:bool=True):
        """
        Make predictions

        Parameters
        ----------
        X : ndarray
            Input data to make predictions from.

        Returns
        -------
        ndarray
            Model prediction.
        """
        # Add intercept
        if y_intercept:
            col = np.ones((self.n, 1))
            X = np.concatenate((col, X), axis=1)
            
        return np.dot(X, self.params)
    
    def score(self, y_gt, y_pred):
        """
        Compute coefficient of determination

        Parameters
        ----------
        y_gt : ndarray
            Real targets.
        y_pred : ndarray
            Model predictions.

        Returns
        -------
        np.float
            R-squared i.e. coefficient of determination.
        """
        
        return 1-(np.sum((y_gt-y_pred)**2)/np.sum((y_gt-np.mean(y_gt))**2))
    
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
        
    def featureimportance(self, names=None):
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
        p = np.abs(self.params[1:].reshape(-1))
        
        if names is None:
            names = np.array([f'Feature {k}' for k in range (p.shape[0])])
        
        
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
        
    def plot_actualVSpredicted(self, y_gt, y_pred):
        """
        Visual representation of the performance of the linear model

        Parameters
        ----------
        y_gt : ndarray
            Real targets.
        y_pred : ndarray
            Predictions.

        Returns
        -------
        None.

        """
        x = np.linspace(np.min(y_gt)*0.7, np.max(y_gt)*1.3, 50)
        y = x
        error = 0.05 * y
        y_min = y - error
        y_max = y + error

        plt.figure(figsize=(7,7))
        plt.title('Actual vs Predicted')
        plt.scatter(y_gt, y_pred, alpha=0.8, s=10, color='navy')
        plt.plot(x, y, lw=1, ls='-', color='darkorange', label='y=x')
        plt.fill_between(x, y_min, y_max, color="darkorange", alpha=0.2, label="Bande d'erreur (±5%)")
        plt.xlim(np.min(y_gt)*0.8, np.max(y_gt)*1.2)
        plt.ylim(np.min(y_pred)*0.8, np.max(y_pred)*1.2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.legend()

 """ TRY REGULARIZE AND NORMALIZE"""