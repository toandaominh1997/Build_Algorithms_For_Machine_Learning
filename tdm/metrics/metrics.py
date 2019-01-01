/*
 * @Author: bigkizd(Toan Dao Minh) 
 * @Date: 2019-01-01 22:57:46 
 * @Last Modified by: bigkizd(Toan Dao Minh)
 * @Last Modified time: 2019-01-01 22:59:37
 */
import autograd.numpy as np

def squared_error(actual, predicted):
    return (actual-predicted)**2

def mean_squared_error(actual, predicted):
    np.mean(squared_error(actual, predicted))
    