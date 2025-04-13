# -*- coding: utf-8 -*-
"""
The Game of Life (GoL) module named in honour of John Conway

This module defines the classes required for the GoL simulation.

Created on Tue Jan 15 12:21:17 2019

@author: shakes
"""
import numpy as np
from scipy import signal
import rle

class GameOfLife:
    '''
    Object for computing Conway's Game of Life (GoL) cellular machine/automata
    '''
    def __init__(self, N=256, finite=False, fastMode=False):
        self.grid = np.zeros((N,N), np.int64)
        self.neighborhood = np.ones((3,3), np.int64) # 8 connected kernel
        self.neighborhood[1,1] = 0 #do not count centre pixel
        self.finite = finite
        self.fastMode = fastMode
        self.aliveValue = 1
        self.deadValue = 0
        
    def getStates(self):
        '''
        Returns the current states of the cells
        '''
        return self.grid
    
    def getGrid(self):
        '''
        Same as getStates()
        '''
        return self.getStates()
               
    def evolve(self):
        '''
        Given the current states of the cells, apply the GoL rules:
        - Any live cell with fewer than two live neighbors dies, as if by underpopulation.
        - Any live cell with two or three live neighbors lives on to the next generation.
        - Any live cell with more than three live neighbors dies, as if by overpopulation.
        - Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction
        '''
        #get weighted sum of neighbors
        #PART A & E CODE HERE
        grid = self.getGrid()
        rows, cols = grid.shape

        # Create a grid with the same dimension to store number of neighbours
        neighbours = np.zeros((rows, cols), np.int64)

        for row in range(rows):
            for col in range(cols):
                # min max to avoid indexing below 0
                row_min = max(row - 1, 0)
                row_max = min(row + 2, rows)
                col_min = max(col - 1, 0)
                col_max = min(col + 2, cols)

                neighbour_count = np.sum(grid[row_min:row_max, col_min:col_max]) - grid[row, col]
                neighbours[row, col] = neighbour_count
                
        #implement the GoL rules by thresholding the weights
        #PART A CODE HERE
        evolvedGrid = np.zeros((rows, cols), np.int64)

        for row in range(rows):
            for col in range(cols):
                if grid[row, col] == 0: # currently dead
                    if neighbours[row, col] == 3: # reproduction
                        evolvedGrid[row, col] = 1
                else: # currently alive
                    if neighbours[row, col] < 2: # underpopulation
                        evolvedGrid[row, col]= 0
                    elif neighbours[row, col] > 3: # overpopulation
                        evolvedGrid[row, col]= 0
                    else: # survival
                        evolvedGrid[row, col]= 1
        #update the grid
        self.grid = evolvedGrid
    
    def insertBlinker(self, index=(0,0)):
        '''
        Insert a blinker oscillator construct at the index position
        '''
        self.grid[index[0], index[1]+1] = self.aliveValue
        self.grid[index[0]+1, index[1]+1] = self.aliveValue
        self.grid[index[0]+2, index[1]+1] = self.aliveValue
        
    def insertGlider(self, index=(0,0)):
        '''
        Insert a glider construct at the index position
        '''
        self.grid[index[0], index[1]+1] = self.aliveValue
        self.grid[index[0]+1, index[1]+2] = self.aliveValue
        self.grid[index[0]+2, index[1]] = self.aliveValue
        self.grid[index[0]+2, index[1]+1] = self.aliveValue
        self.grid[index[0]+2, index[1]+2] = self.aliveValue
        
    def insertGliderGun(self, index=(0,0)):
        '''
        Insert a glider construct at the index position
        '''
        self.grid[index[0]+1, index[1]+25] = self.aliveValue
        
        self.grid[index[0]+2, index[1]+23] = self.aliveValue
        self.grid[index[0]+2, index[1]+25] = self.aliveValue
        
        self.grid[index[0]+3, index[1]+13] = self.aliveValue
        self.grid[index[0]+3, index[1]+14] = self.aliveValue
        self.grid[index[0]+3, index[1]+21] = self.aliveValue
        self.grid[index[0]+3, index[1]+22] = self.aliveValue
        self.grid[index[0]+3, index[1]+35] = self.aliveValue
        self.grid[index[0]+3, index[1]+36] = self.aliveValue
        
        self.grid[index[0]+4, index[1]+12] = self.aliveValue
        self.grid[index[0]+4, index[1]+16] = self.aliveValue
        self.grid[index[0]+4, index[1]+21] = self.aliveValue
        self.grid[index[0]+4, index[1]+22] = self.aliveValue
        self.grid[index[0]+4, index[1]+35] = self.aliveValue
        self.grid[index[0]+4, index[1]+36] = self.aliveValue
        
        self.grid[index[0]+5, index[1]+1] = self.aliveValue
        self.grid[index[0]+5, index[1]+2] = self.aliveValue
        self.grid[index[0]+5, index[1]+11] = self.aliveValue
        self.grid[index[0]+5, index[1]+17] = self.aliveValue
        self.grid[index[0]+5, index[1]+21] = self.aliveValue
        self.grid[index[0]+5, index[1]+22] = self.aliveValue
        
        self.grid[index[0]+6, index[1]+1] = self.aliveValue
        self.grid[index[0]+6, index[1]+2] = self.aliveValue
        self.grid[index[0]+6, index[1]+11] = self.aliveValue
        self.grid[index[0]+6, index[1]+15] = self.aliveValue
        self.grid[index[0]+6, index[1]+17] = self.aliveValue
        #self.grid[index[0]+6, index[1]+17] = self.aliveValue
        self.grid[index[0]+6, index[1]+18] = self.aliveValue # fix
        self.grid[index[0]+6, index[1]+23] = self.aliveValue
        self.grid[index[0]+6, index[1]+25] = self.aliveValue
        
        self.grid[index[0]+7, index[1]+11] = self.aliveValue
        self.grid[index[0]+7, index[1]+17] = self.aliveValue
        self.grid[index[0]+7, index[1]+25] = self.aliveValue
        
        self.grid[index[0]+8, index[1]+12] = self.aliveValue
        self.grid[index[0]+8, index[1]+16] = self.aliveValue
        
        self.grid[index[0]+9, index[1]+13] = self.aliveValue
        self.grid[index[0]+9, index[1]+14] = self.aliveValue
        
    def insertFromPlainText(self, txtString, pad=0):
        '''
        Assumes txtString contains the entire pattern as a human readable pattern without comments
        '''


    def insertFromRLE(self, rleString, pad=0):
        '''
        Given string loaded from RLE file, populate the game grid
        '''

        