# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 10:21:36 2020

@author: Daniel
"""

##### Don't need this anymore? Now have ordered templates #####



import numpy as np

def distance(x,y):
    assert all([len(i) == 2 for i in [x,y]])
    diff = np.abs(x - y)
    return np.linalg.norm(diff)

#https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))



def find_order(arr):
    arr = arr.astype('int64') # change dtype to avoid overflow errors
    
    index = 0 # starting index should be irrelevant
    
    order = [index]
    
    #while?
    
    #list all distances from indexed point
    dists = [distance(arr[index],point) for point in arr] 
    #need direction too?
    angles = [angle_between(arr[index], point) for point in arr]
    
    #find smallest 3
    closest_idxs = np.argpartition(dists, 3)[:3]
    
    #first will be itself
    front,behind = closest_idxs[1],closest_idxs[2]
    
    #add to order
    order.append(behind)
    order.insert(0, front)
    print(order)
    
    while True:
        
        front_dists = [distance(arr[front],point) for point in arr] 
        front_angles = np.array([angle_between(arr[front], point) for point in arr])
        #print(front_angles)
        behind_dists = [distance(arr[behind],point) for point in arr]
        behind_angles = np.array([angle_between(arr[behind], point) for point in arr])

        # front_closest = np.argpartition(front_dists, 100)
        # print(front_closest[:10])
        # print(np.partition(front_dists, 3)[:10])
        # behind_closest = np.argpartition(behind_dists, 3)
        # print(behind_closest[:10])
        
        front_closest_idx = np.where(front_dists == np.partition(front_dists, 2)[2])[0]
        behind_closest_idx = np.where(behind_dists == np.partition(behind_dists, 2)[2])[0]
        
        angles = [front_angles,behind_angles]
        
        for i,neighbours in enumerate([front_closest_idx,behind_closest_idx]):
            print(f'neighbours are {neighbours}')
            if i == 0:
                #goes in the front
                idx = 0
            elif i == 1:
                #goes in the end
                idx = len(order)+ 1
            else:
                print('PROBLEM!!')
            
            if len(neighbours) == 2:
                assert any([neighbour in order for neighbour in neighbours]), 'neither were in order'
                for neighbour in neighbours:
                    if neighbour not in order:
                        print('easy case, inserting...')
                        order.insert(idx, neighbour)
                
                
                
            elif len(neighbours) < 2:
                print('reached end?')
                if neighbours[0] in order:
                    print('neighbours already in')
            else:
                print(f'there are {len(neighbours)} neighbours')
                not_already_in = np.array([x for x in neighbours if x not in order])
                print(f'{len(not_already_in)} of which are not in order')
                print(f'not_already_in: {not_already_in}')
                if len(not_already_in) == 1:
                    print('inserting...')
                    order.insert(idx,not_already_in[0])
                else:
                    print('comparing angles')
                    not_already_in_angles = angles[i][not_already_in]
                    print(not_already_in_angles)
                    print(f'inserting element {np.argmin(not_already_in_angles)} index {not_already_in[np.argmin(not_already_in_angles)]} ')
                    order.insert(idx,not_already_in[np.argmin(not_already_in_angles)])
            print(order)
    
    
    
    
    
    
    
    
    
    
    
    
    
        # front1,front2 = front_closest[:3][1],front_closest[:3][2]
        # behind1,behind2 = behind_closest[:3][1],behind_closest[:3][2]
        
        # if all([x in order for x in [front1,front2,behind1,behind2]]):
        #     break
        
        # if front1 in order and front2 not in order:
        #     print(f'adding {front2}')
        #     order.insert(0,front2)
        #     front = front2
        #     print(order)
        # elif front2 in order and front1 not in order:
        #     print(f'adding {front1}')
        #     order.insert(0,front1)
        #     front = front1
        #     print(order)
        # else:
        #     print(f'Reached end of front, {front1} & {front2} already in')
            
        # if behind1 in order and behind2 not in order:
        #     print(f'adding {behind2}')
        #     order.append(behind2)
        #     behind = behind2
        #     print(order)
        # elif behind2 in order and behind1 not in order:
        #     print(f'adding {behind1}')
        #     order.append(behind1)
        #     behind = behind1
        #     print(order)
        # else:
        #     print(f'reached end of behind {behind1} & {behind2} alreaedy in')
            
       
    return order