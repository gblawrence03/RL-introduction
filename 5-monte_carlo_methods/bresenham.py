'''
While trying to work out how to calculate which squares the car passes through 
between two time steps, I came across Bresenham's algorithm for rasterizing lines
and decided to implement it myself as an interesting side task. 
The function takes the start and end coordinates of the line and outputs a list 
of coordinates of the squares that the line passes through. 
'''
def bresenham(start, end):
    # horizontal case
    if start[1] == end[1]:
        return [(x, start[1]) for x in range(min(start[0], end[0]), max(start[0], end[0]) + 1)]
    # vertical case
    if start[0] == end[0]:
        return [(start[0], y) for y in range(min(start[1], end[1]), max(start[1], end[1]) + 1)]
    slope = (start[1] - end[1]) / (start[0] - end[0])
    if abs(slope) <= 1:
        if start[0] <= end[0]:
            p1, p2 = start, end
        else:
            p1, p2 = end, start
        return [(p1[0] + x, p1[1] + round(x * slope)) for x in range(0, p2[0] - p1[0] + 1)]
    else:
        if start[1] <= end[1]:
            p1, p2 = start, end
        else:
            p1, p2 = end, start
        return [(p1[0] + round(y * (1 / slope)), p1[1] + y) for y in range(0, p2[1] - p1[1] + 1)]

''' 
Simple function to visualise sets of coordinates in a grid with any character. 
The data arg is a dict. The key is the character and the value is the set of 
coordinates to be represented by that character.
If a coordinate appears in multiple sets of coordinates, the character used 
will be the last one for which the coordinate appears. 
'''
def visualise(bottomleft, topright, data):
    for y in range(topright[1], bottomleft[1] - 1, -1):
        for x in range(bottomleft[0], topright[0] + 1): 
            char = ' '
            for c in data.keys():
                if (x, y) in data[c]:
                    char = c
            print(char, end=' ')
        print()
