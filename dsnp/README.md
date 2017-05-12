## Synopsis

The project is the implementation of k dimension tree.

## Motivation

The project is for the programming assignment of the Course, DSnP Spring 2016, in National Taiwan University.

## Installation

./configure

## Command Line Argument
    
    Usage:   ./bin/kdtree <input file>
    Example: ./bin/kdtree ./input/input1.txt

## Console Options
    
    i   - insert a node
    d   - delete a node
    r   - range search
    n   - nearest neighbor
    q   - exit
    
    Notice: The result of range search will be save in the ouput directory.  
            You see the result, after doing the range search.

# Example

    The time to build this kd tree:
        real:    0.03 (s)
        user:    0.03 (s)
        sys:     0.00 (s)

    (a) Find the nearest neighbor of (0.5, 0.5)

    The nearest neighbor is     (0.498659, 0.500257)
        The distance is 0.001365.

    (b) How many points there are in the rectangle.

        The given rectangle is (0.300000, 0.300000), (0.300000, 0.410000), (0.600000, 0.300000), (0.600000, 0.410000).
        There are 3338 node in the given range.

    (c) How many nearest neighbor calculations can your 2d-tree implementation perform per second.

        find 125000.000000 nearest neighbor per second.

    What would you want to do?

    Options:
         i   - insert a node
         d   - delete a node
         r   - range search
         n   - nearest neighbor
         q   - exit

    > 

## Authors

Roy LU
b01501061@ntu.edu.tw

## License

This project is licensed under the MIT License
