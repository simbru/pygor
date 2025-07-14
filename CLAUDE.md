## Project Structure and Design Principles

- Take note of how repository is structured, and how we went from tinkering with code in a notebook to various implementations in scripts. Simple methods live in the corresponding class file, and advanced functions live in the directory structure. That way, simple short-hand access to complex functions can happen by passing functions through to methods in the class
- Always adhere to projects modular organisation structure

## Module Specific Notes

- In the strf module, the axes are 0, 1, 2, 3 representing cell, time, y, x
- In the strf module, n receptive fields will belong to the same cell as denoted by obj.num_colour, and therefore the multidimensional_reshape function is crucial in a lot of operations