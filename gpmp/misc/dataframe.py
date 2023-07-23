## --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022, CentraleSupelec
# License: GPLv3 (see LICENSE)
## --------------------------------------------------------------

import numpy as np

def ftos(x, fp=3):
    if x == float('inf'):
        return "+Inf"
    elif x == float('-inf'):
        return "-Inf"
    abs_x = abs(x)
    if x == 0:
        return "0.0"
    elif abs_x >= 0.1 and abs_x < 1000:
        return f"{x:.{fp}f}"
    elif abs_x >= 0.01 and abs_x < 0.1:
        return f"{x:.{fp+1}f}"
    else:
        exponent = int(np.floor(np.log10(abs_x)))
        coeff = x / 10**exponent
        return f"{coeff:.{fp}f}e{exponent}"


class DataFrame:
    def __init__(self, data, colnames, rownames):
        self.data = np.array(data)
        self.rownames = rownames
        self.colnames = colnames
        
    def __getitem__(self, key):
        if isinstance(key, tuple):
            row_key, col_key = key
            if isinstance(row_key, slice) and isinstance(col_key, slice):
                return DataFrame(self.data[row_key, col_key], self.rownames[row_key], self.colnames[col_key])
            elif isinstance(row_key, slice):
                return DataFrame(self.data[row_key, self.colnames.index(col_key)], self.rownames[row_key], [col_key])
            elif isinstance(col_key, slice):
                return DataFrame(self.data[self.rownames.index(row_key), col_key], [row_key], self.colnames[col_key])
            else:
                return self.data[self.rownames.index(row_key), self.colnames.index(col_key)]
        elif isinstance(key, str):
            if key in self.rownames:
                return DataFrame(self.data[self.rownames.index(key), :], [key], self.colnames)
            elif key in self.colnames:
                return DataFrame(self.data[:, self.colnames.index(key)], self.rownames, [key])
            else:
                raise KeyError(f"Key '{key}' not found in row or column names")
        else:
            raise TypeError("Invalid key type. Must be a tuple or a string.")
    
    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            row_key, col_key = key
            if isinstance(row_key, slice) and isinstance(col_key, slice):
                self.data[row_key, col_key] = value
            elif isinstance(row_key, slice):
                self.data[row_key, self.colnames.index(col_key)] = value
            elif isinstance(col_key, slice):
                self.data[self.rownames.index(row_key), col_key] = value
            else:
                self.data[self.rownames.index(row_key), self.colnames.index(col_key)] = value
        elif isinstance(key, str):
            if key in self.rownames:
                self.data[self.rownames.index(key), :] = value
            elif key in self.colnames:
                self.data[:, self.colnames.index(key)] = value
            else:
                raise KeyError(f"Key '{key}' not found in row or column names")
        else:
            raise TypeError("Invalid key type. Must be a tuple or a string.")

    def __repr__(self):
        header = [[''] + self.colnames] 
        rows = header + \
            [[self.rownames[i]+':'] + \
             [ftos(self.data[i, j]) for j in range(self.data.shape[1])] \
             for i in range(self.data.shape[0])]

        min_width = 8
        col_widths = [max(min_width, max(len(str(rows[i][j])) for i in range(len(rows)))) \
                      for j in range(len(rows[0]))]

        formatted_rows = [' '.join(str(rows[i][j]).rjust(col_widths[j]) \
                                   for j in range(len(rows[0]))) \
                          for i in range(len(rows))]

        return '\n'.join(formatted_rows)
        
    def append_row(self, row_data, row_name):
        self.data = np.vstack([self.data, row_data])
        self.rownames.append(row_name)
    
    def append_col(self, col_data, col_name):
        self.data = np.hstack([self.data, np.atleast_2d(col_data).T])
        self.colnames.append(col_name)

    def concat(self, other, axis=0):
        if axis == 0:
            if self.colnames != other.colnames:
                raise ValueError("DataFrames must have the same column names to concatenate vertically")
            new_data = np.concatenate([self.data, other.data], axis=0)
            new_rownames = self.rownames + other.rownames
            return DataFrame(new_data, self.colnames, new_rownames)
        elif axis == 1:
            if self.rownames != other.rownames:
                raise ValueError("DataFrames must have the same row names to concatenate horizontally")
            new_data = np.concatenate([self.data, other.data], axis=1)
            new_colnames = self.colnames + other.colnames
            return DataFrame(new_data, new_colnames, self.rownames)
        else:
            raise ValueError("Axis must be 0 or 1")
