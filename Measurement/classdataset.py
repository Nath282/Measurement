#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author : Nathan Le Rétif


# Library import 
import matplotlib.pyplot as plt
import numpy as np
from .classmeasure import Measure

def _isscalar(x) : 
    try :
        float(x)
        return True 
    except ValueError : 
        return False
    
def _pattern_matching(value : list, sep = str) :
    # Check if value is equivalent to a measure, otherwise returns the remerged string
    if len(value) == 1 and _isscalar(value[0]) :
        res = Measure(float(value[0]))
    elif len(value) == 2 and _isscalar(value[0]) and not _isscalar(value[1]) : 
        res = Measure(float(value[0]), unit=value[1])
    elif len(value) == 2 and _isscalar(value[0]) and _isscalar(value[1]) : 
        res = Measure(float(value[0]), float(value[1]))
    elif len(value) == 3 and _isscalar(value[0]) and _isscalar(value[1]) and not _isscalar(value[2]) : 
        res = Measure(float(value[0]), float(value[1]), unit=value[2])
    else : 
        res = sep.join(value)
    return res


# =============== #
# Class defintion #
# =============== #

class DataSet : 

    # ======================= #
    # Instance initialisation #
    # ======================= #

    def __init__(self, data, metadata=None):
        self._data = data
        self._metadata = metadata or {} 

    @property
    def data(self) : return self._data

    @property
    def metadata(self) : return self._metadata

    def __getitem__(self, key):
        return self.data[key]

    # ============ #
    # File reading #
    # ============ #

    @staticmethod
    def _parse_data(lines, data_unc) : 
        # parsing of data section
        labels = lines[0]
        if data_unc : 
            sigmas = np.array(lines[1], dtype=float)
            data_aux = np.array(lines[2:], dtype=float)
        else : 
            sigmas = np.zeros(len(labels))
            data_aux = np.array(lines[1:], dtype=float)
        # Data formatting in presection_contentation for a dataset class
        data = {}
        for k in range(len(labels)) : 
            data[labels[k]] = Measure(data_aux[:,k], sigmas[k])
        return data
    
    @staticmethod
    def _parse_section(lines, sep) :
        parameters = {}
        for line in lines : 
            label, value = line[0], line[1:]
            parameters[label] = _pattern_matching(value, sep)
        return parameters


    @staticmethod
    def read_file(filepath,                             # path of the file 
                  sep = ',',                            # string used to separate value
                  data_section='Data',                  # name of the section containing the data
                  data_unc = True,                      # if True, will consider the first line of data as the corresponding data columns uncertainties
                  metadata_sections=['Parameters'],     # list of all parsed metadata sections 
                  section_delimiter=('--','--'),        # delimiter used in the data file to indicate sections (ex : --Data-- or --Parameters--)
                  encoding='utf8') :                    # file encoding
        
        with open(filepath, 'r', encoding=encoding) as file : 

            d1,d2 = section_delimiter
            current_section = None
            sections = {}
            data, metadata = {}, {}

            # Line by line reading of the file to parse into sections
            for line in file.readlines() :

                line = line.strip()
                if line.startswith(d1) and line.endswith(d2) : 
                    current_section = line[len(d1):-len(d2)]
                    sections[current_section] = []
                    continue

                if current_section is not None and line != '' :
                    sections[current_section].append(line.split(sep))

            # Check if a data section has been found
            if data_section not in sections.keys() : 
                raise ValueError("Data section coult not be found in the file")
            
            # Parsing ad formatting of each individual section
            for section,lines in sections.items() : 
                if section == data_section : 
                    data = DataSet._parse_data(lines, data_unc)
                elif section in metadata_sections : 
                    metadata[section] = DataSet._parse_section(lines, sep)
            
        return DataSet(data, metadata)
            

                        


            
