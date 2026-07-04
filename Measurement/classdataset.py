#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author : Nathan Le Rétif

"""
Defines a DataSet class based on the Measure class to encapsulating the reading of txt-like data files from devices and the conversion to Measure instances
Specifically, the class was made to parse data but also metadata, usually found in such files, which can be useful for data treatment (like the devices paramters which could change for each value depending on the experiment)
Attributes : - data : dictionary of key,value of the form m.label,m where m is a Measure instance
             - metadata : dictionnary storing all quantitative metadata parameters (each described by a Measure instance), organized in different sections (each containing their own parameters)
Rmq : only quantitative parameters are stored, purely textual information in the file will not be accessible from the DataSet instance
      is heavily based on the Measure._string_match() method for the parsing
"""

# Library import 
import matplotlib.pyplot as plt
import numpy as np
from .classmeasure import Measure


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
    def _parse_data(lines, sep) : 
        """
        private method for parsing the data section based on the Measure._string_match() method, according to the following rules :
        - the data section must be organised in different columns separated by sep,
        - each column must follow the ordering rules defined by Measure._string_match() 
        Arguments : lines : list of string where each string corresponds to a line of the data section
                    sep : string dividing each field on one line
        Rmq : a column is generally expected to be of the form label, unit, unc, value1, value2,...
              see Measure.string_match() for more precisions 
        """
        try : 
            elements = np.asarray([line.split(sep) for line in lines], dtype=str)
        except ValueError : 
            raise ValueError("lines are inhomegeneous, check each line size")

        data = {}
        for k in range(len(elements[0])) :
            m = Measure._string_match(sep.join(elements[:,k]))
            if m.label == '' : raise ValueError(f'No label found for row {k}')
            data[m.label] = m
        return data
    
    @staticmethod
    def _parse_section(lines, sep) :
        """
        private method for parsing a metadata section based on the Measure._string_match() method according the following rules : 
        - each line of the section is tested to be a Measure instance (cf Measure._string_match()) 
        - if recognized, the Measure instance m will be added to a dictionary following the pattern : key=m.label,value=m
        Arguments : lines : list of string where each string corresponds to a line of the section
                    sep : string dividing each field on one line
        Rmq : a generally expected field is of the form : Wavelength, 532, .2, nm
        """
        parameters = {}
        for line in lines : 
            try : 
                ms = Measure._string_match(line, sep)
                parameters[ms.label] = ms
            except ValueError : 
                pass
        return parameters


    @staticmethod
    def read_file(filepath,                             # path of the file 
                  sep = ',',                            # string used to separate value
                  data_section='Data',                  # name of the section containing the data
                  metadata_sections=['Parameters'],     # list of all parsed metadata sections 
                  section_delimiters=('--','--'),       # delimiter used in the data file to indicate sections (ex : --Data-- or --Parameters--)
                  encoding='utf8') :                    # file encoding
        """
        Method to parse a txt-like file and match it to Measures instances
        each section is recognised by its name surronded by the section delimiters (ex : --Data-- or --Parameters-- )
        returns a DataSet instance where : 
            - data : {m.label : m} is a dictionnary containing all measure m found in the data section (see _parse_data for more details)
            - metadata : {section : {p.label : p}} is a dictionnary containing all metadata sections and their recognized paramters p (where p is a Measure instance) ((see _parse_section for more details))
        """
        with open(filepath, 'r', encoding=encoding) as file : 

            d1,d2 = section_delimiters
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
                    sections[current_section].append(line)

            # Check if a data section has been found
            if data_section not in sections : 
                raise ValueError("Data section coult not be found in the file")
            
            # Parsing ad formatting of each individual section
            for section,lines in sections.items() : 
                if section == data_section : 
                    data = DataSet._parse_data(lines, sep)
                elif section in metadata_sections : 
                    metadata[section] = DataSet._parse_section(lines, sep)
        
        return DataSet(data, metadata)
            

                        


            
