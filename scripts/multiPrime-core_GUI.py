#!/bin/python
__date__ = "2023-7-13"
__author__ = "Junbo Yang"
__email__ = "yang_junbo_hi@126.com"
__license__ = "MIT"

import itertools

import dearpygui.dearpygui as dpg
import webbrowser
import math
from math import log10
import re
import sys
import numpy as np
import pandas as pd
import json
import os
from bisect import bisect_left
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import reduce
from itertools import product
from itertools import repeat
import multiprocessing
# pyinstaller: multiprocessing.freeze_support()
from multiprocessing import Manager
from operator import mul
from statistics import mean
import time

dpg.create_context()

base_path = os.path.dirname(os.path.realpath(sys.argv[0]))
def get_abs_path():
    return os.path.join(base_path)
print(get_abs_path)
# dpg.set_global_font_scale(1.6)


def _hyperlink(text, address):
    b = dpg.add_button(label=text, callback=lambda: webbrowser.open(address))
    dpg.bind_item_theme(b, "__demo_hyperlinkTheme")


def _update_dynamic_textures(sender, app_data, user_data):
    new_color = app_data
    new_color[0] = new_color[0]
    new_color[1] = new_color[1]
    new_color[2] = new_color[2]
    new_color[3] = new_color[3]

    if user_data == 1:
        texture_data = []
        for i in range(100 * 100):
            texture_data.append(new_color[0])
            texture_data.append(new_color[1])
            texture_data.append(new_color[2])
            texture_data.append(new_color[3])
        dpg.set_value("__demo_dynamic_texture_1", texture_data)

    elif user_data == 2:
        texture_data = []
        for i in range(50 * 50):
            texture_data.append(new_color[0])
            texture_data.append(new_color[1])
            texture_data.append(new_color[2])
            texture_data.append(new_color[3])
        dpg.set_value("__demo_dynamic_texture_2", texture_data)


def _help(message):
    last_item = dpg.last_item()
    group = dpg.add_group(horizontal=True)
    dpg.move_item(last_item, parent=group)
    dpg.capture_next_item(lambda s: dpg.move_item(s, parent=group))
    t = dpg.add_text("(?)", color=[0, 255, 0])
    with dpg.tooltip(t):
        dpg.add_text(message)


def _hsv_to_rgb(h, s, v):
    if s == 0.0:
        return v, v, v
    i = int(h * 6.)  # XXX assume int() truncates!
    f = (h * 6.) - i
    p, q, t = v * (1. - s), v * (1. - s * f), v * (1. - s * (1. - f))
    i %= 6
    if i == 0: return 255 * v, 255 * t, 255 * p
    if i == 1: return 255 * q, 255 * v, 255 * p
    if i == 2: return 255 * p, 255 * v, 255 * t
    if i == 3: return 255 * p, 255 * q, 255 * v
    if i == 4: return 255 * t, 255 * p, 255 * v
    if i == 5: return 255 * v, 255 * p, 255 * q


def _config(sender, keyword, user_data):
    widget_type = dpg.get_item_type(sender)
    # Gets the item’s type.
    #
    # Returns
    # type as a string or None
    items = user_data

    if widget_type == "mvAppItemType::mvRadioButton":
        value = True
    else:
        keyword = dpg.get_item_label(sender)
        value = dpg.get_value(sender)

    if isinstance(user_data, list):
        for item in items:
            dpg.configure_item(item, **{keyword: value})
    else:
        dpg.configure_item(items, **{keyword: value})
        # Configures an item after creation.


def _add_config_options(item, columns, *names, **kwargs):
    if columns == 1:
        if 'before' in kwargs:
            for name in names:
                dpg.add_checkbox(label=name, callback=_config, user_data=item, before=kwargs['before'],
                                 default_value=dpg.get_item_configuration(item)[name])
        else:
            for name in names:
                dpg.add_checkbox(label=name, callback=_config, user_data=item,
                                 default_value=dpg.get_item_configuration(item)[name])
    else:
        if 'before' in kwargs:
            dpg.push_container_stack(dpg.add_table(header_row=False, before=kwargs['before']))
        else:
            dpg.push_container_stack(dpg.add_table(header_row=False))

        for i in range(columns):
            dpg.add_table_column()
        for i in range(int(len(names) / columns)):
            with dpg.table_row():
                for j in range(columns):
                    dpg.add_checkbox(label=names[i * columns + j],
                                     callback=_config, user_data=item,
                                     default_value=dpg.get_item_configuration(item)[names[i * columns + j]])
        dpg.pop_container_stack()


def DP_reset_settings():
    dpg.set_value(item="DPrime_input", value="")
    dpg.set_value(item="DPrime_output", value="DPrime.out")
    dpg.set_value(item="DPrime_plen", value=18)
    dpg.set_value(item="DPrime_coverage", value=0.6)
    dpg.set_value(item="DPrime_dnum", value=4)
    dpg.set_value(item="DPrime_degeneracy", value=10)
    dpg.set_value(item="DPrime_entropy", value=3.6)
    dpg.set_value(item="DPrime_size", value="200,500")
    dpg.set_value(item="DPrime_positions", value="1,2,-1")
    dpg.set_value(item="DPrime_var", value=1)
    dpg.set_value(item="DPrime_hairpin", value=4)
    dpg.set_value(item="DPrime_GC", value="0.2,0.7")
    dpg.set_value(item="DPrime_clamp", value=0.8)
    dpg.set_value(item="DPrime_proc", value=5)
    dpg.set_value(item="DPrime_sprimer", value="DPrime.tmp")
    dpg.set_value(item="DPrime_adaptor", value="TCTTTCCCTACACGACGCTCTTCCGATCT,TCTTTCCCTACACGACGCTCTTCCGATCT")
    dpg.set_value(item="DPrime_diffTm", value=2)
    dpg.set_value(item="DPrime_dposition", value=4)
    dpg.set_value(item="DP_method", value="multiPrime1")
    dpg.set_value(item="DPrime_di_nucl", value=4)

def Dimer_reset_settings():
    dpg.set_value(item="Dimer_input", value="")
    dpg.set_value(item="Dimer_output", value="Dimer.output.xls")
    dpg.set_value(item="Dimer_threshold", value=3.6)
    dpg.set_value(item="Dimer_num", value=5)


def _create_static_textures():
    texture_data1 = []
    for i in range(100 * 100):
        texture_data1.append(255 / 255)
        texture_data1.append(0)
        texture_data1.append(255 / 255)
        texture_data1.append(255 / 255)

    texture_data2 = []
    for i in range(50 * 50):
        texture_data2.append(255 / 255)
        texture_data2.append(255 / 255)
        texture_data2.append(0)
        texture_data2.append(255 / 255)

    texture_data3 = []
    for row in range(50):
        for column in range(50):
            texture_data3.append(255 / 255)
            texture_data3.append(0)
            texture_data3.append(0)
            texture_data3.append(255 / 255)
        for column in range(50):
            texture_data3.append(0)
            texture_data3.append(255 / 255)
            texture_data3.append(0)
            texture_data3.append(255 / 255)
    for row in range(50):
        for column in range(50):
            texture_data3.append(0)
            texture_data3.append(0)
            texture_data3.append(255 / 255)
            texture_data3.append(255 / 255)
        for column in range(50):
            texture_data3.append(255 / 255)
            texture_data3.append(255 / 255)
            texture_data3.append(0)
            texture_data3.append(255 / 255)

    dpg.add_static_texture(100, 100, texture_data1, parent="__demo_texture_container", tag="__demo_static_texture_1",
                           label="Static Texture 1")
    dpg.add_static_texture(50, 50, texture_data2, parent="__demo_texture_container", tag="__demo_static_texture_2",
                           label="Static Texture 2")
    dpg.add_static_texture(100, 100, texture_data3, parent="__demo_texture_container", tag="__demo_static_texture_3",
                           label="Static Texture 3")


def _create_dynamic_textures():
    texture_data1 = []
    for i in range(100 * 100):
        texture_data1.append(255 / 255)
        texture_data1.append(0)
        texture_data1.append(255 / 255)
        texture_data1.append(255 / 255)

    texture_data2 = []
    for i in range(50 * 50):
        texture_data2.append(255 / 255)
        texture_data2.append(255 / 255)
        texture_data2.append(0)
        texture_data2.append(255 / 255)

    dpg.add_dynamic_texture(100, 100, texture_data1, parent="__demo_texture_container", tag="__demo_dynamic_texture_1")
    dpg.add_dynamic_texture(50, 50, texture_data2, parent="__demo_texture_container", tag="__demo_dynamic_texture_2")


def multiPrime_logo_file(sender, app_data):
    """Function to get all files and directories within the selected directory.
    Will populate the listbox "files_listbox" with the items.

    Args:
        sender (obj): Dear PyGui sender widget
        app_data ([obj]): information from the file dialog: file_path_name, file_name, current_path, current_filter, selections[]
    """
    # print(sender)
    # print(app_data)
    file = app_data["file_path_name"]
    # print(file)
    return file


def DPrime_select_directory(sender, app_data):
    with dpg.file_dialog(directory_selector=False, show=True, width=600, height=400,
                         callback=DPrime_select_file):
        dpg.add_file_extension(".msa", color=(255, 0, 255, 255), custom_text="header")
        dpg.add_file_extension(".fasta", color=(255, 255, 0, 255))
        dpg.add_file_extension(".fa", color=(255, 255, 0, 255))
        dpg.add_file_extension(".*", color=(255, 255, 255, 255))
        dpg.add_file_extension("Python(.py){.py}", color=(0, 255, 0, 255))


def DPrime_select_directory_only(sender, app_data):
    with dpg.file_dialog(directory_selector=True, show=True, width=600, height=400,
                         callback=DPrime_select_file):
        dpg.add_file_extension(".msa", color=(255, 0, 255, 255), custom_text="header")
        dpg.add_file_extension(".fasta", color=(255, 255, 0, 255))
        dpg.add_file_extension(".fa", color=(255, 255, 0, 255))
        dpg.add_file_extension(".*", color=(255, 255, 255, 255))
        dpg.add_file_extension("Python(.py){.py}", color=(0, 255, 0, 255))


def DPrime_select_fasta_directory(sender, app_data):
    with dpg.file_dialog(directory_selector=False, show=True, width=600, height=400,
                         callback=DPrime_select_fasta_file):
        dpg.add_file_extension(".fa", color=(255, 255, 0, 255), custom_text="header")
        dpg.add_file_extension(".fasta", color=(255, 255, 0, 255), custom_text="header")
        dpg.add_file_extension(".msa", color=(255, 0, 255, 255))
        dpg.add_file_extension(".*", color=(255, 255, 255, 255))
        dpg.add_file_extension("Python(.py){.py}", color=(0, 255, 0, 255))

def finDimer_select_fasta_directory(sender, app_data):
    with dpg.file_dialog(directory_selector=False, show=True, width=600, height=400,
                         callback=finDimer_select_fasta_file):
        dpg.add_file_extension(".fa", color=(255, 255, 0, 255), custom_text="header")
        dpg.add_file_extension(".fasta", color=(255, 255, 0, 255), custom_text="header")
        dpg.add_file_extension(".msa", color=(255, 0, 255, 255))
        dpg.add_file_extension(".*", color=(255, 255, 255, 255))
        dpg.add_file_extension("Python(.py){.py}", color=(0, 255, 0, 255))

def DPrime_select_fasta_file(sender, app_data):
    # print(sender)
    print(app_data)
    file = app_data["file_path_name"]
    Current_path = app_data["current_path"]
    print(file)
    dpg.set_value(item="DPrime_input_fasta", value=file)
    dpg.set_value(item="DPrime_msa_output", value=Current_path + "\\" + dpg.get_value("DPrime_msa_output"))

def finDimer_select_fasta_file(sender, app_data):
    # print(sender)
    print(app_data)
    file = app_data["file_path_name"]
    Current_path = app_data["current_path"]
    print(file)
    dpg.set_value(item="Dimer_input", value=file)
    dpg.set_value(item="Dimer_output", value=Current_path + "\\" + dpg.get_value("Dimer_output"))


def DPrime_select_file(sender, app_data):
    # print(sender)
    print(app_data)
    file = app_data["file_path_name"]
    Current_path = app_data["current_path"]
    print(file)
    dpg.set_value(item="DPrime_input", value=file)
    dpg.set_value(item="DPrime_sprimer", value=Current_path + "\\" + dpg.get_value("DPrime_sprimer"))
    dpg.set_value(item="DPrime_output", value=Current_path + "\\" + dpg.get_value("DPrime_output"))


def start_loading():
    dpg.configure_item("loading", show=True)
    return


def stop_loading():
    dpg.configure_item("loading", show=False)
    return


def start_loading1():
    dpg.configure_item("loading1", show=True)
    return


def stop_loading1():
    dpg.configure_item("loading1", show=False)
    return

def start_loading2():
    dpg.configure_item("loading2", show=True)
    return


def stop_loading2():
    dpg.configure_item("loading2", show=False)
    return

#######################################################################################################################
degenerate_base = {"-": ["-"], "A": ["A"], "G": ["G"], "C": ["C"], "T": ["T"], "R": ["A", "G"], "Y": ["C", "T"],
                   "M": ["A", "C"], "K": ["G", "T"], "S": ["G", "C"], "W": ["A", "T"], "H": ["A", "T", "C"],
                   "B": ["G", "T", "C"], "V": ["G", "A", "C"], "D": ["G", "A", "T"], "N": ["A", "T", "G", "C"]}

score_table = {"-": 100, "#": 0.00, "A": 1, "G": 1.11, "C": 1.21, "T": 1.40, "R": 2.11, "Y": 2.61, "M": 2.21,
               "K": 2.51, "S": 2.32, "W": 2.40, "H": 3.61, "B": 3.72, "V": 3.32, "D": 3.51, "N": 4.72}

#######################################################################################################################
# This is used for the new version of multiPrime. it is not used in this version of the program.
non_score_table2 = {0.11: 0, 0.21: 0, 0.4: 0, 1.61: 0, 1.51: 0, 1.32: 0, 2.72: 0, -1: 0, -99: 0,
                    -0.4: 3, -0.19: 3, -0.29: 3, 0.71: 3, 0.81: 3, 0.92: 3, 1.92: 3, -1.4: 3, -98.6: 3,
                    -0.21: 1, 0.19: 1, -0.1: 1, 0.9: 1, 1.3: 1, 1.19: 1, 2.3: 1, -1.21: 1, -98.79: 1,
                    -0.11: 2, 0.29: 2, 0.1: 2, 1.5: 2, 1.1: 2, 1.29: 2, 2.5: 2, -1.11: 2, -98.89: 2}
non_score_table = {0.11: "A", 0.21: "A", 0.4: "A", 1.61: "A", 1.51: "A", 1.32: "A", 2.72: "A", -1: "A", -99: "A",
                   -0.4: "T", -0.19: "T", -0.29: "T", 0.71: "T", 0.81: "T", 0.92: "T", 1.92: "T", -1.4: "T", -98.6: "T",
                   -0.21: "C", 0.19: "C", -0.1: "C", 0.9: "C", 1.3: "C", 1.19: "C", 2.3: "C", -1.21: "C", -98.79: "C",
                   -0.11: "G", 0.29: "G", 0.1: "G", 1.5: "G", 1.1: "G", 1.29: "G", 2.5: "G", -1.11: "G", -98.89: "G"}
#######################################################################################################################

trans_score_table = {v: k for k, v in score_table.items()}

##############################################################################################
############################# Calculate free energy ##########################################
##############################################################################################
freedom_of_H_37_table = [[-0.7, -0.81, -0.65, -0.65],
                         [-0.67, -0.72, -0.8, -0.65],
                         [-0.69, -0.87, -0.72, -0.81],
                         [-0.61, -0.69, -0.67, -0.7]]

penalty_of_H_37_table = [[0.4, 0.575, 0.33, 0.73],
                         [0.23, 0.32, 0.17, 0.33],
                         [0.41, 0.45, 0.32, 0.575],
                         [0.33, 0.41, 0.23, 0.4]]

H_bonds_number = [[2, 2.5, 2.5, 2],
                  [2.5, 3, 3, 2.5],
                  [2.5, 3, 3, 2.5],
                  [2, 2.5, 2.5, 2]]
adjust_initiation = {"A": 0.98, "T": 0.98, "C": 1.03, "G": 1.03}
adjust_terminal_TA = 0.4
symmetry_correction = 0.4

##############################################################################################
base2bit = {"A": 0, "C": 1, "G": 2, "T": 3, "#": 4}

##############################################################################################
# 37°C and 1 M NaCl
Htable2 = [[-7.9, -8.5, -8.2, -7.2, 0],
           [-8.4, -8, -9.8, -8.2, 0],
           [-7.8, -10.6, -8, -8.5, 0],
           [-7.2, -7.8, -8.4, -7.9, 0],
           [0, 0, 0, 0, 0]]
Stable2 = [[-22.2, -22.7, -22.2, -21.3, 0],
           [-22.4, -19.9, -24.4, -22.2, 0],
           [-21, -27.2, -19.9, -22.7, 0],
           [-20.4, -21, -22.4, -22.2, 0],
           [0, 0, 0, 0, 0]]
Gtable2 = [[-1, -1.45, -1.3, -0.58, 0],
           [-1.44, -1.84, -2.24, -1.3, 0],
           [-1.28, -2.17, -1.84, -1.45, 0],
           [-0.88, -1.28, -1.44, -1, 0],
           [0, 0, 0, 0, 0]]
H_adjust_initiation = {"A": 2.3, "T": 2.3, "C": 0.1, "G": 0.1}
S_adjust_initiation = {"A": 4.1, "T": 4.1, "C": -2.8, "G": -2.8}
G_adjust_initiation = {"A": 1.03, "T": 1.03, "C": 0.98, "G": 0.98}
H_symmetry_correction = 0
S_symmetry_correction = -1.4
G_symmetry_correction = 0.4
##############################################################################################
# ng/ul
primer_concentration = 100
Mo_concentration = 50
Di_concentration = 1.5
dNTP_concentration = 0.25
Kelvin = 273.15
# reference (Owczarzy et al.,2008)
crossover_point = 0.22

bases = np.array(["A", "C", "G", "T"])
di_bases = []
for i in bases:
    for j in bases:
        di_bases.append(i + j)


def Penalty_points(length, GC, d1, d2):
    return log10((2 ** length * 2 ** GC) / ((2 ** d1 - 0.9) * (2 ** d2 - 0.9)))


di_nucleotides = set()
for i in base2bit.keys():
    single = i * 4
    di_nucleotides.add(single)
    for j in base2bit.keys():
        if i != j:
            di = (i + j) * 4
            di_nucleotides.add(di)
        for k in base2bit.keys():
            if i != j != k:
                tri = (i + j + k) * 3
                di_nucleotides.add(tri)


def score_trans(sequence):
    return reduce(mul, [math.floor(score_table[x]) for x in list(sequence)])


def dege_number(sequence):
    return sum(math.floor(score_table[x]) > 1 for x in list(sequence))


TRANS = str.maketrans("ATGCRYMKSWHBVDN", "TACGYRKMSWDVBHN")


def RC(seq):
    return seq.translate(TRANS)[::-1]


##############################################################################################
############## m_distance which is used to calculate (n)-nt variation coverage ###############
# Caution: this function only works when degeneracy of seq2 < 2 (no degenerate in seq2).
##############################################################################################
def Y_distance(seq1, seq2):
    seq_diff = list(np.array([score_table[x] for x in list(seq1)]) - np.array([score_table[x] for x in list(seq2)]))
    m_dist = [idx for idx in range(len(seq_diff)) if round(seq_diff[idx], 2) not in score_table.values()]
    # print(seq_diff)
    return m_dist


def Y_position(seq1, seq2):
    seq_diff = list(np.array([score_table[x] for x in list(seq1)]) - np.array([score_table[x] for x in list(seq2)]))
    m_dist = [str(idx) + "|" + non_score_table[round(seq_diff[idx], 2)] for idx in range(len(seq_diff)) if
              round(seq_diff[idx], 2) not in score_table.values()]
    # print(seq_diff)
    return m_dist


##############################################################################################
def symmetry(seq):
    if len(seq) % 2 == 1:
        return False
    else:
        F = seq[:int(len(seq) / 2)]
        R = RC(seq[int(len(seq) / 2):][::-1])
        if F == R:
            return True
        else:
            return False


def Calc_deltaH_deltaS(seq):
    Delta_H = 0
    Delta_S = 0
    for n in range(len(seq) - 1):
        i, j = base2bit[seq[n + 1]], base2bit[seq[n]]
        Delta_H += Htable2[i][j]
        Delta_S += Stable2[i][j]
    seq = seq.replace("#", '')
    Delta_H += H_adjust_initiation[seq[0]] + H_adjust_initiation[seq[-1]]
    Delta_S += S_adjust_initiation[seq[0]] + S_adjust_initiation[seq[-1]]
    if symmetry(seq):
        Delta_S += S_symmetry_correction
    return Delta_H * 1000, Delta_S


def Calc_Tm_v2(seq):
    delta_H, delta_S = Calc_deltaH_deltaS(seq)
    Tm_Na_adjust = Mo_concentration

    if dNTP_concentration >= Di_concentration:
        free_divalent = 0.00000000001
    else:
        free_divalent = (Di_concentration - dNTP_concentration) / 1000.0
    R_div_monov_ratio = (math.sqrt(free_divalent)) / (Mo_concentration / 1000)

    if R_div_monov_ratio < crossover_point:
        # use only monovalent salt correction, [equation 22] (Owczarzy et al., 2004)
        correction = (((4.29 * GC_fraction(seq)) - 3.95) * pow(10, -5) * math.log(Tm_Na_adjust / 1000.0, math.e)) \
                     + (9.40 * pow(10, -6) * (pow(math.log(Tm_Na_adjust / 1000.0, math.e), 2)))
    else:
        # magnesium effects are dominant, [equation 16] (Owczarzy et al., 2008) is used
        # Table 2
        a = 3.92 * pow(10, -5)
        b = - 9.11 * pow(10, -6)
        c = 6.26 * pow(10, -5)
        d = 1.42 * pow(10, -5)
        e = - 4.82 * pow(10, -4)
        f = 5.25 * pow(10, -4)
        g = 8.31 * pow(10, -5)
        if R_div_monov_ratio < 6.0:
            a = 3.92 * pow(10, -5) * (
                    0.843 - (0.352 * math.sqrt(Tm_Na_adjust / 1000.0) * math.log(Tm_Na_adjust / 1000.0, math.e)))
            d = 1.42 * pow(10, -5) * (
                    1.279 - 4.03 * pow(10, -3) * math.log(Tm_Na_adjust / 1000.0, math.e) - 8.03 * pow(10, -3) * pow(
                math.log(Tm_Na_adjust / 1000.0, math.e), 2))
            g = 8.31 * pow(10, -5) * (
                    0.486 - 0.258 * math.log(Tm_Na_adjust / 1000.0, math.e) + 5.25 * pow(10, -3) * pow(
                math.log(Tm_Na_adjust / 1000.0, math.e), 3))
        # Eq 16
        correction = a + (b * math.log(free_divalent, math.e))
        + GC_fraction(seq) * (c + (d * math.log(free_divalent, math.e)))
        + (1 / (2 * (len(seq) - 1))) * (e + (f * math.log(free_divalent, math.e))
                                        + g * (pow((math.log(free_divalent, math.e)), 2)))

    if symmetry(seq):
        # Equation A
        Tm = round(1 / ((1 / (delta_H / (delta_S + 1.9872 * math.log(primer_concentration / (1 * pow(10, 9)), math.e))))
                        + correction) - Kelvin, 2)
    else:
        # Equation B
        Tm = round(1 / ((1 / (delta_H / (delta_S + 1.9872 * math.log(primer_concentration / (4 * pow(10, 9)), math.e))))
                        + correction) - Kelvin, 2)
    return Tm


##############################################################################################
def degenerate_seq(primer):
    seq = []
    cs = ""
    for s in primer:
        if s not in degenerate_base:
            cs += s
        else:
            seq.append([cs + i for i in degenerate_base[s]])
            cs = ""
    if cs:
        seq.append([cs])
    return ["".join(i) for i in product(*seq)]


################### hairpin ######################
def hairpin_check(primer, distance):
    n = 0
    distance = distance
    while n <= len(primer) - 5 - 5 - distance:
        kmer = degenerate_seq(primer[n:n + 5])
        left = degenerate_seq(primer[n + 5 + distance:])
        for k in kmer:
            for l in left:
                if re.search(RC(k), l):
                    return True
        n += 1
    return False


################# GC content #####################
def GC_fraction(sequence):
    sequence_expand = degenerate_seq(sequence)
    GC_list = []
    for seq in sequence_expand:
        GC_list.append(round((list(seq).count("G") + list(seq).count("C")) / len(list(seq)), 3))
    GC_average = round(mean(GC_list), 2)
    return GC_average

#####################################################
def get_di_nucleotides(n):
    di_nucleotides = set()
    if n!=0:
        for i in base2bit.keys():
            single = i * n
            di_nucleotides.add(single)
            for j in base2bit.keys():
                if i != j:
                    di = (i + j) * n
                    di_nucleotides.add(di)
                for k in base2bit.keys():
                    if i != j != k:
                        tri = (i + j + k) * (n-1)
                        di_nucleotides.add(tri)
    return di_nucleotides

################## GC Clamp ######################
def GC_clamp(primer, clamp, num=4, length=13):
    for i in range(num, (num + length)):
        s = primer[-i:]
        gc_fraction = GC_fraction(s)
        if gc_fraction > clamp:
            return True
    return False


################# position of degenerate base #####################
def dege_filter_in_term_N_bp(sequence, term):
    if term == 0:
        term_base = ["A"]
    else:
        term_base = sequence[-term:]
    score = score_trans(term_base)
    if score > 1:
        return True
    else:
        return False


# Import multi-alignment results and return a dict ==> {ID：sequence}

def parse_seq(Input, threshold):
    seq_dict = defaultdict(str)
    seq_array = []
    total_sequence_number = 0
    with open(Input, "r") as f:
        for i in f:
            if i.startswith("#"):
                pass
            else:
                if i.startswith(">"):
                    i = i.strip().split(" ")
                    acc_id = i[0]
                    total_sequence_number += 1
                else:
                    # carefully !, make sure that Ns have been replaced!
                    sequence = re.sub("[^ACGTRYMKSWHBVD]", "-", i.strip().upper())
                    seq_dict[acc_id] += sequence
                    seq_array.append(list(sequence))
    if threshold != "F":
        seq_array = pd.DataFrame(seq_array)
        del_index = []
        # total_range = set(range(0, total_sequence_number))
        for col in seq_array.columns.values:
            count_index = seq_array[col][seq_array[col] == "-"].count()
            if round(count_index / total_sequence_number, 2) >= threshold:
                del_index.append(col)
        if len(del_index) > 0:
            # trim = total_range - set(del_index)
            # print(trim)
            # for acc in seq_dict.keys():
            #     trim_seq = ''.join(pd.DataFrame(list(seq_dict[acc])).iloc[:, list(trim)].values().tolist())
            #     seq_dict[acc] = trim_seq
            #     print(trim_seq)
            for acc in seq_dict.keys():
                # print(pd.DataFrame(list(seq_dict[acc])).T)
                trim_seq = ''.join(pd.DataFrame(list(seq_dict[acc])).T.drop(del_index, axis=1).values.tolist()[0])
                seq_dict[acc] = trim_seq
                # print(">{}".format(acc))
                # print(trim_seq)
        else:
            pass
    return seq_dict, total_sequence_number


def current_end_list(primer, adaptor="", num=5, length=14):
    primer_extend = adaptor + primer
    end_seq = []
    for i in range(num, (num + length)):
        s = primer_extend[-i:]
        if s:
            end_seq.extend(degenerate_seq(s))
    return end_seq


def deltaG(sequence):
    Delta_G_list = []
    Na = 50
    for seq in degenerate_seq(sequence):
        Delta_G = 0
        for n in range(len(seq) - 1):
            base_i, base_j = base2bit[seq[n + 1]], base2bit[seq[n]]
            Delta_G += freedom_of_H_37_table[base_i][base_j] * H_bonds_number[base_i][base_j] + \
                       penalty_of_H_37_table[base_i][base_j]
        term5 = sequence[-2:]
        if term5 == "TA":
            Delta_G += adjust_initiation[seq[0]] + adjust_initiation[seq[-1]] + adjust_terminal_TA
        else:
            Delta_G += adjust_initiation[seq[0]] + adjust_initiation[seq[-1]]
        # adjust by concentration of Na+
        Delta_G -= (0.175 * math.log(Na / 1000, math.e) + 0.20) * len(seq)
        if symmetry(seq):
            Delta_G += symmetry_correction
        Delta_G_list.append(Delta_G)
    return round(max(Delta_G_list), 2)


def dimer_check(primer):
    current_end = current_end_list(primer)
    current_end_sort = sorted(current_end, key=lambda i: len(i), reverse=True)
    for end in current_end_sort:
        for p in degenerate_seq(primer):
            idx = p.find(RC(end))
            if idx >= 0:
                end_length = len(end)
                end_GC = end.count("G") + end.count("C")
                end_d1 = 0
                end_d2 = len(p) - len(end) - idx
                Loss = Penalty_points(
                    end_length, end_GC, end_d1, end_d2)
                delta_G = deltaG(end)
                if Loss >= 3 or (delta_G < -5 and (end_d1 == end_d2)):
                    return True
    return False


def nan_removing(pre_list):
    while np.nan in pre_list:
        pre_list.remove(np.nan)
    return pre_list


####################################################
def closest(my_list, my_number1, my_number2):
    index_left = bisect_left(my_list, my_number1)
    # find the first element index in my_list which greater than my_number.
    if my_number2 > my_list[-1]:
        index_right = len(my_list) - 1  # This is index.
    else:
        index_right = bisect_left(my_list, my_number2) - 1
    return index_left, index_right


def degenerate_merge(optimal_primer, degenerate_index):
    optimal_primer_list = list(optimal_primer)
    for i in degenerate_index:
        i = i.split("|")
        if i[1] not in degenerate_base[optimal_primer_list[int(i[0])]]:
            optimal_primer_list[int(i[0])] = trans_score_table[
                round(score_table[optimal_primer_list[int(i[0])]] + score_table[i[1]], 2)]
    return ''.join(optimal_primer_list)


class Dimer(object):

    def __init__(self, primer_file="", outfile="", threshold=3.96, nproc=10):
        self.nproc = nproc
        self.primers_file = primer_file
        self.threshold = threshold
        self.outfile = os.path.abspath(outfile)
        self.primers = self.parse_primers()
        self.primers_list = list(self.primers.keys())
        self.resQ = Manager().Queue()

    def parse_primers(self):
        primer_dict = defaultdict(str)
        with open(self.primers_file, "r") as f:
            for i in f:
                if i.startswith(">"):
                    name = i.strip()
                else:
                    primer_dict[i.strip()] = name
        return primer_dict

    def current_end(self, primer, adaptor="", num=5, length=14):
        primer_extend = adaptor + primer
        end_seq = []
        for i in range(num, (num + length)):
            s = primer_extend[-i:]
            if s:
                end_seq.extend(degenerate_seq(s))
        return end_seq

    def deltaG(self, sequence):
        Delta_G_list = []
        Na = 50
        for seq in degenerate_seq(sequence):
            Delta_G = 0
            for n in range(len(seq) - 1):
                i, j = base2bit[seq[n + 1]], base2bit[seq[n]]
                Delta_G += freedom_of_H_37_table[i][j] * H_bonds_number[i][j] + penalty_of_H_37_table[i][j]
            term5 = sequence[-2:]
            if term5 == "TA":
                Delta_G += adjust_initiation[seq[0]] + adjust_initiation[seq[-1]] + adjust_terminal_TA
            else:
                Delta_G += adjust_initiation[seq[0]] + adjust_initiation[seq[-1]]
            # adjust by concentration of Na+
            Delta_G -= (0.175 * math.log(Na / 1000, math.e) + 0.20) * len(seq)
            if symmetry(seq):
                Delta_G += symmetry_correction
            Delta_G_list.append(Delta_G)
        return round(max(Delta_G_list), 2)

    def dimer_check(self, position):
        current_end_set = self.current_end(self.primers_list[position])
        current_end_list = sorted(list(current_end_set), key=lambda i: len(i), reverse=True)
        dimer = False
        for ps in self.primers_list[position:]:
            for end in current_end_list:
                for p in degenerate_seq(ps):
                    idx = p.find(RC(end))
                    if idx >= 0:
                        end_length = len(end)
                        end_GC = end.count("G") + end.count("C")
                        end_d1 = 0
                        end_d2 = len(p) - len(end) - idx
                        Loss = Penalty_points(
                            end_length, end_GC, end_d1, end_d2)
                        delta_G = self.deltaG(end)
                        if Loss >= self.threshold or (delta_G < -5 and (end_d1 == end_d2)):
                            line = (self.primers[self.primers_list[position]], self.primers_list[position],
                                    end, delta_G, end_length, end_d1,
                                    end_GC, self.primers[ps],
                                    ps, end_d2, Loss
                                    )
                            self.resQ.put(line)
                            # The put method also has two optional parameters: blocked and timeout. If blocked is
                            # true (the default value) and timeout is positive, the method will block the time
                            # specified by timeout until there is space left in the queue. If the timeout occurs,
                            # a Queue.Full exception will be thrown. If blocked is false, but the queue is full,
                            # a Queue.Full exception will be thrown immediately.
                            dimer = True
                            break
                if dimer:
                    dimer = False
                    break
        self.resQ.put(None)  # Found a None, you can rest now

    #  The queue in multiprocessing cannot be used for pool process pool, but there is a manager in multiprocessing.
    #  Inter process communication in the pool uses the queue in the manager. Manager().Queue().
    #  Queue. qsize(): returns the number of messages contained in the current queue;
    #  Queue. Empty(): returns True if the queue is empty, otherwise False;
    #  Queue. full(): returns True if the queue is full, otherwise False;
    #  Queue. get(): get a message in the queue, and then remove it from the queue,
    #                which can pass the parameter timeout.
    #  Queue.get_Nowait(): equivalent to Queue. get (False).
    #                If the value cannot be obtained, an exception will be triggered: Empty;
    #  Queue. put(): add a value to the data sequence to transfer the parameter timeout duration.
    #  Queue.put_Nowait(): equivalent to Queue. get (False). When the queue is full, an error is reported: Full.

    #  Realize interprocess communication through pipe, and the performance of pipe is higher than that of queue.
    #  Pipe can only be used for communication between two processes.

    def run(self):
        p = ProcessPoolExecutor(self.nproc)
        for position in range(len(self.primers_list)):
            p.submit(self.dimer_check, position)
            #  This will submit all tasks to one place without blocking, and then each
            #  thread in the thread pool will fetch tasks.
        n = 0
        primer_id_sum = defaultdict(int)
        dimer_primer_id_sum = defaultdict(int)
        with open(self.outfile, "w") as fo:
            headers = ["Primer_ID", "Primer seq", "Primer end", "Delta G", "Primer end length", "End (distance 1)",
                       "End (GC)", "Dimer-primer_ID", "Dimer-primer seq", "End (distance 2)", "Loss"]
            fo.write("\t".join(headers) + "\n")
            while n < len(self.primers):
                res = self.resQ.get()
                # The get method can read and delete an element from the queue. Similarly, the get method has two
                # optional parameters: blocked and timeout. If blocked is true (the default value) and timeout is
                # positive, no element is retrieved during the waiting time, and a Queue is thrown Empty exception.
                # If blocked is false, there are two cases. If a value of Queue is available, return the value
                # immediately. Otherwise, if the queue is empty, throw a Queue.Empty exception immediately.
                if res is None:
                    n += 1
                    continue
                primer_id_sum[res[0]] += 1
                dimer_primer_id_sum[res[7]] += 1
                fo.write("\t".join(map(str, res)) + "\n")
            #  get results before shutdown. Synchronous call mode: call, wait for the return value, decouple, but slow.
        p.shutdown()
        # After I run the main, I don't care whether the sub thread is alive or dead. With this parameter, after all
        # the sub threads are executed, the main function is executed.
        # get results after shutdown. Asynchronous call mode: only call, unequal return values, coupling may exist,
        # but the speed is fast.

        with open(self.outfile + ".dimer_num", "w") as fo:
            fo.write("SeqName\tPrimer_ID\tDimer-primer_ID\tRowSum\n")
            for k in primer_id_sum.keys():
                p_id = primer_id_sum[k]
                d_id = dimer_primer_id_sum[k]
                RowSum = p_id + d_id
                fo.write("\t".join(map(str, [k, p_id, d_id, RowSum])) + "\n")


class NN_degenerate(object):
    def __init__(self, Seq_dict, Total_sequence_number, primer_length=18, coverage=0.8, number_of_dege_bases=18,
                 score_of_dege_bases=1000, method="multiPrime1", di_nucl=0,
                 product_len="200,500", position="2,-1", variation=2, raw_entropy_threshold=3.6, distance=4,
                 GC="0.4,0.6", nproc=10, outfile=""):
        self.primer_length = primer_length  # primer length
        self.coverage = coverage  # min coverage
        self.number_of_dege_bases = number_of_dege_bases
        self.score_of_dege_bases = score_of_dege_bases
        self.product = product_len
        self.position = position  # gap position
        self.Y_strict, self.Y_strict_R = self.get_Y()
        self.variation = variation  # coverage of n-nt variation and max_gap_number
        self.distance = distance  # haripin
        self.GC = GC.split(",")
        self.di_nucl=di_nucl
        self.nproc = nproc  # GC content
        self.seq_dict, self.total_sequence_number = Seq_dict, Total_sequence_number
        self.position_list = self.seq_attribute(self.seq_dict)
        self.start_position = self.position_list[0]
        self.stop_position = self.position_list[1]
        self.method = method
        self.length = self.position_list[2]
        self.raw_entropy_threshold = raw_entropy_threshold
        self.entropy_threshold = self.entropy_threshold_adjust(self.length)
        self.outfile = outfile
        self.resQ = Manager().Queue()

    ####################################################################
    ##### pre-filter by GC content / di-nucleotide / hairpin ###########
    def primer_pre_filter(self, primer):
        information = []
        min_GC, max_GC = self.GC
        primer_GC_content = GC_fraction(primer)
        if not float(min_GC) <= primer_GC_content <= float(max_GC):
            information.append("GC_out_of_range (" + str(primer_GC_content) + ")")
        if self.di_nucleotide(primer):
            information.append("di_nucleotide")
        if hairpin_check(primer, self.distance):
            information.append("hairpin")

        if len(information) == 0:
            return primer_GC_content
        else:
            return '|'.join(information)
    ################# di_nucleotide #####################
    def di_nucleotide(self, primer):
        di_nucleotides = get_di_nucleotides(self.di_nucl)
        primers = degenerate_seq(primer)
        for m in primers:
            for n in di_nucleotides:
                if re.search(n, m):
                    return True
        return False

    ####################################################################
    # if full degenerate primer is ok, we don't need to continue NN-array
    def pre_degenerate_primer_check(self, primer):
        primer_degeneracy = score_trans(primer)
        primer_dege_number = dege_number(primer)
        if primer_degeneracy < self.score_of_dege_bases and primer_dege_number < self.number_of_dege_bases:
            return True
        else:
            return False

    def full_degenerate_primer(self, freq_matrix):
        # degenerate transformation in each position
        max_dege_primers = ''
        for col in freq_matrix.columns.values:
            tmp = freq_matrix[freq_matrix[col] > 0].index.values.tolist()
            max_dege_primers += trans_score_table[round(sum([score_table[x] for x in tmp]), 2)]
        return max_dege_primers

    def state_matrix(self, primers_db):
        pieces = []
        for col in primers_db.columns.values:
            tmp_series = primers_db[col].value_counts()  # (normalize=True)
            tmp_series.name = col
            pieces.append(tmp_series)
        nodes = pd.concat(pieces, axis=1)
        nodes.fillna(0, inplace=True)
        nodes = nodes.sort_index(ascending=True)
        nodes = nodes.astype(int)
        row_names = nodes.index.values.tolist()
        if "-" in row_names:
            nodes.drop("-", inplace=True, axis=0)
        return nodes

    def di_matrix(self, primers):
        primers_trans = []
        for i in primers.keys():
            slice = []
            for j in range(len(i) - 1):
                slice.append(i[j:j + 2])
            primers_trans.extend(repeat(slice, primers[i]))
        return pd.DataFrame(primers_trans)

    def trans_matrix(self, primers):
        primers_di_db = self.di_matrix(primers)
        pieces = []
        for col in primers_di_db.columns.values:
            tmp_list = []
            for i in di_bases:
                # row: A, T, C ,G; column: A, T, C, G
                number = list(primers_di_db[col]).count(i)
                tmp_list.append(number)
            pieces.append(tmp_list)
        a, b = primers_di_db.shape
        trans = np.array(pieces).reshape(b, 4, 4)
        return trans

    def get_optimal_primer_by_viterbi(self, nodes, trans):
        nodes = np.array(nodes.T)
        seq_len, num_labels = len(nodes), len(trans[0])
        labels = np.arange(num_labels).reshape((1, -1))
        scores = nodes[0].reshape((-1, 1))
        primer_index = labels
        for t in range(1, seq_len):
            observe = nodes[t].reshape((1, -1))
            current_trans = trans[t - 1]
            M = scores + current_trans + observe
            scores = np.max(M, axis=0).reshape((-1, 1))
            idxs = np.argmax(M, axis=0)
            primer_index = np.concatenate([primer_index[:, idxs], labels], 0)
        best_primer_index = primer_index[:, scores.argmax()]
        return best_primer_index

    def get_optimal_primer_by_MM(self, cover_for_MM):
        sort_cover = sorted(cover_for_MM.items(), key=lambda x: x[1], reverse=True)
        L_seq = list(sort_cover[0][0])
        best_primer_index = [base2bit[x] for x in L_seq]
        # Return the maximum of an array or maximum along an axis. axis=0 代表列 , axis=1 代表行
        return best_primer_index

    def entropy(self, cover, cover_number, gap_sequence, gap_sequence_number):
        # cBit: entropy of cover sequences
        # tBit: entropy of total sequences
        cBit = 0
        tBit = 0
        for c in cover.keys():
            cBit += (cover[c] / cover_number) * math.log((cover[c] / cover_number), 2)
            tBit += (cover[c] / (cover_number + gap_sequence_number)) * \
                    math.log((cover[c] / (cover_number + gap_sequence_number)), 2)
        for t in gap_sequence.keys():
            tBit += (gap_sequence[t] / (cover_number + gap_sequence_number)) * \
                    math.log((gap_sequence[t] / (cover_number + gap_sequence_number)), 2)
        return round(-cBit, 2), round(-tBit, 2)

    # Sequence processing. Return a list contains sequence length, start and stop position of each sequence.
    def seq_attribute(self, Input_dict):
        start_dict = {}
        stop_dict = {}
        # pattern_start = re.compile('[A-Z]')
        # pattern_stop = re.compile("-*$")
        for acc_id in Input_dict.keys():
            # start_dict[acc_id] = pattern_start.search(Input_dict[acc_id]).span()[0]
            # stop_dict[acc_id] = pattern_stop.search(Input_dict[acc_id]).span()[0] - 1
            t_length = len(Input_dict[acc_id])
            start_dict[acc_id] = t_length - len(Input_dict[acc_id].lstrip("-"))
            stop_dict[acc_id] = len(Input_dict[acc_id].rstrip("-"))
            # start position should contain [coverage] sequences at least.
        # start = np.quantile(np.array(list(start_dict.values())).reshape(1, -1), self.coverage, interpolation="higher")
        # for python 3.9.9
        start = np.quantile(np.array(list(start_dict.values())).reshape(1, -1), self.coverage, method="higher")
        # stop position should contain [coverage] sequences at least.
        # stop = np.quantile(np.array(list(stop_dict.values())).reshape(1, -1), self.coverage, interpolation="lower")
        stop = np.quantile(np.array(list(stop_dict.values())).reshape(1, -1), self.coverage, method="lower")
        min_len = self.product.split(",")
        if stop - start < int(self.product[0]):
            print("Error: max length of PCR product is shorter than the default min Product length with {} "
                  "coverage! Non candidate primers !!!".format(self.coverage))
            sys.exit(1)
        else:
            return [start, stop, stop - start]

    def entropy_threshold_adjust(self, length):
        if length < 5000:
            return self.raw_entropy_threshold
        else:
            if length < 10000:
                return self.raw_entropy_threshold * 0.95
            else:
                return self.raw_entropy_threshold * 0.9

    def get_primers(self, sequence_dict, primer_start):  # , primer_info, non_cov_primer_out
        # record sequence and acc id
        non_gap_seq_id = defaultdict(list)
        # record sequence (no gap) and number
        cover = defaultdict(int)
        cover_for_MM = defaultdict(int)
        # record total coverage sequence number
        cover_number = 0
        # record sequence (> variation gap) and number
        gap_sequence = defaultdict(int)
        gap_seq_id = defaultdict(list)
        # record total sequence (> variation gap) number
        gap_sequence_number = 0
        primers_db = []
        for seq_id in sequence_dict.keys():
            sequence = sequence_dict[seq_id][primer_start:primer_start + self.primer_length].upper()
            # replace "-" which in start or stop position with nucleotides
            if sequence == "-" * self.primer_length:
                pass
            else:
                if sequence.startswith("-"):
                    sequence_narrow = sequence.lstrip("-")
                    append_base_length = len(sequence) - len(sequence_narrow)
                    left_seq = sequence_dict[seq_id][0:primer_start].replace("-", "")
                    if len(left_seq) >= append_base_length:
                        sequence = left_seq[len(left_seq) - append_base_length:] + sequence_narrow
                if sequence.endswith("-"):
                    sequence_narrow = sequence.rstrip("-")
                    append_base_length = len(sequence) - len(sequence_narrow)
                    right_seq = sequence_dict[seq_id][primer_start + self.primer_length:].replace("-", "")
                    if len(right_seq) >= append_base_length:
                        sequence = sequence_narrow + right_seq[0:append_base_length]
            if len(sequence) < self.primer_length:
                append_base_length = self.primer_length - len(sequence)
                left_seq = sequence_dict[seq_id][0:primer_start].replace("-", "")
                if len(left_seq) >= append_base_length:
                    sequence = left_seq[len(left_seq) - append_base_length:] + sequence
            # gap number. number of gap > 2
            if list(sequence).count("-") > self.variation:
                gap_sequence[sequence] += 1
                gap_sequence_number += 1
                if round(gap_sequence_number / self.total_sequence_number, 2) >= (1 - self.coverage):
                    break
                else:
                    # record acc ID of gap sequences
                    expand_sequence = degenerate_seq(sequence)
                    for i in expand_sequence:
                        gap_seq_id[i].append(seq_id)
            # # accepted gap, number of gap <= variation
            else:
                expand_sequence = degenerate_seq(sequence)
                cover_number += 1
                for i in expand_sequence:
                    cover[i] += 1
                    primers_db.append(list(i))
                    # record acc ID of non gap sequences, which is potential mis-coverage
                    non_gap_seq_id[i].append(seq_id)
                    if re.search("-", i):
                        pass
                    else:
                        cover_for_MM[i] += 1
        # number of sequences with too many gaps greater than (1 - self.coverage)
        if round(gap_sequence_number / self.total_sequence_number, 2) >= (1 - self.coverage):
            # print("Gap fail")
            self.resQ.put(None)
        elif len(cover) < 1:
            self.resQ.put(None)
            # print("Cover fail")
        else:
            # cBit: entropy of cover sequences
            # tBit: entropy of total sequences
            cBit, tBit = self.entropy(cover, cover_number, gap_sequence, gap_sequence_number)
            if tBit > self.entropy_threshold:
                # print("Entropy fail")
                # This window is not a conserved region, and not proper to design primers
                self.resQ.put(None)
            else:
                primers_db = pd.DataFrame(primers_db)
                # frequency matrix
                freq_matrix = self.state_matrix(primers_db)
                # print(freq_matrix)
                colSum = np.sum(freq_matrix, axis=0)
                a, b = freq_matrix.shape
                # a < 4 means base composition of this region is less than 4 (GC bias).
                # It's not a proper region for primer design.
                if a < 4:
                    self.resQ.put(None)
                elif (colSum == 0).any():
                    # print(colSum)  # if 0 in array; pass
                    self.resQ.put(None)
                else:
                    gap_seq_id_info = [primer_start, gap_seq_id]
                    if self.method == "multiPrime1":
                        mismatch_coverage, non_cov_primer_info = \
                            self.degenerate_by_NN_algorithm(primer_start, freq_matrix, cover, non_gap_seq_id,
                                                            cover_for_MM, cover_number, cBit, tBit)
                        # self.resQ.put([mismatch_coverage, non_cov_primer_info, gap_seq_id_info])
                        # F, R = mismatch_coverage[1][6], mismatch_coverage[1][7]
                        sequence = mismatch_coverage[1][2]
                        if dimer_check(sequence):
                            # print("Dimer fail")
                            self.resQ.put(None)
                        else:
                            self.resQ.put([mismatch_coverage, non_cov_primer_info, gap_seq_id_info])
                        # if F < cover_number * 0.5 or R < cover_number * 0.5:
                        #     self.resQ.put(None)
                        # else:
                        #     self.resQ.put([mismatch_coverage, non_cov_primer_info, gap_seq_id_info])
                    elif self.method == "multiPrime2":
                        mismatch_coverage, non_cov_primer_info = \
                            self.refine_by_multiPrime2(primer_start, freq_matrix, cover, non_gap_seq_id, cover_for_MM,
                                                       cover_number, cBit, tBit)
                        sequence = mismatch_coverage[1][2]
                        if dimer_check(sequence):
                            # print("Dimer fail")
                            self.resQ.put(None)
                        else:
                            self.resQ.put([mismatch_coverage, non_cov_primer_info, gap_seq_id_info])

    def degenerate_by_NN_algorithm(self, primer_start, freq_matrix, cover, non_gap_seq_id, cover_for_MM,
                                   cover_number, cBit, tBit):
        # full degenerate primer
        # full_degenerate_primer = self.full_degenerate_primer(freq_matrix)
        # unique covered primers, which is used to calculate coverage and
        # mis-coverage in the following step.
        cover_primer_set = set(cover.keys())
        # if full_degenerate_primer is ok, then return full_degenerate_primer
        # mismatch_coverage, non_cov_primer_info = {}, {}
        F_non_cover, R_non_cover = {}, {}
        ######################################################################################################
        ############ need prone. not all primers is proper for primer-F or primer-R ##########################
        # If a primer is located in the start region, there is no need to calculate its coverage for primer-R#
        ## here is a suggestion. we can assert candidate primer as primer-F or primer-R by primer attribute ##
        ######################################################################################################
        NN_matrix = self.trans_matrix(cover)
        if len(cover_for_MM) != 0:
            optimal_primer_index_NM = self.get_optimal_primer_by_viterbi(freq_matrix, NN_matrix)
            optimal_primer_index_MM = self.get_optimal_primer_by_MM(cover_for_MM)
            # print(optimal_primer_index_NM.tolist()) # array
            # print(optimal_primer_index_MM) # list
            #  if (optimal_primer_index_NM == optimal_primer_index_MM).all():
            if optimal_primer_index_NM.tolist() == optimal_primer_index_MM:
                optimal_primer_index = optimal_primer_index_NM
                row_names = np.array(freq_matrix.index.values).reshape(1, -1)
                # build a list to store init base information in each position.
                optimal_primer_list = row_names[:, optimal_primer_index][0].tolist()
                # initiation coverage (optimal primer, used as base coverage)
                optimal_coverage_init = cover["".join(optimal_primer_list)]
                optimal_primer_current, F_mis_cover, R_mis_cover, information, F_non_cover, R_non_cover = \
                    self.coverage_stast(cover, optimal_primer_index, NN_matrix, optimal_coverage_init, cover_number,
                                        optimal_primer_list, cover_primer_set, non_gap_seq_id, F_non_cover,
                                        R_non_cover)
                # print(F_mis_cover)
                # print(R_mis_cover)
            else:
                F_non_cover_NM, R_non_cover_NM, F_non_cover_MM, R_non_cover_MM = {}, {}, {}, {}
                row_names = np.array(freq_matrix.index.values).reshape(1, -1)
                # build a list to store init base information in each position.
                optimal_primer_list_NM = row_names[:, optimal_primer_index_NM][0].tolist()
                # initiation coverage (optimal primer, used as base coverage)
                optimal_coverage_init_NM = cover["".join(optimal_primer_list_NM)]
                NN_matrix_NM = NN_matrix.copy()
                optimal_primer_current_NM, F_mis_cover_NM, R_mis_cover_NM, information_NM, F_non_cover_NM, \
                R_non_cover_NM = self.coverage_stast(cover, optimal_primer_index_NM, NN_matrix_NM,
                                                     optimal_coverage_init_NM, cover_number, optimal_primer_list_NM,
                                                     cover_primer_set, non_gap_seq_id, F_non_cover_NM,
                                                     R_non_cover_NM)
                optimal_primer_list_MM = row_names[:, optimal_primer_index_MM][0].tolist()
                # initiation coverage (optimal primer, used as base coverage)
                optimal_coverage_init_MM = cover["".join(optimal_primer_list_MM)]
                NN_matrix_MM = NN_matrix.copy()
                optimal_primer_current_MM, F_mis_cover_MM, R_mis_cover_MM, information_MM, F_non_cover_MM, \
                R_non_cover_MM = self.coverage_stast(cover, optimal_primer_index_MM, NN_matrix_MM,
                                                     optimal_coverage_init_MM, cover_number,
                                                     optimal_primer_list_MM, cover_primer_set, non_gap_seq_id,
                                                     F_non_cover_MM, R_non_cover_MM)
                if (F_mis_cover_NM + R_mis_cover_NM) > (F_mis_cover_MM + R_mis_cover_MM):
                    optimal_primer_current, F_mis_cover, R_mis_cover, information, optimal_coverage_init, \
                    F_non_cover, R_non_cover, NN_matrix = optimal_primer_current_NM, F_mis_cover_NM, \
                                                          R_mis_cover_NM, information_NM, optimal_coverage_init_NM, \
                                                          F_non_cover_NM, R_non_cover_NM, NN_matrix_NM
                else:
                    optimal_primer_current, F_mis_cover, R_mis_cover, information, optimal_coverage_init, \
                    F_non_cover, R_non_cover, NN_matrix = optimal_primer_current_MM, F_mis_cover_MM, \
                                                          R_mis_cover_MM, information_MM, optimal_coverage_init_MM, \
                                                          F_non_cover_MM, R_non_cover_MM, NN_matrix_MM
                # print(F_mis_cover)
                # print(R_mis_cover)
        else:
            optimal_primer_index_NM = self.get_optimal_primer_by_viterbi(freq_matrix, NN_matrix)
            F_non_cover_NM, R_non_cover_NM, F_non_cover_MM, R_non_cover_MM = {}, {}, {}, {}
            row_names = np.array(freq_matrix.index.values).reshape(1, -1)
            # build a list to store init base information in each position.
            optimal_primer_list_NM = row_names[:, optimal_primer_index_NM][0].tolist()
            # initiation coverage (optimal primer, used as base coverage)
            optimal_coverage_init_NM = cover["".join(optimal_primer_list_NM)]
            NN_matrix_NM = NN_matrix.copy()
            optimal_primer_current_NM, F_mis_cover_NM, R_mis_cover_NM, information_NM, F_non_cover_NM, \
            R_non_cover_NM = self.coverage_stast(cover, optimal_primer_index_NM, NN_matrix_NM,
                                                 optimal_coverage_init_NM, cover_number, optimal_primer_list_NM,
                                                 cover_primer_set, non_gap_seq_id, F_non_cover_NM, R_non_cover_NM)
            optimal_primer_current, F_mis_cover, R_mis_cover, information, optimal_coverage_init, F_non_cover, \
            R_non_cover, NN_matrix = optimal_primer_current_NM, F_mis_cover_NM, R_mis_cover_NM, information_NM, \
                                     optimal_coverage_init_NM, F_non_cover_NM, R_non_cover_NM, NN_matrix_NM
            # print(F_mis_cover)
            # print(R_mis_cover)
        nonsense_primer_number = len(set(degenerate_seq(optimal_primer_current)) - set(cover.keys()))
        primer_degenerate_number = dege_number(optimal_primer_current)
        Tm, coverage = [], []
        for seq in degenerate_seq(optimal_primer_current):
            Tm.append(Calc_Tm_v2(seq))
            coverage.append(cover[seq])
        Tm_average = round(mean(Tm), 2)
        perfect_coverage = sum(coverage)
        degeneracy = score_trans(optimal_primer_current)
        out_mismatch_coverage = [primer_start,
                                 [cBit, tBit, optimal_primer_current, primer_degenerate_number, degeneracy,
                                  nonsense_primer_number, perfect_coverage, F_mis_cover,
                                  R_mis_cover, Tm_average, information]]
        non_cov_primer_info = [primer_start, [F_non_cover, R_non_cover]]
        return out_mismatch_coverage, non_cov_primer_info

    def coverage_stast(self, cover, optimal_primer_index, NN_matrix, optimal_coverage_init, cover_number,
                       optimal_primer_list, cover_primer_set, non_gap_seq_id, F_non_cover, R_non_cover):
        # if the coverage is too low, is it necessary to refine?
        # mis-coverage as threshold? if mis-coverage reached to 100% but degeneracy is still very low,
        optimal_NN_index = []
        optimal_NN_coverage = []
        for idx in range(len(optimal_primer_index) - 1):
            # NN index
            optimal_NN_index.append([optimal_primer_index[idx], optimal_primer_index[idx + 1]])
            # NN coverage
            # Is the minimum number in NN coverage = optimal_primer_coverage ? No!
            optimal_NN_coverage.append(
                NN_matrix[idx, optimal_primer_index[idx], optimal_primer_index[idx + 1]])
        # mis-coverage initialization
        F_mis_cover_cover, F_non_cover_in_cover, R_mis_cover_cover, R_non_cover_in_cover = \
            self.mis_primer_check(cover_primer_set, ''.join(optimal_primer_list), cover,
                                  non_gap_seq_id)
        # print(optimal_coverage_init + F_mis_cover_cover)
        # print(optimal_coverage_init + R_mis_cover_cover)
        # print(cover_number)
        # print(optimal_primer_list)
        if optimal_coverage_init + F_mis_cover_cover < cover_number or \
                optimal_coverage_init + R_mis_cover_cover < cover_number:
            while optimal_coverage_init + F_mis_cover_cover < cover_number or \
                    optimal_coverage_init + R_mis_cover_cover < cover_number:
                # optimal_primer_update, coverage_update, NN_coverage_update,
                # NN array_update, degeneracy_update, degenerate_update
                optimal_primer_list, optimal_coverage_init, optimal_NN_coverage_update, \
                NN_matrix, degeneracy, number_of_degenerate = \
                    self.refine_by_NN_array(optimal_primer_list, optimal_coverage_init, cover, optimal_NN_index,
                                            optimal_NN_coverage, NN_matrix)
                F_mis_cover_cover, F_non_cover_in_cover, R_mis_cover_cover, R_non_cover_in_cover = \
                    self.mis_primer_check(cover_primer_set, ''.join(optimal_primer_list), cover,
                                          non_gap_seq_id)
                # If there is no increase in NN_coverage,
                # it suggests the presence of bugs or a mismatch in continuous positions.
                # Is this step necessary? or shall we use DegePrime method? or shall we use machine learning?
                if max(F_mis_cover_cover, R_mis_cover_cover) == cover_number:
                    break
                elif optimal_NN_coverage_update == optimal_NN_coverage:
                    break
                # If the degeneracy exceeds the threshold, the loop will break.
                elif 2 * degeneracy > self.score_of_dege_bases or 3 * degeneracy / 2 > self.score_of_dege_bases \
                        or number_of_degenerate == self.number_of_dege_bases:
                    break
                else:
                    optimal_NN_coverage = optimal_NN_coverage_update
        # If the primer coverage does not increase after degeneration,
        # the process will backtrack and assess the original optimal primer.
        # print(optimal_primer_list)
        optimal_primer_current = ''.join(optimal_primer_list)
        # print(optimal_primer_current)
        information = self.primer_pre_filter(optimal_primer_current)
        # F_mis_cover_cover, F_non_cover_in_cover, R_mis_cover_cover, R_non_cover_in_cover = \
        #     self.mis_primer_check(cover_primer_set, optimal_primer_current, cover,
        #                           non_gap_seq_id)
        F_non_cover.update(F_non_cover_in_cover)
        R_non_cover.update(R_non_cover_in_cover)
        F_mis_cover = optimal_coverage_init + F_mis_cover_cover
        R_mis_cover = optimal_coverage_init + R_mis_cover_cover
        # print(F_mis_cover)
        return optimal_primer_current, F_mis_cover, R_mis_cover, information, F_non_cover, R_non_cover

    def refine_by_NN_array(self, optimal_primer_list, optimal_coverage_init, cover,
                           optimal_NN_index, optimal_NN_coverage, NN_array):
        # use minimum index of optimal_NN_coverage as the position to refine
        refine_index = np.where(optimal_NN_coverage == np.min(optimal_NN_coverage))[0]  # np.where[0] is a list
        # build dict to record coverage and NN array
        primer_update_list, coverage_update_list, NN_array_update_list, NN_coverage_update = [], [], [], []
        for i in refine_index:
            optimal_NN_coverage_tmp = optimal_NN_coverage.copy()
            NN_array_tmp = NN_array.copy()
            optimal_list = optimal_primer_list.copy()
            # initiation score
            # initiation coverage
            coverage_renew = optimal_coverage_init
            if i == 0:
                # two position need refine
                # position 0 and 1
                # decide which position to choose
                row = optimal_NN_index[i][0]
                column = optimal_NN_index[i][1]
                if len(np.where(NN_array_tmp[0, :, column] > 0)[0]) > 1:
                    init_score = score_table[optimal_list[i]]
                    refine_column = NN_array_tmp[i, :, column]
                    refine_row_arg_sort = np.argsort(refine_column, axis=0)[::-1]
                    new_primer = optimal_list
                    # print(row, refine_column)
                    for idx in refine_row_arg_sort:
                        # init refine,  We must ensure that there are no double counting.
                        # position 0.
                        if idx != row:
                            init_score += score_table[bases[idx]]
                            new_primer[i] = bases[idx]
                            # Calculate coverage after refine
                            for new_primer_update in degenerate_seq("".join(new_primer)):
                                if new_primer_update in cover.keys():
                                    coverage_renew += cover["".join(new_primer_update)]
                            new_primer[i] = trans_score_table[round(init_score, 2)]
                            # reset NN_array. row names will update after reset.
                            NN_array_tmp[i, row, :] += NN_array_tmp[i, idx, :]
                            NN_array_tmp[i, idx, :] -= NN_array_tmp[i, idx, :]
                            optimal_NN_coverage_tmp[i] = NN_array_tmp[i, row, column]
                            break
                        # primer update
                    optimal_list_update = optimal_list
                    optimal_list_update[i] = trans_score_table[round(init_score, 2)]
                # position 1
                elif len(np.where(NN_array_tmp[0, row, :] > 0)[0]) > 1:
                    init_score = score_table[optimal_list[i + 1]]
                    next_row = optimal_NN_index[i + 1][0]
                    next_column = optimal_NN_index[i + 1][1]
                    # concat row of layer i and column of layer i+1
                    refine_row = NN_array_tmp[i, row, :].reshape(1, -1)
                    refine_column = NN_array_tmp[i + 1, :, next_column].reshape(1, -1)
                    refine = np.concatenate([refine_row, refine_column], 0)
                    refine_min = np.min(refine, axis=0)
                    refine_row_arg_sort = np.argsort(refine_min, axis=0)[::-1]
                    # Return the minimum of an array or maximum along an axis. axis=0: column , axis=1: row
                    new_primer = optimal_list
                    if len(np.where(refine_min > 0)[0]) > 1:
                        for idx in refine_row_arg_sort:
                            # We must ensure that there are no double counting.
                            # position 1.
                            if idx != column:
                                init_score += score_table[bases[idx]]
                                # Calculate coverage after refine
                                new_primer[i + 1] = bases[idx]
                                for new_primer_update in degenerate_seq("".join(new_primer)):
                                    if new_primer_update in cover.keys():
                                        coverage_renew += cover["".join(new_primer_update)]
                                new_primer[i + 1] = trans_score_table[round(init_score, 2)]
                                # reset NN_array. column + (column idx) of layer i and row + (row idx) of layer i+1.
                                NN_array_tmp[i, :, column] += NN_array_tmp[i, :, idx]
                                NN_array_tmp[i, :, idx] -= NN_array_tmp[i, :, idx]
                                NN_array_tmp[i + 1, next_row, :] += NN_array_tmp[i + 1, idx, :]
                                NN_array_tmp[i + 1, idx, :] -= NN_array_tmp[i + 1, idx, :]
                                optimal_NN_coverage_tmp[i] = NN_array_tmp[i, row, column]
                                optimal_NN_coverage_tmp[i + 1] = NN_array_tmp[i + 1, next_row, next_column]
                                break
                    # primer update
                    optimal_list_update = optimal_list
                    optimal_list_update[i + 1] = trans_score_table[round(init_score, 2)]
                else:
                    optimal_list_update = optimal_list
            elif i == len(optimal_NN_index) - 1:
                init_score = score_table[optimal_list[i + 1]]
                row = optimal_NN_index[i][0]
                column = optimal_NN_index[i][1]
                refine_row = NN_array_tmp[i, row, :]
                refine_row_arg_sort = np.argsort(refine_row, axis=0)[::-1]
                # If number of refine_row > 1, then the current position need to refine.
                if len(np.where(refine_row > 0)[0]) > 1:
                    new_primer = optimal_list
                    for idx in refine_row_arg_sort:
                        # We must ensure that there are no double counting.
                        # position -1.
                        if idx != column:
                            init_score += score_table[bases[idx]]
                            # Calculate coverage after refine
                            new_primer[i + 1] = bases[idx]
                            for new_primer_update in degenerate_seq("".join(new_primer)):
                                if new_primer_update in cover.keys():
                                    coverage_renew += cover["".join(new_primer_update)]
                            new_primer[i + 1] = trans_score_table[round(init_score, 2)]
                            # reset NN_array. column names will update after reset.
                            NN_array_tmp[i, :, column] += NN_array_tmp[i, :, idx]
                            NN_array_tmp[i, :, idx] -= NN_array_tmp[i, :, idx]
                            optimal_NN_coverage_tmp[i] = NN_array_tmp[i, row, column]
                            break
                # primer update
                optimal_list_update = optimal_list
                optimal_list_update[i + 1] = trans_score_table[round(init_score, 2)]
            else:
                init_score = score_table[optimal_list[i + 1]]
                row = optimal_NN_index[i][0]
                column = optimal_NN_index[i][1]
                next_row = optimal_NN_index[i + 1][0]
                next_column = optimal_NN_index[i + 1][1]
                # concat row of layer i and column of layer i+1
                refine_row = NN_array_tmp[i, row, :].reshape(1, -1)
                refine_column = NN_array_tmp[i + 1, :, next_column].reshape(1, -1)
                refine = np.concatenate([refine_row, refine_column], 0)
                refine_min = np.min(refine, axis=0)
                # Return the minimum of an array or maximum along an axis. axis=0: column , axis=1: row
                refine_min_arg_sort = np.argsort(refine_min, axis=0)[::-1]
                if len(np.where(refine_min > 0)[0]) > 1:
                    new_primer = optimal_list
                    # for idx in np.where(refine_min_sort > 0)[0]:
                    for idx in refine_min_arg_sort:
                        # We must ensure that there are no double counting.
                        # position i+1.
                        if idx != column:
                            # or if idx != next_row
                            # init trans score update
                            init_score += score_table[bases[idx]]
                            # Calculate coverage after refine
                            new_primer[i + 1] = bases[idx]
                            for new_primer_update in degenerate_seq("".join(new_primer)):
                                if new_primer_update in cover.keys():
                                    coverage_renew += cover["".join(new_primer_update)]
                            new_primer[i + 1] = trans_score_table[round(init_score, 2)]
                            # reset NN_array. column + (column idx) of layer i and row + (row idx) of layer i+1.
                            NN_array_tmp[i, :, column] += NN_array_tmp[i, :, idx]
                            NN_array_tmp[i, :, idx] -= NN_array_tmp[i, :, idx]
                            NN_array_tmp[i + 1, next_row, :] += NN_array_tmp[i + 1, idx, :]
                            NN_array_tmp[i + 1, idx, :] -= NN_array_tmp[i + 1, idx, :]
                            optimal_NN_coverage_tmp[i] = NN_array_tmp[i, row, column]
                            optimal_NN_coverage_tmp[i + 1] = NN_array_tmp[i + 1, next_row, next_column]

                            break
                # primer update
                optimal_list_update = optimal_list
                optimal_list_update[i + 1] = trans_score_table[round(init_score, 2)]
            # primer_update = "".join(primer_list_update)
            primer_update_list.append(optimal_list_update)
            NN_coverage_update.append(optimal_NN_coverage_tmp)
            # current_primers_set = set(degenerate_seq(primer_update))
            # coverage of update primers
            coverage_update_list.append(coverage_renew)
            # new NN_array
            NN_array_update_list.append(NN_array_tmp)
        optimal_idx = coverage_update_list.index(max(coverage_update_list))
        degeneracy_update = score_trans(primer_update_list[optimal_idx])
        degenerate_number_update = sum([math.floor(score_table[x]) > 1 for x in primer_update_list[optimal_idx]])
        # optimal_primer_update, coverage_update,
        # NN_coverage_update, NN array_update,
        # degeneracy_update, degenerate_update
        return primer_update_list[optimal_idx], coverage_update_list[optimal_idx], \
               NN_coverage_update[optimal_idx], NN_array_update_list[optimal_idx], \
               degeneracy_update, degenerate_number_update

    def get_Y(self):
        Y_strict, Y_strict_R = [], []
        for y in self.position.split(","):
            y_index = int(y.strip())
            if y_index > 0:
                Y_strict.append(y_index)
                Y_strict_R.append(self.primer_length - y_index)
            else:
                Y_strict.append(self.primer_length + y_index + 1)
                Y_strict_R.append(-y_index + 1)
        return set(Y_strict), set(Y_strict_R)

    ##############################################################################################################
    ##############################################################################################################
    ##############################################################################################################
    ##############################################################################################################
    ##############################################################################################################
    ##############################################################################################################
    def refine_by_multiPrime2(self, primer_start, freq_matrix, cover, non_gap_seq_id, cover_for_MM,
                              cover_number, cBit, tBit):
        cover_primer_set = set(cover.keys())
        NN_matrix = self.trans_matrix(cover)
        row_names = np.array(freq_matrix.index.values).reshape(1, -1)
        if len(cover_for_MM) != 0:
            optimal_primer_index_NM = self.get_optimal_primer_by_viterbi(freq_matrix, NN_matrix)
            optimal_primer_index_MM = self.get_optimal_primer_by_MM(cover_for_MM)
            # build a list to store init base information in each position.
            optimal_primer_NM = ''.join(row_names[:, optimal_primer_index_NM][0].tolist())
            optimal_primer_MM = ''.join(row_names[:, optimal_primer_index_MM][0].tolist())
            # print(optimal_primer_NM)
            # print(optimal_primer_MM)
            if optimal_primer_NM == optimal_primer_MM:
                optimal_primer_current = optimal_primer_NM
                optimal_degenerate_primer, coverage = self.get_Y_position(optimal_primer_current, cover_primer_set,
                                                                          cover_number, cover)
            else:
                optimal_degenerate_primer_NM, NM_coverage = self.get_Y_position(optimal_primer_NM, cover_primer_set,
                                                                                cover_number, cover)
                optimal_degenerate_primer_MM, MM_coverage = self.get_Y_position(optimal_primer_MM, cover_primer_set,
                                                                                cover_number, cover)
                if NM_coverage >= MM_coverage:
                    optimal_degenerate_primer = optimal_degenerate_primer_NM
                    coverage = NM_coverage
                else:
                    optimal_degenerate_primer = optimal_degenerate_primer_MM
                    coverage = MM_coverage
        else:
            optimal_primer_index_NM = self.get_optimal_primer_by_viterbi(freq_matrix, NN_matrix)
            optimal_primer_NM = ''.join(row_names[:, optimal_primer_index_NM][0].tolist())
            optimal_degenerate_primer, coverage = self.get_Y_position(optimal_primer_NM, cover_primer_set,
                                                                      cover_number, cover)
        F_mis_cover, F_non_cover, R_mis_cover, R_non_cover = \
            self.mis_primer_check(cover_primer_set, optimal_degenerate_primer, cover, non_gap_seq_id)
        nonsense_primer_number = len(set(degenerate_seq(optimal_degenerate_primer)) - set(cover.keys()))
        primer_degenerate_number = dege_number(optimal_degenerate_primer)
        Tm = []
        for seq in degenerate_seq(optimal_degenerate_primer):
            Tm.append(Calc_Tm_v2(seq))
        Tm_average = round(mean(Tm), 2)
        degeneracy = score_trans(optimal_degenerate_primer)
        information = self.primer_pre_filter(optimal_degenerate_primer)
        primer_F = coverage + F_mis_cover
        primer_R = coverage + R_mis_cover
        out_mismatch_coverage = [primer_start,
                                 [str(cBit) + ":" + str(cover_number), str(tBit) + ":" + str(len(self.seq_dict)),
                                  optimal_degenerate_primer, primer_degenerate_number, degeneracy,
                                  nonsense_primer_number, coverage, primer_F,
                                  primer_R, Tm_average, information]]
        non_cov_primer_info = [primer_start, [F_non_cover, R_non_cover]]
        return out_mismatch_coverage, non_cov_primer_info

    #####################################################################
    def get_Y_position(self, optimal_primer, all_primers, cover_number, cover):
        optimal_primer_set = set(degenerate_seq(optimal_primer))
        uncover_primer_set = all_primers - optimal_primer_set
        primer_perfect_coverage = sum([cover[seq] for seq in optimal_primer_set])
        # length: Y_distance
        Y_dist_len_collection = defaultdict(list)
        # Y_dist_nuber calculation.
        Y_dist_number = defaultdict(int)
        for uncover_seq in uncover_primer_set:
            # Y_dist is list
            Y_dist = Y_position(optimal_primer, uncover_seq)
            # key is a set
            Y_dist_number["_".join(Y_dist)] += cover[uncover_seq]
            # print(Y_dist)
            # print(len(Y_dist))
            if Y_dist not in Y_dist_len_collection.values():
                Y_dist_len_collection[len(Y_dist)].append(Y_dist)
        # 注意并非所有mismatch都被允许，应考虑对引物效率有影响的关键位点
        coverage, degenerate_index = self.remove_elements(Y_dist_number, Y_dist_len_collection, primer_perfect_coverage,
                                                          self.number_of_dege_bases, self.variation, cover_number)
        # print(coverage, degenerate_index)
        optimal_degenerate_primer = degenerate_merge(optimal_primer, degenerate_index)
        # print(optimal_degenerate_primer)
        return optimal_degenerate_primer, coverage

    # subset 要仔细考虑，例如n=1时只需要考虑variation=2的情况，n=2时需要考虑variation=2,3的情况
    def remove_elements(self, Y_dist_number, Y_dist_len_collection, initial_coverage, N, variation, cover_number):
        max_count = 0
        max_subset = []
        for degenerate_number in range(1, N + 1):
            temp_variation = []
            temp_set = set()
            for variation_length in Y_dist_len_collection.keys():
                if 0 + variation < variation_length < degenerate_number + variation:
                    temp_variation.extend(Y_dist_len_collection[variation_length])
                    for tmp in Y_dist_len_collection[variation_length]:
                        temp_set = temp_set.union(set(tmp))
            # print(temp_set)
            if len(temp_set) >= degenerate_number:
                for comb in itertools.combinations(temp_set, degenerate_number):
                    count = 0
                    for Y_dist in temp_variation:
                        # print(Y_dist)
                        temp_result = set(Y_dist) - set(comb)
                        # print(temp_result)
                        if len(temp_result) <= variation:
                            count += Y_dist_number["_".join(Y_dist)]
                            if count + initial_coverage == cover_number:
                                return max_count, comb
                    if count > max_count:
                        max_count = count
                        max_subset = comb
        return max_count + initial_coverage, max_subset

    ##############################################################################################################
    ##############################################################################################################
    ##############################################################################################################
    ##############################################################################################################
    ##############################################################################################################
    ##############################################################################################################

    def mis_primer_check(self, all_primers, optimal_primer, cover, non_gap_seq_id):
        # uncoverage sequence in cover dict
        optimal_primer_set = set(degenerate_seq(optimal_primer))
        uncover_primer_set = all_primers - optimal_primer_set
        F_non_cover, R_non_cover = {}, {}
        F_mis_cover, R_mis_cover = 0, 0
        for uncover_primer in uncover_primer_set:
            Y_dist = Y_distance(optimal_primer, uncover_primer)
            # print(uncover_primer)
            # print(Y_dist)
            # print(set(Y_dist))
            if len(Y_dist) > self.variation:
                # record sequence and acc_ID which will never mis-coverage. too many mismatch!
                F_non_cover[uncover_primer] = non_gap_seq_id[uncover_primer]
                R_non_cover[uncover_primer] = non_gap_seq_id[uncover_primer]
            # if len(Y_dist) <= self.variation:
            else:
                if len(set(Y_dist).intersection(self.Y_strict)) > 0:
                    F_non_cover[uncover_primer] = non_gap_seq_id[uncover_primer]
                else:
                    F_mis_cover += cover[uncover_primer]
                if len(set(Y_dist).intersection(self.Y_strict_R)) > 0:
                    R_non_cover[uncover_primer] = non_gap_seq_id[uncover_primer]
                else:
                    R_mis_cover += cover[uncover_primer]
        # print(optimal_primer)
        # print(F_mis_cover)
        return F_mis_cover, F_non_cover, R_mis_cover, R_non_cover

    ################# get_primers #####################
    def run(self):
        p = ProcessPoolExecutor(self.nproc)  #
        sequence_dict = self.seq_dict
        start_primer = self.start_position
        stop_primer = self.stop_position
        # primer_info = Manager().list()
        # non_cov_primer_out = Manager().list()
        # for position in range(1245,  stop_primer - self.primer_length):
        for position in range(start_primer, stop_primer - self.primer_length):
            # print(position)
            p.submit(self.get_primers(sequence_dict, position))  # , primer_info, non_cov_primer_out
            # This will submit all tasks to one place without blocking, and then each
            # thread in the thread pool will fetch tasks.
        n = 0
        candidate_list, non_cov_primer_out, gap_seq_id_out = [], [], []
        with open(self.outfile, "w") as fo:
            headers = ["Position", "Entropy of cover (bit)", "Entropy of total (bit)", "Optimal_primer",
                       "Primer_degenerate_number", "Degeneracy",
                       "Nonsense_primer_number", "Optimal_coverage", "Mis-F-coverage", "Mis-R-coverage", "Tm",
                       "Information"]
            fo.write("\t".join(map(str, headers)) + "\n")
            while n < stop_primer - start_primer - self.primer_length:
                res = self.resQ.get()
                # The get method can read and delete an element from the queue. Similarly, the get method has two
                # optional parameters: blocked and timeout. If blocked is true (the default value) and timeout is
                # positive, no element is retrieved during the waiting time, and a Queue is thrown Empty exception.
                # If blocked is false, there are two cases. If a value of Queue is available, return the value
                # immediately. Otherwise, if the queue is empty, throw a Queue.Empty exception immediately.
                if res is None:
                    n += 1
                    continue
                candidate_list.append(res[0])
                non_cov_primer_out.append(res[1])
                gap_seq_id_out.append(res[2])
                n += 1
            sorted_candidate_dict = dict(sorted(dict(candidate_list).items(), key=lambda x: x[0], reverse=False))
            for position in sorted_candidate_dict.keys():
                fo.write(str(position) + "\t" + "\t".join(map(str, sorted_candidate_dict[position])) + "\n")
            fo.close()
            with open(self.outfile + '.non_coverage_seq_id_json', "w") as fj:
                json.dump(dict(non_cov_primer_out), fj, indent=4)
            fj.close()
            with open(self.outfile + '.gap_seq_id_json', "w") as fg:
                json.dump(dict(gap_seq_id_out), fg, indent=4)
            fg.close()
            # get results before shutdown. Synchronous call mode: call, wait for the return value, decouple,
            # but slow.
        p.shutdown()


class Primers_filter(object):
    def __init__(self, Total_sequence_number, primer_file, adaptor, rep_seq_number=0, distance=4, outfile="", diff_Tm=5,
                 clamp=0.6, di_nucl=0, size="300,700", position=9, GC="0.4,0.6", nproc=10, fraction=0.6):
        self.nproc = nproc
        self.primer_file = primer_file
        self.adaptor = adaptor
        self.size = size
        self.outfile = os.path.abspath(outfile)
        self.distance = distance
        self.fraction = fraction
        self.GC = GC
        self.clamp = clamp
        self.di_nucl=di_nucl
        self.diff_Tm = diff_Tm
        self.rep_seq_number = rep_seq_number
        self.number = Total_sequence_number
        self.position = position
        self.primers, self.gap_id, self.non_cover_id = self.parse_primers()
        self.resQ = Manager().Queue()
        self.pre_filter_primers = self.pre_filter()

    def parse_primers(self):
        primer_dict = {}
        with open(self.primer_file) as f:
            for i in f:
                if i.startswith("Pos"):
                    pass
                else:
                    i = i.strip().split("\t")
                    position = int(i[0])
                    primer_seq = i[3]
                    F_coverage = int(i[8])
                    R_coverage = int(i[9])
                    fraction = round(int(i[7]) / self.number, 2)
                    Tm = round(float(i[10]), 2)
                    primer_dict[position] = [primer_seq, fraction, F_coverage, R_coverage, Tm]
        # print(primer_dict)
        with open(self.primer_file + ".gap_seq_id_json") as g:
            gap_dict = json.load(g)
            g.close()
        with open(self.primer_file + ".non_coverage_seq_id_json") as n:
            non_cover_dict = json.load(n)
            g.close()
        return primer_dict, gap_dict, non_cover_dict

    ################# Dimer #####################
    def Pair_dimer_check(self, primer_F, primer_R):
        current_end_set = set(current_end_list(primer_F)).union(set(current_end_list(primer_R)))
        primer_pairs = [primer_F, primer_R]
        for pp in primer_pairs:
            for end in current_end_set:
                for p in degenerate_seq(pp):
                    idx = p.find(RC(end))
                    if idx >= 0:
                        end_length = len(end)
                        end_GC = end.count("G") + end.count("C")
                        end_d1 = 0
                        end_d2 = len(p) - len(end) - idx
                        Loss = Penalty_points(
                            end_length, end_GC, end_d1, end_d2)
                        delta_G = deltaG(end)
                        # threshold = 3 or 3.6 or 3.96
                        if Loss > 3.6 or (delta_G < -5 and (end_d1 == end_d2)):
                            return True
        return False

    ################# di_nucleotide #####################
    def di_nucleotide(self, primer):
        di_nucleotides = get_di_nucleotides(self.di_nucl)
        primers = degenerate_seq(primer)
        for m in primers:
            for n in di_nucleotides:
                if re.search(n, m):
                    return True
        return False

    def pre_filter(self):
        limits = self.GC.split(",")
        min = float(limits[0])
        max = float(limits[1])
        # min_cov = self.fraction
        candidate_primers_position = []
        primer_info = self.primers
        for primer_position in primer_info.keys():
            primer = primer_info[primer_position][0]
            # coverage = primer_info[primer_position][1]
            if hairpin_check(primer, self.distance):
                pass
            elif GC_fraction(primer) > max or GC_fraction(primer) < min:
                pass
            elif self.di_nucleotide(primer):
                pass
            else:
                candidate_primers_position.append(primer_position)
        return sorted(candidate_primers_position)

    def primer_pairs(self, start, adaptor, min_len, max_len, candidate_position, primer_pairs, threshold):
        primerF_extend = adaptor[0] + self.primers[candidate_position[start]][0]
        if hairpin_check(primerF_extend, self.distance):
            # print("hairpin!")
            pass
        elif dege_filter_in_term_N_bp(self.primers[candidate_position[start]][0], self.position):
            # print("term N!")
            pass
        elif GC_clamp(self.primers[candidate_position[start]][0],self.clamp):
            # print("GC_clamp!")
            pass
        else:
            start_index, stop_index = closest(candidate_position, candidate_position[start] + min_len,
                                              candidate_position[start] + max_len)
            if start_index > stop_index:
                pass
            else:
                for stop in range(start_index, stop_index + 1):
                    primerR_extend = adaptor[1] + RC(self.primers[candidate_position[stop]][0])
                    if hairpin_check(primerR_extend, self.distance):
                        # print("self hairpin!")
                        pass
                    elif dege_filter_in_term_N_bp(
                            RC(self.primers[candidate_position[stop]][0]), self.position):
                        pass
                    elif GC_clamp(RC(self.primers[candidate_position[stop]][0]),self.clamp):
                        pass
                    else:
                        distance = int(candidate_position[stop]) - int(candidate_position[start]) + 1
                        if distance > int(max_len):
                            print("Error! PCR product greater than max length !")
                            break
                        elif int(min_len) <= distance <= int(max_len):
                            # print(self.primers[candidate_position[start]][0],
                            #                     reversecomplement(self.primers[candidate_position[stop]][0]))
                            if self.Pair_dimer_check(self.primers[candidate_position[start]][0],
                                                     RC(self.primers[candidate_position[stop]][0])):
                                # print("Dimer detection between Primer-F and Primer-R!")
                                pass
                            else:
                                # primer_pairs.append((candidate_position[start], candidate_position[stop]))
                                difference_Tm = self.primers[candidate_position[start]][4] - \
                                                self.primers[candidate_position[stop]][4]
                                # difference of Tm between primer-F and primer-R  should less than threshold
                                if abs(difference_Tm) > self.diff_Tm:
                                    pass
                                else:
                                    start_pos = str(candidate_position[start])
                                    # print(start_pos)
                                    stop_pos = str(candidate_position[stop])
                                    # print(stop_pos)
                                    un_cover_list = []
                                    # start
                                    for o in list(dict(self.gap_id[start_pos]).values()):
                                        un_cover_list.extend(set(o))
                                    for p in list(dict(self.non_cover_id[start_pos][0]).values()):
                                        un_cover_list.extend(set(p))
                                    # stop
                                    for m in list(dict(self.gap_id[stop_pos]).values()):
                                        un_cover_list.extend(set(m))
                                    for n in list(dict(self.non_cover_id[stop_pos][1]).values()):
                                        un_cover_list.extend(set(n))
                                    all_non_cover_number = len(set(un_cover_list))
                                    if all_non_cover_number / self.number > threshold:
                                        pass
                                    else:
                                        all_coverage = self.number - all_non_cover_number
                                        cover_percentage = round(all_coverage / self.number, 4)
                                        average_Tm = str(round(mean([self.primers[candidate_position[start]][4],
                                                                     self.primers[candidate_position[stop]][4]]), 2))
                                        line = [self.primers[candidate_position[start]][0],
                                                RC(self.primers[candidate_position[stop]][0]),
                                                str(distance) + ":" + average_Tm + ":" + str(cover_percentage),
                                                all_coverage,
                                                str(candidate_position[start]) + ":" + str(candidate_position[stop]),
                                                set(un_cover_list)]
                                        primer_pairs.append(line)

    def run(self):
        p = ProcessPoolExecutor(self.nproc)  #
        size_list = self.size.split(",")
        min_len = int(size_list[0])
        max_len = int(size_list[1])
        candidate_position = self.pre_filter_primers
        adaptor = self.adaptor.split(",")
        primer_pairs = Manager().list()
        # print(candidate_position)
        coverage_threshold = 1 - self.fraction
        if int(candidate_position[-1]) - int(candidate_position[0]) < min_len:
            print("Max PCR product legnth < min len!")
            ID = str(self.outfile)
            with open(self.outfile, "w") as fo:
                fo.write(ID + "\n")
        else:
            for start in range(len(candidate_position)):
                # print(start)
                p.submit(self.primer_pairs(start, adaptor, min_len, max_len, candidate_position, primer_pairs,
                                           coverage_threshold))
            p.shutdown()
            if len(primer_pairs) < 10:
                primer_pairs = Manager().list()
                new_p = ProcessPoolExecutor(self.nproc)
                coverage_threshold -= 0.1
                for start in range(len(candidate_position)):
                    new_p.submit(self.primer_pairs(start, adaptor, min_len, max_len, candidate_position, primer_pairs,
                                                   coverage_threshold))
                new_p.shutdown()
            # ID = str(self.outfile)
            primer_ID = str(self.outfile).split("/")[-1].split(".")[0]
            with open(self.outfile + ".xls", "w") as fo_xls:
                headers = ["Primer_F_seq", "Primer_R_seq", "Product length:Tm:coverage_percentage",
                           "Target number", "Primer_start_end", "Uncovered ID"]
                fo_xls.write("\t".join(headers) + "\n")
                with open(self.outfile + ".fa", "w") as fa:
                    primer_pairs_sort = sorted(primer_pairs, key=lambda k: k[3], reverse=True)
                    for i in primer_pairs_sort:
                        fo_xls.write("\t".join(map(str, i)) + "\n")
                        start_stop = i[4].split(":")
                        fa.write(
                            ">" + primer_ID + "_" + start_stop[0] + "F\n" + i[0] + "\n>" + primer_ID + "_" +
                            start_stop[
                                1]
                            + "R\n" + i[1] + "\n")
                    fo_xls.close()
                    fa.close()


def MP_main():
    MP_input = dpg.get_value("DPrime_input")
    MP_out = dpg.get_value("DPrime_output")
    MP_plen = dpg.get_value("DPrime_plen")
    MP_fraction = dpg.get_value("DPrime_coverage")
    MP_di_nucl = dpg.get_value("DPrime_di_nucl")
    MP_dnum = dpg.get_value("DPrime_dnum")
    MP_degeneracy = dpg.get_value("DPrime_degeneracy")
    MP_entropy = dpg.get_value("DPrime_entropy")
    MP_size = dpg.get_value("DPrime_size")
    MP_coordinate = dpg.get_value("DPrime_positions")
    MP_variation = dpg.get_value("DPrime_var")
    MP_away = dpg.get_value("DPrime_hairpin")
    MP_gc = dpg.get_value("DPrime_GC")
    MP_clamp = dpg.get_value("DPrime_clamp")
    MP_proc = dpg.get_value("DPrime_proc")
    MP_sprimer = dpg.get_value("DPrime_sprimer")
    MP_adaptor = dpg.get_value("DPrime_adaptor")
    MP_diffTm = dpg.get_value("DPrime_diffTm")
    MP_dposition = dpg.get_value("DPrime_dposition")
    MP_method = dpg.get_value("DP_method")
    MP_indel = dpg.get_value("DP_indel")
    start_loading()
    e1 = time.time()
    print("INFO: Start times: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), e1))
    if MP_indel == "T":
        MP_indel = 0.9
    Seq_dict, Total_sequence_number = parse_seq(MP_input, threshold=MP_indel)
    NN_APP = NN_degenerate(Seq_dict=Seq_dict, Total_sequence_number=Total_sequence_number, primer_length=MP_plen,
                           coverage=MP_fraction, number_of_dege_bases=MP_dnum, score_of_dege_bases=MP_degeneracy,
                           raw_entropy_threshold=MP_entropy, product_len=MP_size, position=MP_coordinate,
                           variation=MP_variation, distance=MP_away, GC=MP_gc, method=MP_method, di_nucl=MP_di_nucl,
                           nproc=MP_proc, outfile=MP_sprimer)
    NN_APP.run()
    print("Step1 Done !")
    primer_pairs = Primers_filter(Total_sequence_number=Total_sequence_number, primer_file=MP_sprimer, GC=MP_gc,
                                  adaptor=MP_adaptor, distance=MP_away, outfile=MP_out, size=MP_size, clamp=MP_clamp,
                                  di_nucl=MP_di_nucl, position=MP_dposition, fraction=MP_fraction,
                                  diff_Tm=MP_diffTm, nproc=MP_proc)
    primer_pairs.run()
    stop_loading()
    e2 = time.time()
    print("INFO: Stop times: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), e2))
    print("INFO {} Total times: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
                                           round(float(e2 - e1), 2)))

def muscle_main():
    start_loading1()
    muscle_input = dpg.get_value("DPrime_input_fasta")
    muscle_output = dpg.get_value("DPrime_dposition")
    os.system("{}/Scripts/muscle5.1.win64.exe -align {} -output {}".format(os.path.abspath(os.path.dirname(__file__)),
                                                                           muscle_input, muscle_output))
    stop_loading1()

def dimer_main():
    start_loading2()
    Dimer_input = dpg.get_value("Dimer_input")
    Dimer_output = dpg.get_value("Dimer_output")
    Dimer_threshold = dpg.get_value("Dimer_threshold")
    Dimer_num = dpg.get_value("Dimer_num")
    dimer_app = Dimer(primer_file=Dimer_input, threshold=Dimer_threshold,
                      outfile=Dimer_output, nproc=Dimer_num)
    dimer_app.run()
    stop_loading2()


dpg.add_texture_registry(label="Demo Texture Container", tag="__demo_texture_container")
dpg.add_colormap_registry(label="Demo Colormap Registry", tag="__demo_colormap_registry")
_create_static_textures()
_create_dynamic_textures()

with dpg.font_registry() as main_font_registry:
    regular_font = dpg.add_font('fonts/Arial.ttf', 16)
    bold_font = dpg.add_font('fonts/Arial-Bold.ttf', 21)


# def loading_callback(sender, app_data, user_data: int):
#     if not dpg.does_item_exist("Loading Window"):
#         with dpg.window(label="Loading Window", tag="Loading Window", width=200, no_resize=True,
#                         no_move=True, no_close=True):
#             dpg.add_text("Now Loading")
#             dpg.add_loading_indicator()

# dpg.delete_item("Loading Window")


def show_info(title, message):
    # guarantee these commands happen in the same frame
    with dpg.mutex():
        viewport_width = dpg.get_viewport_client_width()
        viewport_height = dpg.get_viewport_client_height()

        with dpg.window(label=title, modal=True, no_close=True) as modal_id:
            dpg.add_text(message, wrap=viewport_width - 20)
            dpg.add_button(label="Ok", width=50, user_data=(modal_id, True), callback=lambda: dpg.delete_item(modal_id))

    # guarantee these commands happen in another frame
    dpg.split_frame()
    width = dpg.get_item_width(modal_id)
    height = dpg.get_item_height(modal_id)
    dpg.set_item_pos(modal_id, [viewport_width // 2 - width // 2, viewport_height // 2 - height // 2])


def _dimer_window():
    with dpg.child_window(autosize_y=False, width=800, height=150):
        with dpg.group(horizontal=True):
            dpg.add_text("Input", label="Label", show_label=False)
            dpg.add_button(label="?", callback=lambda: show_info("Info", "Please provide the primer file."))
            dpg.add_text("Output", label="Label", show_label=False, indent=400)
            dpg.add_button(label="?", callback=lambda: show_info("Info", "Please provide the Output file."))
        with dpg.group(horizontal=True):
            dpg.add_input_text(hint="Enter primer file here", width=300,
                               tag="Dimer_input")
            dpg.add_button(label="...", callback=finDimer_select_fasta_directory)
            dpg.add_input_text(hint="Enter output here", default_value="Dimer.output.xls", indent=400,
                               tag="Dimer_output", width=300)
        with dpg.group(horizontal=True):
            dpg.add_text("Dimer_threshold", label="Label", show_label=False)
            dpg.add_button(label="?",
                           callback=lambda: show_info("Info", "Badness threshold"))
            dpg.add_text("Dimer_process", label="Label", show_label=False, indent=400)
            dpg.add_button(label="?",
                           callback=lambda: show_info("Info", "Number of process to launch."))
        with dpg.group(horizontal=True):
            dpg.add_input_float(label="", default_value=3.6, tag="Dimer_threshold", width=100)
            dpg.add_input_int(label="", default_value=5, tag="Dimer_num", indent=400, width=100)

        with dpg.group(horizontal=True):
            # with dpg.theme(tag="__demo_theme"):
            #     with dpg.theme_component(dpg.mvButton):
            #         dpg.add_theme_color(dpg.mvThemeCol_Button, _hsv_to_rgb(3 / 7.0, 0.6, 0.6))
            #         dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, _hsv_to_rgb(3 / 7.0, 0.8, 0.8))
            #         dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, _hsv_to_rgb(3 / 7.0, 0.7, 0.7))
            #         dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 3 * 5)
            #         dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3 * 3, 3 * 3)
            dpg.add_button(label="Start", arrow=True, direction=dpg.mvDir_Right, callback=dimer_main,
                           tag="Start_button2")
            dpg.add_loading_indicator(color=(169, 127, 156), style=1, radius=2.2, show=False,
                                      tag="loading2")
            dpg.add_button(label="Reset Setting", width=100, height=22, indent=60,
                           callback=Dimer_reset_settings)


def _muscle_window():
    with dpg.child_window(autosize_y=False, width=800, height=70):
        with dpg.group(horizontal=True):
            dpg.add_text("Input", label="Label", show_label=False)
            dpg.add_button(label="?", callback=lambda: show_info("Info", "Please provide the FASTA file."))
            dpg.add_text("Output", label="Label", show_label=False, indent=400)
            dpg.add_button(label="?", callback=lambda: show_info("Info", "Please provide the Output file."))
        with dpg.group(horizontal=True):
            dpg.add_input_text(hint="Enter fasta file here", width=300,
                               tag="DPrime_input_fasta")
            dpg.add_button(label="...", callback=DPrime_select_fasta_directory)
            dpg.add_input_text(hint="Enter output here", default_value="Output.msa", indent=400,
                               tag="DPrime_msa_output", width=300)
            # with dpg.group(horizontal=True, horizontal_spacing=10):
            #     dpg.add_text("Start", label="Label", show_label=False)
            with dpg.group(horizontal=True):
                # with dpg.theme(tag="__demo_theme"):
                #     with dpg.theme_component(dpg.mvButton):
                #         dpg.add_theme_color(dpg.mvThemeCol_Button, _hsv_to_rgb(3 / 7.0, 0.6, 0.6))
                #         dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, _hsv_to_rgb(3 / 7.0, 0.8, 0.8))
                #         dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, _hsv_to_rgb(3 / 7.0, 0.7, 0.7))
                #         dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 3 * 5)
                #         dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3 * 3, 3 * 3)
                dpg.add_button(label="Start", arrow=True, direction=dpg.mvDir_Right, callback=muscle_main,
                               tag="Start_button1")
                # dpg.bind_item_theme(dpg.last_item(), "__demo_theme")
                dpg.add_button(label="!",
                               callback=lambda: show_info("Info", "Please note that this step may "
                                                                  "take a long time to complete. "
                                                                  "We recommend running this step "
                                                                  "on a High-Performance Computing "
                                                                  "(HPC) system to ensure faster "
                                                                  "processing."))
                dpg.add_loading_indicator(color=(169, 127, 156), style=1, radius=2.2, show=False,
                                          tag="loading1")


def _multiPrime_window():
    with dpg.group(horizontal=True, horizontal_spacing=10):
        with dpg.child_window(autosize_y=False, width=280, height=260):
            # 加载图片
            #print(os.path.dirname(os.path.abspath(__file__)))
            width, height, channels, data = dpg.load_image(
                # get_abs_path() + "/multiprime.ico")
                get_abs_path()+ "/img/Logo-multiPrime4.png")
                # 单独使用一下命令，当前目录会比纳城C盘
                # os.path.dirname(os.path.abspath(__file__)) + "/img/Logo-multiPrime4.png")
            # 0: width, 1: height, 2: channels, 3: data
            # 注册图片
            with dpg.texture_registry():
                dpg.add_static_texture(width, height, data, tag="image_id")
            with dpg.drawlist(width=242, height=242):
                # 绘制*3
                dpg.draw_image("image_id", (15, 10), (242, 237), uv_min=(0, 0), uv_max=(1, 1))
                # dpg.draw_image("image_id", (400, 300), (600, 500), uv_min=(0, 0), uv_max=(0.5, 0.5))
                # dpg.draw_image("image_id", (0, 0), (300, 300), uv_min=(0, 0), uv_max=(2.5, 2.5))
        with dpg.child_window(autosize_y=False, width=510, height=260):
            with dpg.group(horizontal=True):
                dpg.add_text("Input", label="Label", show_label=False)
                dpg.add_button(label="?",
                               callback=lambda: show_info("Info", "Multi-alignment output (muscle "
                                                                  "or others)."))
            with dpg.group(horizontal=True):
                dpg.add_input_text(hint="Enter Multiple alignment file here", width=400,
                                   tag="DPrime_input")
                dpg.add_button(label="...", callback=DPrime_select_directory)
            with dpg.group(horizontal=True):
                dpg.add_text("Primer length", label="Label", show_label=False)
                dpg.add_button(label="?", callback=lambda: show_info("Info", "Length of canddiate primer."))
                dpg.add_text("Indel Filter", label="Label", show_label=False, indent=300)
                dpg.add_button(label="?",indent=370,
                               callback=lambda: show_info("Info",
                                                          "Eliminate Indel regions where the percentage of "
                                                          "the Indel is greater than 90%. This stage involves "
                                                          "restructuring the input multiple alignment, and while "
                                                          "it demands a significant time investment, "
                                                          "the outcome is expected to yield more pronounced results."
                                                          "Use T (True) to encompass the filtration process. "
                                                          "Default: F (False)"))
            with dpg.group(horizontal=True):
                dpg.add_input_int(label="", default_value=18, width=90, tag="DPrime_plen")
                dpg.add_combo(("T", "F"), default_value="F", tag="DP_indel", indent=300, width=90)

            with dpg.group(horizontal=True):
                dpg.add_text("Single candidate primer", label="Label", show_label=False)
                dpg.add_button(label="?", callback=lambda: show_info("Info", "Single primer file."))
            dpg.add_input_text(hint="Enter positions here", default_value="DPrime.tmp",
                               tag="DPrime_sprimer", width=400)
            with dpg.group(horizontal=True):
                dpg.add_text("Primer pairs", label="Label", show_label=False)
                dpg.add_button(label="?", callback=lambda: show_info("Info", "Primer pair file: candidate "
                                                                             "primer pairs. Header of output: "
                                                                             "Primer_F_seq, Primer_R_seq, "
                                                                             "Product length:Tm:coverage_percentage, "
                                                                             "coverage_number, Primer_start_end"))
            dpg.add_input_text(hint="Enter output here", default_value="DPrime.out",
                               tag="DPrime_output", width=400)
            # with dpg.group(horizontal=True, horizontal_spacing=10):
            #     dpg.add_text("Start", label="Label", show_label=False)
            with dpg.group(horizontal=True):
                with dpg.theme(tag="__demo_theme"):
                    with dpg.theme_component(dpg.mvButton):
                        dpg.add_theme_color(dpg.mvThemeCol_Button, _hsv_to_rgb(3 / 7.0, 0.6, 0.6))
                        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, _hsv_to_rgb(3 / 7.0, 0.8, 0.8))
                        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, _hsv_to_rgb(3 / 7.0, 0.7, 0.7))
                        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 3 * 5)
                        dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3 * 3, 3 * 3)
                dpg.add_button(label="Start", arrow=True, direction=dpg.mvDir_Right, callback=MP_main,
                               tag="Start_button")
                dpg.bind_item_theme(dpg.last_item(), "__demo_theme")
                dpg.add_loading_indicator(color=(169, 127, 156), style=1, radius=2.2, show=False,
                                          tag="loading")
                dpg.add_button(label="Reset setting", width=100, height=30, indent=300,
                               callback=DP_reset_settings)

    dpg.add_text("Setting", label="Label", show_label=False)
    with dpg.group(horizontal=True, horizontal_spacing=10):
        with dpg.child_window(width=200, height=300, menubar=True, horizontal_scrollbar=True):
            with dpg.menu_bar():
                dpg.add_menu_item(label="Options 1")
            with dpg.group(horizontal=True):
                dpg.add_text("Degenerate number", label="Label", show_label=False)
                dpg.add_button(label="?",
                               callback=lambda: show_info("Info", "Maximum allowed count of degenerate "
                                                                  "bases during degenerate primer "
                                                                  "design."))
            dpg.add_input_int(label="", default_value=4, tag="DPrime_dnum")
            with dpg.group(horizontal=True):
                dpg.add_text("Degeneracy", label="Label", show_label=False)
                dpg.add_button(label="?",
                               callback=lambda: show_info("Info", "Maximum allowed degeneracy during "
                                                                  "degenerate primer design."))
            dpg.add_input_int(label="", default_value=10, tag="DPrime_degeneracy")
            with dpg.group(horizontal=True):
                dpg.add_text("Mismatch number", label="Label", show_label=False)
                dpg.add_button(label="?",
                               callback=lambda: show_info("Info", "Maximum allowed mismatch number "
                                                                  "during degenerate primer design."))

            dpg.add_slider_int(label="", min_value=0, max_value=2, default_value=1,
                               tag="DPrime_var")
            with dpg.group(horizontal=True):
                dpg.add_text("Entropy", label="Label", show_label=False)
                dpg.add_button(label="?",
                               callback=lambda: show_info("Info", "Entropy of primer-length window "
                                                                  "indicate a measure of disorder.  "
                                                                  "This parameter is employed to "
                                                                  "determine the conservation status "
                                                                  "of the window."))

            dpg.add_input_float(label="", default_value=3.6, tag="DPrime_entropy")
            with dpg.group(horizontal=True):
                dpg.add_text("GC", label="Label", show_label=False)
                dpg.add_button(label="?", callback=lambda: show_info("Info", "GC range of primer."))
                dpg.add_text("GC_clamp", label="Label", show_label=False, indent=80)
                dpg.add_button(label="?", indent=150,
                               callback=lambda: show_info("Info",
                                                          "Screen primers based on a 3-terminal GC clamp. "
                                                          "The default value is 0.6, but if you do not obtain any results, "
                                                          "you can adjust it to 0.8"))
            with dpg.group(horizontal=True):
                dpg.add_input_text(hint="Enter gc content here", default_value="0.2,0.7",
                                   tag="DPrime_GC", width=70)
                dpg.add_input_float(label="", default_value=0.8, tag="DPrime_clamp", indent=80, width=100)
        with dpg.child_window(width=200, height=300, menubar=True, horizontal_scrollbar=True):
            with dpg.menu_bar():
                dpg.add_menu_item(label="Options 2")
            with dpg.group(horizontal=True):
                dpg.add_text("Product size", label="Label", show_label=False)
                dpg.add_button(
                    label="?",
                    callback=lambda: show_info("Info", "The estimated length of the PCR (Polymerase Chain "
                                                       "Reaction) product that will be amplified by the two "
                                                       "primers."))
            dpg.add_input_text(hint="Enter product size here", default_value="200,500",
                               tag="DPrime_size")
            with dpg.group(horizontal=True):
                dpg.add_text("Primer coverage", label="Label", show_label=False)
                dpg.add_button(label="?",
                               callback=lambda: show_info("Info", "Filter primers by match fraction ("
                                                                  "Coverage with errors)."))
            dpg.add_input_float(label="", default_value=0.6, tag="DPrime_coverage")
            with dpg.group(horizontal=True):
                dpg.add_text("Positions", label="Label", show_label=False)
                dpg.add_button(
                    label="?",
                    callback=lambda: show_info("Info",
                                               "Mismatch index is not allowed to locate in specific "
                                               "positions. otherwise, it won't be regard as the "
                                               "mis-coverage. With this param, you can control the index "
                                               "of Y-distance (number=variation and position of "
                                               "mismatch). when calculate coverage with error. coordinate "
                                               "> 0: 5' == >3'; coordinate<0: 3' ==> 5'. You can set this "
                                               "param to any value that you prefer. Default: 1,"
                                               "-1. 1:  I dont want mismatch at the 2nd position, "
                                               "start from 0. -1: I dont want mismatch at the -1st "
                                               "position, start from -1."))
            dpg.add_input_text(hint="Enter positions here", default_value="1,2,-1",
                               tag="DPrime_positions")
            with dpg.group(horizontal=True):
                dpg.add_text("Process", label="Label", show_label=False)
                dpg.add_button(label="?",
                               callback=lambda: show_info("Info", "Number of process to launch."))
            dpg.add_input_int(label="", default_value=5, tag="DPrime_proc")
            with dpg.group(horizontal=True):
                dpg.add_text("Hairpin", label="Label", show_label=False)
                dpg.add_button(
                    label="?",
                    callback=lambda: show_info("Info",
                                               "Filter hairpin structure, which means distance of the "
                                               "minimal paired bases. Example:(number of X) AGCT["
                                               "XXXX]AGCT. Primers should not have complementary sequences "
                                               "(no consecutive 4 bp complementarities),otherwise the "
                                               "primers themselves will fold into hairpin structure."))
            dpg.add_input_int(label="", default_value=4, tag="DPrime_hairpin")
        with dpg.child_window(width=380, height=300, menubar=True, horizontal_scrollbar=True):
            with dpg.menu_bar():
                dpg.add_menu_item(label="Options 3")
            with dpg.group(horizontal=True):
                dpg.add_text("DPrime_adaptor", label="Label", show_label=False)
                dpg.add_button(
                    label="?",
                    callback=lambda: show_info("Info",
                                               "Adaptor sequence, which is used for NGS. Hairpin or "
                                               "dimer examination for [ adaptor--primer]. Example: "
                                               "TCTTTCCCTACACGACGCTCTTCCGATCT,"
                                               "TCTTTCCCTACACGACGCTCTTCCGATCT. If you dont want "
                                               "adaptor, use comma [,] instead"))

            dpg.add_input_text(hint="Enter positions here", default_value="TCTTTCCCTACACGACGCTCTTCCGATCT,"
                                                                          "TCTTTCCCTACACGACGCTCTTCCGATCT",
                               tag="DPrime_adaptor", width=-1)
            with dpg.group(horizontal=True):
                dpg.add_text("DPrime_diffTm", label="Label", show_label=False)
                dpg.add_button(
                    label="?",
                    callback=lambda: show_info("Info",
                                               "Maximum allowed difference of Tm between primer-F and "
                                               "primer-R."))
            dpg.add_input_int(label="", default_value=2, tag="DPrime_diffTm")
            with dpg.group(horizontal=True):
                dpg.add_text("DPrime_di_nucl", label="Label", show_label=False)
                dpg.add_button(
                    label="?",
                    callback=lambda: show_info("Info",
                                               'di_nucleotide refers to a pair of nucleotides that are '
                                               'consecutive in a DNA sequence. It indicates the maximum '
                                               'number of consecutive identical bases that are allowed in '
                                               'the sequence. For example, if the di_nucleotide value is '
                                               'set to 3, it means that up to three consecutive identical '
                                               'bases are allowed in the sequence. Any more than three '
                                               'consecutive identical bases would not be allowed '
                                               'according to the di_nucleotide constraint.'))
            dpg.add_input_int(label="", default_value=4, tag="DPrime_di_nucl")
            with dpg.group(horizontal=True):
                dpg.add_text("Method", label="Label", show_label=False)
                dpg.add_button(
                    label="?",
                    callback=lambda: show_info("Info",
                                               "The multiPrime1 method employs a nearest neighbor model, "
                                               "which is a local optimum strategy. In contrast, "
                                               "the multiPrime2 method adopts a global optimum strategy. "
                                               "The choice between these two methods depends on your "
                                               "specific needs and requirements. The multiPrime1 method is "
                                               "faster compared to multiPrime2. However, it is important to "
                                               "note that the difference in results between the two methods "
                                               "may not be significant. This is because the quality of the "
                                               "primers can vary depending on the degenerate strategy used. "
                                               "It is not solely based on coverage, as there are other "
                                               "parameters to consider."))
            dpg.add_combo(("multiPrime1", "multiPrime2"), default_value="multiPrime1", tag="DP_method")

            with dpg.group(horizontal=True):
                dpg.add_text("DPrime_dposition", label="Label", show_label=False)
                dpg.add_button(
                    label="?",
                    callback=lambda: show_info("Info",
                                               "Filter primers by degenerate base position. e.g. [4] "
                                               "means I dont want degenerate base appear at the end four "
                                               "bases when primer pre-filter"))
            dpg.add_input_int(label="", default_value=4, tag="DPrime_dposition")


def main_window():
    with dpg.window(label="multiPrime", width=1000, height=800, pos=(0, 0), tag="multiPrime"):
        with dpg.menu_bar():
            with dpg.menu(label="Usage", tag="Usage"):
                dpg.add_separator()
                dpg.add_text("multiPrime v2: Designing degenerate primers for maximum coverage with minimal degeneracy.")
                dpg.add_text("Input file: Result of multiple alignment. (muscle, mafft or others).", bullet=True,
                             wrap=700,
                             indent=20)
                dpg.add_text("Primer Length: Length of primer. Default: 18", bullet=True, wrap=700, indent=20)
                dpg.add_text("Output: Prefix of output file. Default: DPrime.out", bullet=True, wrap=700, indent=20)
                dpg.add_text("Output dir: directory of output file. Default: Absolute path of multiPrime", bullet=True,
                             wrap=700, indent=20)
                dpg.add_text("Number of degenerate bases: Max number of degenerate bases. Default: 4.  ", bullet=True,
                             wrap=700, indent=20)
                dpg.add_text("Degeneracy: Max Number of degeneracy. Default: 10.  ", bullet=True, wrap=700, indent=20)
                dpg.add_text("Mismatch number: Max mismatch number of primer. Default: 1. ", bullet=True, wrap=700,
                             indent=20)
                dpg.add_text("Entropy: A measure of disorder. This parameter is used to judge whether the window is "
                             "conservation. Entropy of primer-length window. Default: 3.6. ", bullet=True, wrap=700,
                             indent=20)
                dpg.add_text("GC: GC range of primer. Default [0.2,0.7].", bullet=True, wrap=700, indent=20)
                dpg.add_text("Product size: Mini product size: Default: 100. ", bullet=True, wrap=700, indent=20)
                dpg.add_text("Primer coverage: Filter primers by match fraction (Coverage with errors). Default: "
                             "0.6.", bullet=True, wrap=700, indent=20)
                dpg.add_text("Positions: Mismatch index is not allowed to locate in specific positions. "
                             "otherwise, it won't be regard as the mis-coverage. With this param, you can control the "
                             "index of Y-distance (number=variation and position of mismatch). when calculate "
                             "coverage with error. coordinate > 0: 5' == >3'; coordinate<0: 3' ==> 5'. You can set "
                             "this param to any value that you prefer. Default: 1,-1. 1:  I dont want mismatch at the "
                             "2nd position, start from 0. -1: I dont want mismatch at the -1st position, start from -1.",
                             bullet=True, wrap=700, indent=20)
                dpg.add_text("Process: Number of process to launch. Default: 5. ", bullet=True, wrap=700, indent=20)
                dpg.add_text("Hairpin: Filter hairpin structure, which means distance of the minimal paired bases. "
                             "Default: 4. Example:(number of X) AGCT[XXXX]AGCT. Primers should not have complementary "
                             "sequences (no consecutive 4 bp complementarities),otherwise the primers themselves will "
                             "fold into hairpin structure.", bullet=True, wrap=700, indent=20)
                dpg.add_text("DPrime_sprimer: A file that stores the single orientation primer and informations.",
                             bullet=True, wrap=700, indent=20)
                dpg.add_text("DPrime_adaptor: Adaptor sequence used for NGS next. This is used for hairpin or dimer "
                             "detection for [adaptor--primer]. Default: TCTTTCCCTACACGACGCTCTTCCGATCT,"
                             "TCTTTCCCTACACGACGCTCTTCCGATCT. If you don't want an adaptor, you can use a comma to "
                             "indicate no adaptor", bullet=True, wrap=700, indent=20)
                dpg.add_text(
                    "DPrime_diffTm: The difference in melting temperature (Tm) between the forward and reverse "
                    "primers. Default: 2", bullet=True, wrap=700, indent=20)
                dpg.add_text(
                    "DPrime_dposition:  Filters primers based on the position of degenerate bases. For example, "
                    "using [-e 4] means that degenerate bases should not appear in the last four bases during "
                    "primer pre-filtering. Default: 4.", bullet=True, wrap=700, indent=20)
                dpg.add_separator()

            with dpg.menu(label="Help"):
                with dpg.menu(label="About"):
                    with dpg.group(horizontal=False):
                        dpg.add_text(
                            "MultiPrime v2 is a user-friendly and one-step tool for designing degenerate primer"
                            "pairs. It integrates degenerate primer design theory with mismatch handling, resulting "
                            "in improved accuracy and specificity in detecting broad spectrum sequences. It "
                            "outperformed conventional programs in terms of run time, primer number, and primer "
                            "coverage. In the multiPrime Method, the 'multiPrime1' approach represents a local optimum "
                            "solution. In contrast, the 'multiPrime2' Method stands as a global optimum solution, "
                            "capable of achieving maximum primer coverage with minimal degeneracy.", wrap=500, indent=0)
                with dpg.menu(label="Version"):
                    with dpg.group(horizontal=True):
                        dpg.add_text("2.1.1")
                with dpg.menu(label="Source"):
                    with dpg.group(horizontal=True):
                        dpg.add_text("The source code of this demo can be found here:")
                        _hyperlink("multiPrime", "https://github.com/joybio/multiPrime")
                dpg.add_menu_item(label="Style Editor", callback=lambda: dpg.show_tool(dpg.mvTool_Style))
                dpg.add_menu_item(label="Font Manager", callback=lambda: dpg.show_tool(dpg.mvTool_Font))
                dpg.add_menu_item(label="Item Registry", callback=lambda: dpg.show_tool(dpg.mvTool_ItemRegistry))
                dpg.add_menu_item(label="Toggle Fullscreen", callback=lambda: dpg.toggle_viewport_fullscreen())
            with dpg.menu(label="Contact"):
                dpg.add_text("Please send comments, suggestions, bug reports and bug fixes to 1806389316@pku.edu.cn"
                             " or yang_junbo_hi@126.com.", wrap=500, indent=0)
                _hyperlink("Report an issue", "https://github.com/joybio/multiPrime/issues")

        with dpg.tab_bar(label='tabbar'):
            with dpg.tab(label='multiPrime'):
                _multiPrime_window()
            with dpg.tab(label='muscle'):
                _muscle_window()
            with dpg.tab(label='finDimer'):
                _dimer_window()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main_window()
    dpg.create_viewport(title='multiPrime', width=850, height=700, x_pos=0, y_pos=0)
    dpg.set_viewport_max_height(800)
    dpg.set_viewport_max_width(850)
    dpg.bind_font(regular_font)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("multiPrime", True)
    dpg.start_dearpygui()
    dpg.destroy_context()
