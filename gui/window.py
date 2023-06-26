import sys
import time
import functools

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QSizePolicy, QSlider, QSpacerItem, \
    QVBoxLayout, QWidget, QMessageBox
from PyQt5.QtGui import QImage, QPixmap

import pyqtgraph as pg
import numpy as np

import cv2


class SliderV(QWidget):
    '''
    Widget for creating a vertical-oriented slider
    following https://stackoverflow.com/questions/42007434/slider-widget-for-pyqtgraph
    '''
    def __init__(self, name, range, parent=None):
        '''
        params:
        :name: str, parameter name
        :range: list, [min (int), max (int), interval (int)]
        '''
        super(SliderV, self).__init__(parent=parent)
        
        self.verticalLayout = QVBoxLayout(self)

        self.name_label = QLabel(self)
        self.verticalLayout.addWidget(self.name_label)
        self.name_label.setText(name)
        self.label = QLabel(self)
        # self.label.setStyleSheet("color: rgb(255, 0, 0);")
        self.verticalLayout.addWidget(self.label)

        self.slider = QSlider(self)
        self.slider.setMinimum(range[0])
        self.slider.setMaximum(range[1])
        self.slider.setTickInterval(range[2])
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setOrientation(Qt.Vertical)
        self.verticalLayout.addWidget(self.slider)
        self.resize(self.sizeHint())

        self.minimum = range[0]
        self.maximum = range[1]
        self.slider.valueChanged.connect(self.setLabelValue)
        self.x = None
        self.setLabelValue() # initialize: minimum

        # TODO: show min, max

    def setLabelValue(self):
        self.label.setText("{0:.4g}".format(self.slider.value()))
    
    def setValue(self, value):
        self.slider.setValue(value)
        QApplication.processEvents()

    def setMinimum(self, min):
        self.slider.setMinimum(min)
        QApplication.processEvents()
        self.minimum = min

    def setMaximum(self, max):
        self.slider.setMaximum(max)
        QApplication.processEvents()
        self.maximum = max


class CurveWidget(QWidget):
    '''
    Widget for displaying the function curve
    '''
    def __init__(self, parent=None):
        super(CurveWidget, self).__init__(parent=parent)

        self.verticalLayout = QVBoxLayout(self)

        self.win = pg.GraphicsLayoutWidget() # title
        self.verticalLayout.addWidget(self.win)
        self.fig = self.win.addPlot(title="")
        self.curve = self.fig.plot(pen='r')

    def update_curve(self, y): 
        self.curve.setData(y)


class ImageWidget(QWidget):
    '''
    Widget for displaying the image
    '''
    def __init__(self, img_size, parent=None):
        '''
        params:
        :img_size: list, [h, w]
        '''
        super(ImageWidget, self).__init__(parent=parent)
        
        self.verticalLayout = QVBoxLayout(self)

        self.img_label = QLabel(self)
        self.verticalLayout.addWidget(self.img_label)
        
        # initialize pixmap
        self.h, self.w = img_size
        img_arr = np.ones((self.h, self.w))*255
        self.update_img(img_arr)
        
    def update_img(self, img_arr):
        assert (self.h,self.w) == img_arr.shape
        qImg = QImage(img_arr, self.w, self.h, self.w*3, QImage.Format_BGR888)
        self.img_label.setPixmap(QPixmap(qImg))


class UIWidget(QWidget):
    '''
    Widget for showing the whole interface with the parameter space being searched
    by the sequential-line-search algorithm by moving the 1-d slider
    '''
    def __init__(self, obj_fn, params, optimizer, determine_fn=None, parent=None):
        '''
        params:
        :obj_fn: the objective function to maximize
        :determine_fn: if not using the slider, using a function to automatically
        determine the position by some rule alternatively
        '''
        super(UIWidget, self).__init__(parent=parent)

        self.sliders = []
        self.obj_fn = obj_fn
        self.determine_fn = determine_fn
        self.optimizer = optimizer

        # GUI elements
        self.horizontalLayout = QHBoxLayout(self)       

        # param bars
        for k,v in params.items():
            slider = SliderV(k, v)
            self.horizontalLayout.addWidget(slider)
            self.sliders.append(slider)
        self.slider = SliderV('slider', [0,10,1])
        self.horizontalLayout.addWidget(self.slider)
        
        # graph window
        # self.image = ImageWidget([256,256])
        self.curve = CurveWidget()
        self.horizontalLayout.addWidget(self.curve)

        # timesteps
        self.t = 0
        self.text_label = QLabel(self)
        self.horizontalLayout.addWidget(self.text_label) 
        self.text_label.setText(f'timestep: {self.t}')
        self.text_label.adjustSize()

        # # update
        # if not determine_fn:
        #     self.slider.slider.valueChanged.connect(self.update_sliders)
        # self.sliders[-1].slider.valueChanged.connect(self.update_window)
    
    def update_sliders(self):
        slider_ends = self.optimizer.get_slider_ends()
        if self.determine_fn:
            pos = self.determine_fn(slider_ends)
            self.slider.setValue(round(pos*10))
        else:
            pos = self.slider.slider.value()/10       
        self.optimizer.submit_feedback_data(pos)
        opt_x = self.optimizer.get_maximizer() # [0,1]
        unscaled_xs = []
        # update param bars
        for i,x in enumerate(opt_x):
            slider = self.sliders[i]
            # unscale
            unscaled_x = slider.minimum + (slider.maximum - slider.minimum)*x
            unscaled_xs.append(unscaled_x)
            slider.setValue(round(unscaled_x))
        
        print('parameters updated!')
        return unscaled_xs

    def update_window(self):
        # update param bars
        unscaled_xs = self.update_sliders()

        # update display
        y = self.obj_fn(np.array(unscaled_xs))
        print(np.max(y))
        self.curve.update_curve(y)

        self.t += 1
        self.text_label.setText(f'timestep: {self.t}')