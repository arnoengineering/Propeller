#
print('gg')
import sys
from PyQt5.QtWidgets import *
import numpy as np
from numpy import sin, cos, tan, arctan, pi
from PyQt5.QtCore import Qt
from PyQt5 import QtCore
import pyqtgraph as pg
from functools import partial
# import pyqtgraph.opengl as gl


# import os
# from functools import partial


class Prop:
    def __init__(self):
        self.vals = {'R': 20,
                     'NACA': '4412',
                     'r_hub': 8,
                     'Points per section': 100,
                     'Maximum cord length': 8,
                     'Number of sections': 100,
                     'Initial pitch angle': 20,
                     'Number of blades': 3,
                     'Section': 50}

        self.depend_vals = {'Cord Length': 0.0,
                            'Rad of Section': 0.0,
                            'Max Chamber': 0.0,
                            'Chamber Location': 0.0,
                            'Thickness': 0.0,
                            'Pitch': 0.0}

        self._gen_emp_ls()
        self._gen_init_vals()
        self._naca()
        self._gen_scale()
        self._set_p_vals()
        self.set_current_p_vals()
        self.gen_pitch()

    def _gen_emp_ls(self):
        arr_size = self.vals['Number of sections'],
        self.p_l = np.zeros(arr_size)
        self.r_l = np.zeros(arr_size)
        self.y_t = np.zeros(arr_size)
        self.x_t = np.zeros(arr_size)
        self.c_l = np.zeros(arr_size)
        self.chamber_line_sec = np.zeros(arr_size)
        self.blade = None
        self.chamber_line = None

    def _gen_init_vals(self):
        self.max_chamber = 0
        self.chamber_loc = 0
        self.max_thick = 0
        self.cur_blade = 0
        self.pitch_len = 0
        self.r = 0
        self.pitch = 0

    def _gen_scale(self):
        # percent then find legth
        self.depend_vals['Rad of Section'] = self.vals['Section']/100 * (self.vals['R']-self.vals['r_hub']) + self.vals['r_hub']
        per_sec = int(self.vals['Section']/100*self.vals['Number of sections'])*self.vals['Points per section']
        self.cur_range = (per_sec, per_sec+self.vals['Points per section']*2)

    def set_xy(self):
        self.y_t = self.blade[1, self.cur_range[0]:self.cur_range[1]]
        self.x_t = self.blade[0, self.cur_range[0]:self.cur_range[1]]
        self.chamber_line_sec = self.chamber_line[:, self.cur_range[0]:self.cur_range[1]-self.vals['Points per section']]


    def _naca(self, na=None):
        if na:
            self.vals['NACA'] = na
        if not isinstance(na, str):
            str(na)

        self.max_chamber_naca = int(self.vals['NACA'][0]) / 100
        self.chamber_loc_naca = int(self.vals['NACA'][1]) / 10
        self.max_thick_naca = int(self.vals['NACA'][-2:]) / 100

    def _set_p_vals(self):
        self.cord_legnth = self.cord_len_gen(self.r)

        self.max_chamber = self.max_chamber_naca * self.cord_legnth
        self.chamber_loc = self.chamber_loc_naca * self.cord_legnth
        self.max_thick = self.max_thick_naca * self.cord_legnth

    def set_current_p_vals(self):
        self.depend_vals['Cord Length'] = self.cord_len_gen(self.depend_vals['Rad of Section'])
        self.depend_vals['Max Chamber'] = self.max_chamber_naca * self.depend_vals['Cord Length']
        self.depend_vals['Chamber Location'] = self.chamber_loc_naca * self.depend_vals['Cord Length']
        self.depend_vals['Thickness'] = self.max_thick_naca * self.depend_vals['Cord Length']

    def naca(self, na):
        self._naca(na)

    def cord_len_gen(self, r):
        c_0 = self.vals['R'] / 2
        par = (r - c_0) ** 2 / c_0 ** 2
        return np.sqrt(self.vals['Maximum cord length'] ** 2 * (1 - par))

    def y_t_fun(self, x):
        return 5 * self.max_thick * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1015 * x ** 4)

    def p_x(self, x):
        return 2 * self.chamber_loc * x - x ** 2

    def y_c_fun(self, x):
        if 0 <= x <= self.chamber_loc:
            return self.max_chamber / self.chamber_loc ** 2 * self.p_x(x)
        else:
            return self.max_chamber / (1 - self.chamber_loc) ** 2 * (1 - 2 * self.chamber_loc + self.p_x(x))

    def fun_arr(self, fun, x):
        if isinstance(x, (list, tuple, np.ndarray)):
            return np.array(list(map(fun, x)))
        else:
            return fun(x)

    def theta_fun(self, x):
        if 0 <= x <= self.chamber_loc:
            return arctan(2 * self.max_chamber / self.chamber_loc ** 2 * (self.chamber_loc - x))
        else:
            return arctan(2 * self.max_chamber / (1 - self.chamber_loc) ** 2 * (self.chamber_loc - x))

    def y_c_f(self, x):
        return self.fun_arr(self.y_c_fun, x)

    def y_t_f(self, x):
        return self.fun_arr(self.y_t_fun, x)

    def theta_f(self, x):
        return self.fun_arr(self.theta_fun, x)

    def pitch_f(self, r):
        return np.arctan(self.pitch_len / (r * 2 * np.pi))

    def gen_foil_sec(self):
        # todo temp_d
        cos_space = np.linspace(0, np.pi, int(self.vals['Points per section']))
        x_v = (1 - cos(cos_space)) / 2
        y_t = self.y_t_f(x_v)

        y_c = np.nan_to_num(self.y_c_f(x_v))
        theta = self.theta_f(x_v)

        x_fv = y_t * sin(theta)
        y_fv = y_t * cos(theta)

        x_u = x_v - x_fv
        x_l = x_v + x_fv

        y_u = y_c + y_fv
        y_l = y_c - y_fv
        x_t = np.concatenate((x_u, np.flip(x_l))) * self.r
        y_t = np.concatenate((y_u, np.flip(y_l)))
        chamber_arc = np.array([x_v * self.r, y_c])
        return np.array([x_t, y_t]), chamber_arc

    def wrap_sec(self, point, r):
        # assume y allong shaft, should work either way
        cf = np.pi * 2 * r
        x = point[0]
        y = point[1]
        return np.array([x, r*np.sin(y/cf), -r*np.cos(y/cf)])

    def rot_prop_ar(self, ang):
        return np.array([[1, 0, 0],
                  [0, cos(ang), -sin(ang)],
                  [0, sin(ang), cos(ang)]])

    def rot_sec_arr(self, point, center_point):
        trans_point = point + center_point
        array = np.array([[cos(self.pitch), sin(self.pitch)],
                         [-sin(self.pitch), cos(self.pitch)]])
        rot_point = array @ trans_point
        return rot_point - center_point

    def gen_c_bound(self):
        pass

    def gen_cenline(self):
        pass

    def gen_rake(self):
        pass

    def gen_scew(self):
        pass

    def rot_sec(self):
        pass

    def fullsec(self):
        # u_l = u or l
        # xp = -(i + r* skew*tan(p_ang)) + (0.5*c-x)*sin(p_ang)+cos(p_ang)*y_u_l
        # (0.5 * c - x) * cos(p_ang), todo flip x,y in other
        pass

    def gen_pitch(self):
        self.r_l = np.linspace(self.vals['r_hub'], self.vals['R'], self.vals['Number of sections'])

        self.pitch_len = self.vals['r_hub'] * 2 * np.pi * np.tan(np.deg2rad(self.vals['Initial pitch angle']))
        self.p_l = self.pitch_f(self.r_l)
        self.c_l = self.cord_len_gen(self.r_l)
        for i in range(self.r_l.size):

            self._set_p_vals()
            self.r = self.r_l[i]
            self.pitch = self.p_l[i]
            xy_0 = self.gen_foil_sec()
            rot_p = np.array([self.c_l[i]/2, 0]).reshape((2, 1))
            xy = self.rot_sec_arr(xy_0[0], rot_p)

            secs = np.vstack((xy, np.ones((1, xy.shape[1]))*self.r_l[i]))
            self.blade = np.hstack((self.blade, secs)) if isinstance(self.blade,np.ndarray) else secs
        # todo chamber line
        # todo cord line

        self.depend_vals['Pitch'] = self.pitch_f(self.depend_vals['Rad of Section']) * 180 / np.pi
        # self.set_xy()

    def gen_curr_sec(self):
        self.r = self.depend_vals['Rad of Section']
        self._set_p_vals()
        self.pitch = self.pitch_f(self.r)
        xy_0 = self.gen_foil_sec()

        rot_p = np.array([self.depend_vals['Cord Length']/2 / 2, 0]).reshape((2, 1))
        xy = self.rot_sec_arr(xy_0[0], rot_p)
        # self.chamber_line_sec = self.rot_sec_arr(xy_0[1], rot_p)
        self.chamber_line_sec = xy_0[1]
        self.x_t = xy_0[0][0]
        self.y_t = xy_0[0][1]
        self.depend_vals['Pitch'] = self.pitch_f(self.depend_vals['Rad of Section']) * 180 / np.pi

    def disp_pitch(self):
        # print('pitch: {}, {}'.format(self.p_l.size, self.r_l.size))
        return self.r_l, self.p_l

    def disp_cord(self):
        # print('cord: {}, {}'.format(self.c_l.size, self.r_l.size))
        return self.r_l, self.c_l

    def disp_sec(self):
        # print('sec: {}, {}'.format(self.x_t.size, self.y_t.size))
        return [self.x_t, self.y_t], self.chamber_line_sec

    def update_self(self, n,i):
        if n != 'NACA':
            i = float(i)
        self.vals[n] = i

        self._gen_scale()
        self.set_current_p_vals()
        self.gen_curr_sec()
        if i != 'Section':
            self.gen_pitch()
        # todo multi window
        # todo value
        # todo naca slider
        # todo still change sec on section


class DispWin(pg.GraphicsLayoutWidget):
    def __init__(self, prop):
        super().__init__()
        self.prop = prop

        pg.setConfigOptions(antialias=True)
        self._create_plots()

    def _create_plots(self):
        self.waveform = self.addPlot(1, 0, colspan=2)
        self.waveform.enableAutoRange(True)
        self.waveform.showGrid(True, True)
        self.waveform_plot = self.waveform.plot(pen='c', width=3)

        self._scale_wave()
        # self.set_scale()

    def _scale_wave(self):  # for chunkupdate
        self.rang = self.reset_r()
        # self.waveform.setYRange(*self.rang[1], padding=0)
        # self.waveform.setXRange(*self.rang[0], padding=0.005)

    def reset_scale(self):
        self._scale_wave()

class SectionWin(DispWin):
    def __init__(self, prop):
        super().__init__(prop)
        # lockAspect = False
        self.pitchline = self.waveform.addLine(pen='g')
        pen = pg.mkPen('r', style=QtCore.Qt.DashLine)
        self.chamber = self.waveform.plot(pen=pen)

    def reset(self):
        xy = self.prop.disp_sec()

        self.waveform_plot.setData(*xy[0])
        self.chamber.setData(*xy[1])
        self.pitchline.setValue((self.prop.depend_vals['Cord Length'] / 2, 0))
        self.pitchline.setAngle(self.prop.depend_vals['Pitch'])

    def reset_r(self):
        y_abs = self.prop.depend_vals['Max Chamber']+self.prop.depend_vals['Thickness']
        return [[0, self.prop.depend_vals['Rad of Section']], [-y_abs, y_abs]]


class PitchWin(DispWin):
    def __init__(self, prop):
        super().__init__(prop)

    def reset(self):
        self.waveform_plot.setData(*self.prop.disp_pitch())

    def reset_r(self):
        return [[self.prop.vals['r_hub'], self.prop.vals['R']],
                [self.prop.vals['Initial pitch angle'] * pi / 180, pi * 2]]


class CordWin(DispWin):
    def __init__(self, prop):
        super().__init__(prop)

    def reset(self):
        self.waveform_plot.setData(*self.prop.disp_cord())

    def reset_r(self):
        return [[self.prop.vals['r_hub'], self.prop.vals['R']],
                [0, self.prop.vals['Maximum cord length']]]


# class D3Win(gl.GLViewWidget):
#     def __init__(self):
#         super().__init__()


class Window(QMainWindow):
    # noinspection PyArgumentList
    def __init__(self):
        super().__init__()
        # self.setAcceptDrops(True)
        # self.file = r'N:\PC stuff\Programs\Python\Fourier\4th.jpg'

        self.setWindowTitle('Prop')
        self.prop = Prop()

        self.start = QWidget()
        self.layout0 = QHBoxLayout()
        self.layout = QGridLayout()
        self.start.setLayout(self.layout0)
        self.layout1 = QGridLayout()
        self.layout0.addLayout(self.layout)
        self.layout0.addLayout(self.layout1)
        self.setCentralWidget(self.start)
        # todo on start
        # self.prop_eq = [prop ]
        # self.menu_items = {'R': 20, 'naca': '2414', 'r_hub': 8, 'points': 100, 'cord_max': 8, 'secs': 100, 'p_ang': 20}

        self.running = False

        self._create_controls()
        self._wins()
        self._docks()
        self.timer = QtCore.QTimer()
        self.animation()
        self.on_start()

    def _wins(self):
        # [self.R, self.na, self.r_0, self.points, self.c_max, self.sec_cnt, self.pangle]
        self.pitch = PitchWin(self.prop)

        self.cord = CordWin(self.prop)  # todo rep with func to set
        self.section = SectionWin(self.prop)
        # self.tools = Tab()
        self.widgets = [self.section, self.cord, self.pitch]

    def _docks(self):
        self.pi_dock = QDockWidget('pitch')
        self.cord_dock = QDockWidget('cd')
        self.section_dock = QDockWidget('sec')

        self.pi_dock.setMaximumSize(3000,150)
        self.cord_dock.setMaximumSize(3000,150)
        self.section_dock.setMinimumSize(800,500)

        # self.AllowTabbedDocks()

        self.cord_dock.setWidget(self.cord)
        self.pi_dock.setWidget(self.pitch)
        self.section_dock.setWidget(self.section)

        self.addDockWidget(Qt.BottomDockWidgetArea, self.cord_dock)
        self.addDockWidget(Qt.TopDockWidgetArea, self.pi_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.section_dock)

    # noinspection PyArgumentList
    def _create_controls(self):

        self.but = {}
        self.but_d = {}
        n = 0
        for i, j in self.prop.vals.items():
            if i != 'Section':
                li = QLineEdit()
                li.setText(str(j))
                lab = QLabel(i)
                self.layout.addWidget(lab, n, 0)
                self.layout.addWidget(li, n, 1)
                li.editingFinished.connect(partial(self.update_cmd, i))
                self.but[i] = li
                n += 1
        self.cur = QSlider(Qt.Horizontal)
        self.cur.setMinimum(0)
        self.cur.setMaximum(100)
        self.cur.setValue(50)
        self.cur.valueChanged.connect(partial(self.update_cmd, 'Section'))
        self.layout.addWidget(self.cur, n, 0, 1, 2)

        n = 0
        for i, j in self.prop.depend_vals.items():
            li = QLineEdit()
            li.setText(str(j))
            # li.setReadOnly(True)
            lab = QLabel(i)
            self.but_d[i] = li
            self.layout1.addWidget(lab, n, 0)
            self.layout1.addWidget(li, n, 1)
            n += 1

    def update_cmd(self, typ):
        if typ == 'Section':
            val = self.cur.value()
        else:
            val = self.but[typ].text()
        self.prop.update_self(typ, val)
        for i in self.widgets:
            i.reset_scale()
        for i, j in self.prop.depend_vals.items():
            self.but_d[i].setText(str(round(j, 3)))

    def animation(self):
        print('Animation starting')
        self.timer.setInterval(20)
        self.timer.timeout.connect(self.upd)

    def on_stop(self):
        print('stopped')
        if self.running:
            self.running = False
            self.timer.stop()

    def on_start(self):
        print('started')
        if not self.running:
            self.running = True
            # self.update_cmd()
            self.timer.start()

    def upd(self):
        # fft
        # print('up')
        for i in self.widgets:

            i.reset()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    audio_app = Window()
    audio_app.show()
    sys.exit(app.exec_())
