import numpy as np
from numpy import sin, cos, arctan2
from itertools import cycle
from sys import argv, exit
import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui
import math


class InvertedPendulum(QtGui.QWidget):
    '''Inicjalizacja stałych:
    M - masa wózka
    m - masa kulki
    l - długość ramienia wahadła

    Warunków początkowych:
    x0 - początkowe położenie wózka
    dx0 - początkowa prędkość wózka
    theta0 - początkowe położenie wahadla
    dtheta0 - początkowa prędkość wahadła

    Zakłócenia zewnętrznego:
    dis_cyc - zmienna odpowiada za to, czy zakłócenie jest zapętlone
    disruption - wartości zakłócenia w kolejnych chwilach czasowych

    Parametry planszy/obrazka:
    iw, ih - szerokość i wysokość obrazka
    x_max - maksymalna współrzędna pozioma (oś x jest symetryczna, więc minimalna wynosi -x_max)
    h_min - minialna współrzędna pionowa
    h_max - maksymalna współrzędna pionowa

    Powyższe dane są pobierane z pliku jeśli zmienna f_name nie jest pusta'''

    def __init__(self, M=10, m=5, l=50, x0=0, theta0=0, dx0=0, dtheta0=0, dis_cyc=True, disruption=[0], iw=1000,
                 ih=500, x_max=100, h_min=0, h_max=100, f_name="1.txt"):
        if f_name:
            with open(f_name) as f_handle:
                lines = f_handle.readlines()
                init_cond = lines[0].split(' ')
                self.M, self.m, self.l, self.x0, self.theta0, self.dx0, self.dtheta0 = [float(el) for el in
                                                                                        init_cond[:7]]
                self.image_w, self.image_h, self.x_max, self.h_min, self.h_max = [int(el) for el in init_cond[-5:]]
                if lines[1]:
                    self.disruption = cycle([float(el) for el in lines[2].split(' ')])
                else:
                    self.disruption = iter([float(el) for el in lines[2].split(' ')])
        else:
            self.M, self.m, self.l, self.x0, self.theta0, self.dx0, self.dtheta0 = M, m, l, x0, theta0, dx0, dtheta0
            self.image_w, self.image_h, self.x_max, self.h_min, self.h_max = iw, ih, x_max, h_min, h_max
            if dis_cyc:
                self.disruption = cycle(disruption)
            else:
                self.disruption = iter(disruption)
        super(InvertedPendulum, self).__init__(parent=None)

    # Inicjalizacja obrazka
    def init_image(self):
        self.h_scale = self.image_h / (self.h_max - self.h_min)
        self.x_scale = self.image_w / (2 * self.x_max)
        self.hor = (self.h_max - 10) * self.h_scale
        self.c_w = 16 * self.x_scale
        self.c_h = 8 * self.h_scale
        self.r = 8
        self.x = self.x0
        self.theta = self.theta0
        self.dx = self.dx0
        self.dtheta = self.dtheta0
        self.setFixedSize(self.image_w, self.image_h)
        self.show()
        self.setWindowTitle("Inverted Pendulum")
        self.update()

    # Rysowanie wahadła i miarki
    def paintEvent(self, e):
        x, x_max, x_scale, theta = self.x, self.x_max, self.x_scale, self.theta
        hor, l, h_scale = self.hor, self.l, self.h_scale
        image_w, c_w, c_h, r, image_h, h_max, h_min = self.image_w, self.c_w, self.c_h, self.r, self.image_h, self.h_max, self.h_min
        painter = QtGui.QPainter(self)
        painter.setPen(pg.mkPen('k', width=2.0 * self.h_scale))
        painter.drawLine(0, hor, image_w, hor)
        painter.setPen(pg.mkPen((165, 42, 42), width=2.0 * self.x_scale))
        painter.drawLine(x_scale * (x + x_max), hor, x_scale * (x + x_max - l * sin(theta)),
                         hor - h_scale * (l * cos(theta)))
        painter.setPen(pg.mkPen('b'))
        painter.setBrush(pg.mkBrush('b'))
        painter.drawRect(x_scale * (x + x_max) - c_w / 2, hor - c_h / 2, c_w, c_h)
        painter.setPen(pg.mkPen('r'))
        painter.setBrush(pg.mkBrush('r'))
        painter.drawEllipse(x_scale * (x + x_max - l * sin(theta) - r / 2), hor - h_scale * (l * cos(theta) + r / 2),
                            r * x_scale, r * h_scale)
        painter.setPen(pg.mkPen('k'))
        for i in np.arange(-x_max, x_max, x_max / 10):
            painter.drawText((i + x_max) * x_scale, image_h - 10, str(int(i)))
        for i in np.arange(h_min, h_max, (h_max - h_min) / 10):
            painter.drawText(0, image_h - (int(i) - h_min) * h_scale, str(int(i)))

    # Rozwiązanie równań mechaniki wahadła
    def solve_equation(self, F):
        l, m, M = self.l, self.m, self.M
        g = 9.81
        a11 = M + m
        a12 = -m * l * cos(self.theta)
        b1 = F - m * l * self.dtheta ** 2 * sin(self.theta)
        a21 = -cos(self.theta)
        a22 = l
        b2 = g * sin(self.theta)
        a = np.array([[a11, a12], [a21, a22]])
        b = np.array([b1, b2])
        sol = np.linalg.solve(a, b)
        return sol[0], sol[1]

    # Scałkowanie numeryczne przyśpieszenia, żeby uzyskać pozostałe parametry układu
    def count_state_params(self, F, dt=0.001):
        ddx, ddtheta = self.solve_equation(F)
        self.dx += ddx * dt
        self.x += self.dx * dt
        self.dtheta += ddtheta * dt
        self.theta += self.dtheta * dt
        self.theta = arctan2(sin(self.theta), cos(self.theta))

    # Uruchomienie symulacji
    # Zmienna sandbox mówi o tym, czy symulacja ma zostać przerwana w przypadku nieudanego sterowania -
    # - to znaczy takiego, które pozwoliło na zbyt duże wychylenia iksa lub na zbyt poziomo położenie wahadła
    def run(self, sandbox, frameskip=20):
        self.sandbox = False
        self.frameskip = frameskip
        self.init_image()
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.single_loop_run)
        timer.start(1)

    # n - krotne obliczenie następnego stanu układu
    # Gdzie n - to frameskip
    def single_loop_run(self):
        for i in range(self.frameskip + 1):
            dis = next(self.disruption, 0)
            control = self.fuzzy_control(self.x, self.theta, self.dx, self.dtheta)
            F = dis + control
            self.count_state_params(F)
            if not self.sandbox:
                if self.x < -self.x_max or self.x > self.x_max or np.abs(self.theta) > np.pi / 3:
                    exit(1)
        self.update()




    def AND(self,a, b):
        return min(a, b)

    def OR(self,a, b):
        return max(a, b)

    def NOT(self,a):
        return 1 - a

    # Regulator rozmyty, który trzeba zaimplementować
    def fuzzy_control(self, x, theta, dx, dtheta):
        #



            #######################################     Rozmycie


            ############################## theta

        def mi_theta_neg(a):
            if a <= - 0.05 :
                return 1
            elif a > -0.05 and a < 0:
                return -20* a
            else:
                return 0

        def mi_theta_zero(a):
            if a >= - 0.05 and a <= 0:
                return a * 20 + 1
            elif a < 0.05 and a > 0:
                return -20 * a + 1
            else:
                return 0

        def mi_theta_pos(a):
            if a >= 0.05:
                return 1
            elif a < 0.05 and a > 0:
                return 20 * a
            else:
                return 0
            ############################## dtheta

        def mi_dtheta_neg(a):
            if a <= - 100:
                return 1
            elif a > - 100 and a < 0:
                return -0.01 * a
            else:
                return 0

        def mi_dtheta_zero(a):
            if a >= - 100 and a <= 0:
                return a * 0.01 + 1
            elif a < 100 and a > 0:
                return -0.01 * a + 1
            else:
                return 0

        def mi_dtheta_pos( a):
            if a >= 100:
                return 1
            elif a < 100 and a > 0:
                return 0.01 * a
            else:
                return 0
            ############################        x

        def mi_x_neg(a):
            if a <= - 2:
                return 1
            elif a > - 2 and a < -1:
                return -1  * a  - 1
            else:
                return 0
        def mi_x_zero_minus(a):
            if a >= - 2 and a <= -1:
                return a * 1 + 2
            elif a < 0  and a > -1:
                return -1 * a
            else:
                return 0

        def mi_x_zero(a):
            if a >= - 1 and a <= 0:
                return a * 1 + 1
            elif a < 1  and a > 0:
                return -1 * a + 1
            else:
                return 0

        def mi_x_zero_plus(a):
            if a >= - 0 and a <= 1:
                return a * 1
            elif a < 1  and a > 0:
                return -1 * a + 2
            else:
                return 0
        def mi_x_pos(a):
            if a >= 1 :
                return 1
            elif a < 2  and a > 1:
                return  1* a -1
            else:
                return 0

        ############################################### dx
        def mi_dx_neg(a):
            if a <= - 1:
                return 1
            elif a > - 1  and a < 0:
                return - 1* a
            else:
                return 0

        def mi_dx_zero(a):
            if a >= - 1 and a <= 0:
                return a * 1 + 1
            elif a < 1  and a > 0:
                return -1 * a + 1
            else:
                return 0

        def mi_dx_pos(a):
            if a >= 1 :
                return 1
            elif a < 1  and a > 0:
                return 1 * a
            else:
                return 0

        # #########################      REGULY
        F_neg = mi_theta_neg(theta) #jeśli teta negatywna to sila negatywna
        F_pos = mi_theta_pos(theta) #jesli teta pos to sila pozytywna
        F_zero = mi_theta_zero(theta) # jesli teta zero to sila zero

        F_pos = self.OR (F_pos, self.AND(self.NOT(mi_theta_neg(theta)),mi_dtheta_neg(dtheta)))  #jesli tetea nie jest negatywna  i dteta jest negatywna to sila pozytywna
        F_neg = self.OR(F_neg, self.AND(self.NOT(mi_theta_pos(theta)), mi_dtheta_pos(dtheta)))  # symetrycznie
        F_zero = self.OR(F_zero, self.AND(mi_theta_zero(theta),mi_dtheta_zero(dtheta)))    # jeśli teta jest zero i dteta jest zero to fila zero

        F_pos = self.OR(F_pos, self.AND(mi_x_neg(x),self.AND(self.NOT(mi_theta_neg(theta)),mi_dtheta_zero(dtheta))))      # jesli x jest negatywny i teta negatywna i dteta zero to fila pozytywna
        F_neg= self.OR(F_neg, self.AND(mi_x_pos(x), self.AND(self.NOT(mi_theta_pos(theta)), mi_dtheta_zero(dtheta))))  # symetrycznie
        F_zero = self.OR(F_zero,self.AND(mi_x_zero(x), self.AND(mi_theta_zero(theta), mi_dtheta_zero(dtheta))))  # jeżeli x zero i teta zero i dteta zero to sila zero

        F_pos = self.OR(F_pos,self.AND(mi_x_zero(x), self.AND(mi_theta_zero(theta), mi_dx_pos(dx))))        # jeżeli x jest zero i teta jest zero i dx jest pozytywne to sila pozytywna
        F_neg = self.OR(F_neg, self.AND(mi_x_zero(x), self.AND(mi_theta_zero(theta), mi_dx_neg(dx))))       #symetrycznie

        F_zero = self.OR(F_pos, self.AND(self.NOT(mi_x_pos(x)), self.AND(mi_theta_zero(theta), mi_dx_neg(dx))))     # jeżeli x jest negatywne lub pozytywne i teta jest zero i dx jest negatywne fo sila na zero
        F_zero = self.OR(F_neg, self.AND(self.NOT(mi_x_neg(x)), self.AND(mi_theta_zero(theta), mi_dx_pos(dx))))     #symetrycznie

        F_zero = self.OR(F_zero, self.AND(mi_x_zero(x),self.NOT(mi_dx_zero(dx))))   # jeżeli x jest na zero i dx jest na zero to F jest na zero
        F_pos = self.OR(F_pos, self.AND(mi_dx_neg(dx), self.AND(self.AND(mi_dx_neg(dx),mi_dtheta_neg(dtheta)), self.AND(mi_x_zero(x),mi_theta_zero(theta)))))  #jezeli dx jest negatywne i dteta jest negatywne i x jest zero i tetea jest zero to sila jest pozytywna
        F_neg = self.OR(F_neg, self.AND(mi_dx_pos(dx),self.AND(self.AND(mi_dx_pos(dx),mi_dtheta_pos(dtheta)), self.AND(mi_x_zero(x),mi_theta_zero(theta)))))    # symetrycznie
        F_zero = self.OR(F_zero, self.OR(self.AND(mi_x_pos(x),mi_dx_neg(dx)),self.AND(mi_x_neg(x),mi_dx_pos(dx)))) #jezeli x jest pozytywne i dx jest negatywne lub jezeli x jest negatywne i dx pozytywne to siła na zero

        F_neg = self.OR(F_neg, self.AND(mi_x_zero_minus(x), mi_dx_pos(dx)))  #jezeli x jest troche na zero minus i dx jest na plus to siła na negatywnie
        F_pos = self.OR(F_pos, self.AND(mi_x_zero_plus(x), mi_dx_neg(dx))) # symetrycznie


        ############################################################################# Wyostrzenie przez scałokowanie obcietego wykresu i znalezeinie polowy tego pola
        F_max = 100
        F_control = float

        P_neg = F_neg*(F_max-5*(F_zero-1))               #pole powierzcjni figury obcietej od góry dla przedzialu mniejszego od zera
        P_pos = F_pos*(F_max-5*(F_zero-1))               #pole powierzcjni figury obcietej od góry dla przedzialu większego od zera
        P_zero = (5 + (F_zero-5)*10)* F_zero * 0.5       #pole powierzcjni figury obcietej od góry dla przedzialu w zerze
        Sy_prost_neg = F_neg * (F_max - F_neg) * ( 0.5*(F_max + F_neg)) #momenty bezwladnosci figur prostych oddalonych od zera z których sklada się zała figura
        Sy_troj_neg = 0.333333 * F_neg**3
        Sy_prost_pos = -F_pos * (F_max - F_pos) * (0.5 * (F_max + F_pos))
        Sy_troj_pos = -0.33333 * F_pos ** 3
        F_control = (Sy_prost_neg+Sy_prost_pos+Sy_troj_neg+Sy_troj_pos)/(P_pos + P_neg +P_zero+0.00001)  # wzór na srodek ciezkosci figur złozonych. w mianowniku mala wartosc dodana zeby nie bylo dzielenia przez zero


        print(x, dx, theta, dtheta, F_control)


        return F_control * self.m * self.M



if __name__ == '__main__':
    app = QtGui.QApplication(argv)
    if len(argv) > 1:
        ip = InvertedPendulum(f_name=argv[1])
    else:
        ip = InvertedPendulum(x0=0, dx0=0, theta0=0, dtheta0=0.1, ih=800, iw=1000, h_min=-80, h_max=80)
    ip.run(sandbox=True)
    exit(app.exec_())
