# coding:utf-8
import numpy as np
import time
# from datetime import datetime
from transitions import Machine
from _turntable_controler import StageControl
# from _recode_func import RecodeFunc
# from multiprocessing import Pool
# from _socket_multi_crient import SocketClient
from _function import MyFunc
from _capture_photo import TakePhoto
import sys


class StateMachine(object):
    states = ['Turn', 'Recode', 'Wait', 'Init', 'Next']

    def __init__(self, name):
        self.name = name
        self.machine = Machine(model=self, states=StateMachine.states, initial='Wait', auto_transitions=False)
        self.machine.add_transition(trigger='Turn_start', source='Init', dest='Turn', after='turn_table')
        self.machine.add_transition(trigger='Turn_fin', source='Turn', dest='Recode', before='recode')
        self.machine.add_transition(trigger='Rec_fin', source='Recode', dest='Wait', after='wait')
        self.machine.add_transition(trigger='Init_set', source='Wait', dest='Init', before='init')
        self.machine.add_transition(trigger='Turn_res', source='Wait', dest='Turn', after='turn_table')
        self.machine.add_transition(trigger='Next_dis', source='Wait', dest='Init', after='prepare_next')
        self.stage = StageControl()
        self.func = MyFunc()
        self.photo = TakePhoto()
        self.photo_path = self.func.make_dir_path(photo=True)
        self.adjust = 50

    def turn_table(self, name):
        self.stage.move_table(name)
        print(name-self.adjust, ' [deg]')

    @staticmethod
    def wait(sleep_time=2):
        time.sleep(sleep_time)
        pass

    def recode(self, dir_name):
        # vapor switch on and take a photo
        self.photo.capture(self.photo_path + str(dir_name))
        # pass
        # if self.socket.socket_1 is not None:
        #     self.photo.capture(self.photo_path + '/' + str(dis_name) + '/' + str(dir_name))
        #     self.socket.send_GO(self.mic_path + '/' + str(dis_name) + '/' + str(dir_name) + '.wav')

    def init(self):
        if self.stage.ser is None:
            return -1
        self.stage.calibrate()

    def check_ready(self, wait_count):
        count_num = 0
        # print("Now Moving ...")
        while count_num < wait_count:
            count_num += 1
            if not self.stage.isReady():
                while True:
                    count_num += 1
                    if self.stage.isReady():
                        break
        # print("Ready")

    @staticmethod
    def prepare_next(dis):
        print("Finish ", str(dis), ' [m]')

    def main(self, directions, distances):
        self.Init_set()
        self.check_ready(200)
        # time.sleep(30)
        print("Finish Robot, Mic, Turn Table Set UP")
        print("#################################################")
        # self.func.my_makedirs(self.photo_path + '/' + str(dis))
        self.Turn_start(directions[0] + self.adjust)
        for i, deg in enumerate(directions):
            self.check_ready(200)
            time.sleep(2)
            self.Turn_fin(deg)
            self.Rec_fin()
            # print("******************************************")
            if i + 1 < len(directions):
                self.Turn_res(directions[i + 1] + self.adjust)
                # print("OK")
        print()
        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        self.socket_close()
        self.stage.move_table(0 + self.adjust)
        print('Finish Experiment (^^)/ !!')

    @staticmethod
    def make_dir(order_num):
        directions = [order_num]
        for i in range(4):
            order_num = [k - 1 for k in order_num]
            directions.append(order_num)
        directions = np.array(directions).reshape(1, -1)[0]
        directions = np.append(directions, -50)
        # print(np.array(DIRECTIONS).reshape(1, -1)[0])
        print(directions)
        return directions

    def socket_close(self):
        # self.socket.close()
        print('Disconnect Socket Communication')

    @staticmethod
    def transform_pos_to_command(x_pos=0.0, y_pos=0.0, z_pos=0.0):
        if y_pos > 0:
            print('ERROR')
            sys.exit()
        else:
            return 'x=' + str(x_pos) + ',y=' + str(y_pos) + ',z=' + str(z_pos)


if __name__ == '__main__':
    st = StateMachine("State")
    # repeat_num = 10
    DISTANCES = [0.4]
    '''Experiment'''
    # order = np.arange(-45, 51, 5)
    # DIRECTIONS = st.make_dir(order)
    # st.main(DIRECTIONS, DISTANCES)
    '''for test'''
    order = [-2]
    st.main(order, DISTANCES)
