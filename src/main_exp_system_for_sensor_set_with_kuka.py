# coding:utf-8
import numpy as np
import time
from datetime import datetime
from transitions import Machine
from _turntable_controler import StageControl
# from _recode_func import RecodeFunc
# from multiprocessing import Pool
from _socket_multi_crient import SocketClient
from _function import MyFunc
from _capture_photo import TakePhoto
import sys


class StateMachine(object):
    states = ['Turn', 'Recode', 'Wait', 'Init', 'Next']

    def __init__(self, name, host_ip_1, host_ip_2, port_1, port_2, send_size, interval, retry_times):
        self.name = name
        self.machine = Machine(model=self, states=StateMachine.states, initial='Wait', auto_transitions=False)
        self.machine.add_transition(trigger='Turn_start', source='Init', dest='Turn', after='turn_table')
        self.machine.add_transition(trigger='Turn_fin', source='Turn', dest='Recode', before='recode')
        # self.machine.add_transition(trigger='Turn_fin', source='Init', dest='Recode', before='recode')
        self.machine.add_transition(trigger='Rec_fin', source='Recode', dest='Wait', after='wait')
        # self.machine.add_transition(trigger='Rec_fin', source='Recode', dest='Init', after='wait')
        self.machine.add_transition(trigger='Init_set', source='Wait', dest='Init', before='init')
        self.machine.add_transition(trigger='Turn_res', source='Wait', dest='Turn', after='turn_table')
        self.machine.add_transition(trigger='Next_dis', source='Wait', dest='Init', after='prepare_next')
        self.stage = StageControl()
        # self.rfnc = RecodeFunc()
        self.func = MyFunc()
        self.photo = TakePhoto()
        self.socket = SocketClient(host_ip_1, host_ip_2, port_1, port_2, send_size, interval, retry_times)
        # self.mic_index, self.mic_channels = self.rfnc.get_index('ReSpeaker 4 Mic Array (UAC1.0) ')
        # self.speak_wf, self.speak_stream, self.speak_p = self.rfnc.sound_read("./origin_sound_data/tsp_1num.wav")
        # self.speak_path = "../_exp/Speaker_Sound/1_plus005_4time.wav"
        self.speak_path = "../_exp/Speaker_Sound/2_up_tsp_8num.wav"
        self.mic_path = self.func.make_dir_path(exp=True)
        self.photo_path = self.func.make_dir_path(photo=True)
        # self.func.my_makedirs(self.mic_path)
        self.adjust = 50

    def turn_table(self, name):
        # State.scon.gSend("D:1S5000F100000R600")
        self.stage.move_table(name)
        print(name-self.adjust, ' [deg]')

    @staticmethod
    def wait(sleep_time=2):
        # print("wait")
        time.sleep(sleep_time)
        pass

    def recode(self, dir_name, dis_name):
        if self.socket.socket_1 is not None:
            self.photo.capture(self.photo_path + '/' + str(dis_name) + '/' + str(dir_name))
            self.socket.send_GO(self.mic_path + '/' + str(dis_name) + '/' + str(dir_name) + '.wav')

    def init(self):
        if self.stage.ser is None:
            return -1
        self.stage.calibrate()
        self.socket.connect()
        self.socket.send_INIT()

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
        for j, dis in enumerate(distances):
            if j != 0:
                self.socket.send_SET_DIS(self.transform_pos_to_command(y_pos=round(distances[j-1]- dis, 2)))
            self.func.my_makedirs(self.mic_path + '/' + str(dis))
            self.func.my_makedirs(self.photo_path + '/' + str(dis))
            self.Turn_start(directions[0] + self.adjust)
            for i, deg in enumerate(directions):
                self.check_ready(200)
                time.sleep(2)
                self.Turn_fin(deg, dis)
                self.Rec_fin()
                # print("******************************************")
                if i + 1 < len(directions):
                    self.Turn_res(directions[i + 1] + self.adjust)
                    # print("OK")
                else:
                    self.Next_dis(dis)
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
        self.socket.close()
        print('Disconnect Socket Communication')

    @staticmethod
    def transform_pos_to_command(x_pos=0.0, y_pos=0.0, z_pos=0.0):
        if y_pos > 0:
            print('ERROR')
            sys.exit()
        else:
            return 'x=' + str(x_pos) + ',y=' + str(y_pos) + ',z=' + str(z_pos)


if __name__ == '__main__':
    HOST_IP_1 = "163.221.44.120"  # 接続するサーバーのIPアドレス
    # HOST_IP_2 = '163.221.44.237'
    HOST_IP_2 = '163.221.44.222'
    PORT_1 = 12345  # 接続するサーバーのポート
    PORT_2 = 12345  # 接続するサーバーのポート
    DATE_SIZE = 1024  # 受信データバイト数
    INTERVAL = 3  # ソケット接続時のリトライ待ち時間
    RETRY_TIMES = 5  # ソケット接続時のリトライ回数
    st = StateMachine("State", HOST_IP_1, HOST_IP_2, PORT_1, PORT_2, DATE_SIZE, INTERVAL, RETRY_TIMES)
    # repeat_num = 10
    DISTANCES = [0.4]
    '''Experiment'''
    # order = np.arange(-45, 51, 5)
    # DIRECTIONS = st.make_dir(order)
    # st.main(DIRECTIONS, DISTANCES)
    '''for test'''
    order = [-2]
    st.main(order, DISTANCES)
