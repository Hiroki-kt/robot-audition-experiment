# coding:utf-8
import numpy as np
import time
from datetime import datetime
from transitions import Machine
from _turntable_controler import StageControl
from _recode_func import RecodeFunc
from multiprocessing import Pool
from _socket_crient import SocketClient


class StateMachine(object):
    states = ['Turn', 'Recode', 'Wait', 'Init', 'Next']

    def __init__(self, name, host_ip, port_num, send_size, interval, retry_times):
        self.name = name
        self.machine = Machine(model=self, states=StateMachine.states, initial='Wait', auto_transitions=False)
        self.machine.add_transition(trigger='Turn_start', source='Init', dest='Turn', after='turn_table')
        self.machine.add_transition(trigger='Turn_fin', source='Turn', dest='Recode', before='recode')
        # self.machine.add_transition(trigger='Turn_fin', source='Init', dest='Recode', before='recode')
        self.machine.add_transition(trigger='Rec_fin', source='Recode', dest='Wait', after='wait')
        # self.machine.add_transition(trigger='Rec_fin', source='Recode', dest='Init', after='wait')
        self.machine.add_transition(trigger='Init_set', source='Wait', dest='Init', before='init')
        self.machine.add_transition(trigger='Turn_res', source='Wait', dest='Turn', after='turn_table')
        self.scon = StageControl()
        self.rfnc = RecodeFunc()
        self.socl = SocketClient(host_ip, port_num, send_size, interval, retry_times)
        self.mic_index, self.mic_channels = self.rfnc.get_index('ReSpeaker 4 Mic Array (UAC1.0) ')
        # self.speak_wf, self.speak_stream, self.speak_p = self.rfnc.sound_read("./origin_sound_data/tsp_1num.wav")
        # self.speak_path = "../_exp/Speaker_Sound/1_plus005_4time.wav"
        self.speak_path = "../_exp/Speaker_Sound/2_up_tsp_8num.wav"
        self.mic_path = '../_exp/20' + datetime.today().strftime("%m%d") + '/recode_data/' + \
                        datetime.today().strftime("%H%M%S")
        self.rfnc.my_makedirs(self.mic_path)
        self.adjust = 50

    def turn_table(self, name):
        print(name-self.adjust, "Setting ... ")
        # State.scon.gSend("D:1S5000F100000R600")
        self.scon.move_table(name)

    def wait(self, sleep_time=2):
        # print("wait")
        time.sleep(sleep_time)
        pass

    def recode(self, name, num=0):
        if self.socl.socket is not None:
            self.socl.send_GO(self.mic_path + '/' + str(name) + '.wav')

    def init(self):
        if self.scon.ser is None:
            return -1
        self.scon.calibrate()
        self.socl.connect()

    def check_ready(self, wait_count):
        count_num = 0
        print("Now Moving ...")
        while count_num < wait_count:
            count_num += 1
            if not self.scon.isReady():
                while True:
                    count_num += 1
                    if self.scon.isReady():
                        break
        print("Ready")

    def main(self, DIRECTIONS, num=0):
        self.Init_set()
        self.check_ready(200)
        # time.sleep(30)
        print("Finish Calibration of Turn Table")
        print("#################################################")
        self.Turn_start(DIRECTIONS[0] + self.adjust)
        for i, deg in enumerate(DIRECTIONS):
            self.check_ready(200)
            time.sleep(2)
            self.Turn_fin(deg, num)
            self.Rec_fin()
            print("******************************************")
            if i + 1 < len(DIRECTIONS):
                self.Turn_res(DIRECTIONS[i + 1] + self.adjust)
                print("OK")

        self.scon.move_table(0 + self.adjust)

    @staticmethod
    def make_dir(order):
        DIRECTIONS = [order]
        for i in range(10):
            order = [k - 1 for k in order]
            DIRECTIONS.append(order)
        DIRECTIONS = np.array(DIRECTIONS).reshape(1, -1)[0]
        # print(np.array(DIRECTIONS).reshape(1, -1)[0])
        return DIRECTIONS


if __name__ == '__main__':
    HOST_IP = "163.221.44.239"  # 接続するサーバーのIPアドレス
    PORT = 12345  # 接続するサーバーのポート
    DATE_SIZE = 1024  # 受信データバイト数
    INTERVAL = 3  # ソケット接続時のリトライ待ち時間
    RETRY_TIMES = 5  # ソケット接続時のリトライ回数
    st = StateMachine("State", HOST_IP, PORT, DATE_SIZE, INTERVAL, RETRY_TIMES)
    # repeat_num = 10
    '''Experiment'''
    order = [50, 40, 30, 20, 10, 0, -10, -20, -30, -40]
    DIRECTIONS = st.make_dir(order)
    st.main(DIRECTIONS)
    '''for test'''
    # order = [0, -30]
    # st.main(order)
