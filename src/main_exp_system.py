# coding:utf-8
import numpy as np
import time
from datetime import datetime
from transitions import Machine
from _turntable_controler import StageControl
from _recode_func import RecodeFunc
from multiprocessing import Pool
from _function import MyFunc


class StateMachine(MyFunc):
    states = ['Turn', 'Recode', 'Wait', 'Init', 'Next']

    def __init__(self, name, out_file, recode_second):
        super().__init__()
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
        self.mic_index, self.mic_channels = self.rfnc.get_index('ReSpeaker 4 Mic Array (UAC1.0) ')
        # self.speak_wf, self.speak_stream, self.speak_p = self.rfnc.sound_read("./origin_sound_data/tsp_1num.wav")
        # self.speak_path = "../_exp/Speaker_Sound/1_plus005_4time.wav
        self.speak_path = self.speaker_sound_path + out_file
        self.mic_path = '../_exp/20' + datetime.today().strftime("%m%d") + '/' + datetime.today().strftime("%H%M%S") + '/'
        self.my_makedirs(self.mic_path)
        self.adjust = 50
        self.recode_second = recode_second
        self.mic_name = 'ReSpeaker 4 Mic Array (UAC1.0)'

    def turn_table(self, name):
        print(name-self.adjust, "Setting ... ")
        # State.scon.gSend("D:1S5000F100000R600")
        self.scon.move_table(name)

    def wait(self, sleep_time=2):
        # print("wait")
        time.sleep(sleep_time)
        pass

    def recode(self, name, num=0):
        # print("recode", name)
        # recode_sound = 20.5
        # recode_sound = 1.2
        # pool = Pool(4)
        output_name = self.mic_path + str(name) + '.wav'
        recode_data, sampling = self.rfnc.play_rec(self.speak_path, self.recode_second, device_name=self.mic_name,
                                                   input_file_name=output_name, need_data=True,
                                                   order_index=self.mic_index, order_ch=self.mic_channels)
        #
        # if num == 0:
        #     pool.apply_async(self.rfnc.sound_recode,
        #                      (self.mic_path + "/", name, recode_sound, self.mic_channels, self.mic_index))
        # else:
        #     pool.apply_async(self.rfnc.sound_recode,
        #                      (self.mic_path + "_" + str(num) + "/", name, recode_sound, self.mic_channels, self.mic_index))
        # self.rfnc.sound_out(self.speak_path)

    def init(self):
        if self.scon.ser is None:
            return -1
        self.scon.calibrate()

    def check_ready(self, wait_count):
        count_num = 0
        # print("Now Moving ...")
        while count_num < wait_count:
            count_num += 1
            if not self.scon.isReady():
                while True:
                    count_num += 1
                    if self.scon.isReady():
                        break
        # print("Ready")

    def main(self, DIRECTIONS, num=0):
        self.Init_set()
        self.check_ready(200)
        if len(DIRECTIONS) > 3:
            time.sleep(10)
        print("Finish Calibration of Turn Table")
        print("#################################################")
        self.Turn_start(DIRECTIONS[0] + self.adjust)
        for i, deg in enumerate(DIRECTIONS):
            self.check_ready(200)
            time.sleep(2)
            self.Turn_fin(deg, num)
            self.Rec_fin()
            # print("******************************************")
            if i + 1 < len(DIRECTIONS):
                self.Turn_res(DIRECTIONS[i + 1] + self.adjust)
                # print("OK")

        self.scon.move_table(0 + self.adjust)
        self.scon.ser.close()
        print("serial connection closed")

    @staticmethod
    def make_dir(order):
        DIRECTIONS = [order]
        for i in range(4):
            order = [k - 1 for k in order]
            DIRECTIONS.append(order)
        DIRECTIONS = np.array(DIRECTIONS).reshape(1, -1)[0]
        DIRECTIONS = np.append(DIRECTIONS, -50)
        # print(np.array(DIRECTIONS).reshape(1, -1)[0])
        return DIRECTIONS


if __name__ == '__main__':
    out_file = ['2_up_tsp_8num', 'torn_8num_3000Hz']
    second = [8.5, 9.0]
    # freq = [1000, 2000, 3000]
    for i, out in enumerate(out_file):
        st = StateMachine("State", out + '.wav', second[i])
        # repeat_num = 10
        DIRECTIONS = [50, 40, 30, 20, 10, 0]
        # order = np.arange(-45, 51, 5)
        # DIRECTIONS = st.make_dir(order)
        # DIRECTIONS = [-30, 0]
        # DIRECTIONS = np.arange(-4, 46, 5)
        # DIRECTIONS = np.append(DIRECTIONS, -50)
        print(DIRECTIONS)
        # print(np.array(DIRECTIONS).reshape(1, -1)[0])
        # for i in range(repeat_num):
        st.main(DIRECTIONS)


