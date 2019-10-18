# coding:utf-8
import numpy as np
import time
from datetime import datetime
from transitions import Machine
from _turntable_controler import StageControl
from _recode_func import RecodeFunc
from multiprocessing import Pool


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
        self.scon = StageControl()
        self.rfnc = RecodeFunc()
        self.mic_index, self.mic_channels = self.rfnc.get_index('ReSpeaker 4 Mic Array (UAC1.0) ')
        # self.speak_wf, self.speak_stream, self.speak_p = self.rfnc.sound_read("./origin_sound_data/tsp_1num.wav")
        self.speak_path = "../_exp/Speaker_Sound/2_up_tsp_8num.wav"
        self.mic_path = '../_exp/19' + datetime.today().strftime("%m%d") + '/recode_data/' + \
                        datetime.today().strftime("%H%M%S") + "/"
        self.adjust = 50

    def turn_table(self, name):
        print(name-self.adjust, "Setting ... ")
        # State.scon.gSend("D:1S5000F100000R600")
        self.scon.move_table(name)

    def wait(self, sleep_time=2):
        # print("wait")
        time.sleep(sleep_time)
        pass

    def recode(self, name):
        # print("recode", name)
        recode_sound = 8.5
        pool = Pool(4)
        pool.apply_async(self.rfnc.sound_recode, (self.mic_path, name, recode_sound, self.mic_channels, self.mic_index))
        self.rfnc.sound_out(self.speak_path)

    def init(self):
        if self.scon.ser is None:
            return -1
        self.scon.calibrate()



def check_ready(wait_count):
    count_num = 0
    print("Now Moving ...")
    while count_num < wait_count:
        count_num += 1
        if not st.scon.isReady():
            while True:
                count_num += 1
                if st.scon.isReady():
                    break
    print("Ready")


if __name__ == '__main__':
    st = StateMachine("State")
    order = [50, 40, 30, 20, 10, 0, -10, -20, -30, -40]
    # order = [0]
    DIRECTIONS = [order]
    for i in range(10):
        order = [k - 1 for k in order]
        DIRECTIONS.append(order)
    DIRECTIONS = np.array(DIRECTIONS).reshape(1, -1)[0]
    # print(np.array(DIRECTIONS).reshape(1, -1)[0])
    st.Init_set()
    check_ready(200)
    time.sleep(30)
    print("Finish Calibration of Turn Table")
    print("#################################################")
    st.Turn_start(DIRECTIONS[0] + st.adjust)
    for i, deg in enumerate(DIRECTIONS):
        check_ready(200)
        time.sleep(2)
        st.Turn_fin(deg)
        st.Rec_fin()
        print("******************************************")
        if i + 1 < len(DIRECTIONS):
            st.Turn_res(DIRECTIONS[i + 1] + st.adjust)

    st.scon.move_table(0 + st.adjust)
