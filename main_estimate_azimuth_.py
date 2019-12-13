from _recode_func import RecodeFunc
from multiprocessing import Pool
from datetime import datetime
import joblib
import sounddevice as sd


class Estimate(RecodeFunc):
    def __init__(self):
        self.mic_path = '../_exp/19' + datetime.today().strftime("%m%d") + '/recode_data/' + \
                        datetime.today().strftime("%H%M%S")
        self.mic_index, self.mic_channels = self.get_index('ReSpeaker 4 Mic Array (UAC1.0) ')
        self.speak_path = "../_exp/Speaker_Sound/2_up_tsp_1num.wav"
        self.model = joblib.load('../../../../OneDrive/Research/Model/191125_anechonic/anechonic_svr_model.pkl')

    def recode(self, name, num=0):
        # print("recode", name)
        # recode_sound = 20.5
        recode_sound = 1.5
        pool = Pool(4)
        pool.apply_async(self.sound_recode, (self.mic_path + "/", name, recode_sound, self.mic_channels, self.mic_index))
        pool.apply_async(self.sound_out, self.speak_path)
        # self.sound_recode(self.mic_path + "/", name, recode_sound, self.mic_channels, self.mic_index)

    def estimate(self, data):
        print(self.model.shape)


if __name__ == '__main__':
    es = Estimate()
    es.recode('test')
