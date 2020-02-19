import socket
import time
from datetime import datetime
import wave
# from _recode_func import RecodeFunc

HOST_IP = "163.221.139.120"  # 接続するサーバーのIPアドレス
# HOST_IP = '163.221.126.64'
PORT = 12345  # 接続するサーバーのポート
DATE_SIZE = 1024  # 受信データバイト数
INTERVAL = 3  # ソケット接続時のリトライ待ち時間
RETRY_TIMES = 5  # ソケット接続時のリトライ回数


class SocketClient:
    def __init__(self, host, port, data_size, interval, retry_times):
        self.host = host
        self.port = port
        self.data_size = data_size
        self.interval = interval
        self.retry_times = retry_times
        self.socket = None

    def connect(self):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        for x in range(RETRY_TIMES):  # RETRYTIMESの回数だけリトライ
            try:
                client_socket.connect((self.host, self.port))  # サーバーとの接続
                self.socket = client_socket
                print('[{0}] server connect -> address : {1}:{2}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                                                         self.host, self.port))
                break
            except socket.error:
                print('[{0}] retry after wait{1}s'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), str(INTERVAL)))
                time.sleep(INTERVAL)  # 接続を確立できない場合、INTERVAL秒待ってリトライ

    def send(self, in_data):  # サーバーへデータ送信関数
        # input_data = input("send data:")  # ターミナルから入力された文字を取得
        # print('[{0}] input data : {1}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), input_data))
        # input_data = input_data.encode('utf-8')
        in_data = in_data.encode('utf-8')
        print(in_data)
        self.socket.send(in_data)  # データ送信

    def recv(self):  # サーバーからデータ受信関数
        rcv_data = self.socket.recv(DATE_SIZE)  # データ受信
        rcv_data = rcv_data.decode('utf-8')
        print('[{0}] recv data : {1}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), rcv_data))
        return rcv_data

    def send_rcv(self, in_data):
        self.send(in_data)
        return self.recv()

    def send_GO(self, out_put_name):
        in_data = 'go'
        in_data = in_data.encode('utf-8')
        # print(in_data)
        self.socket.send(in_data)  # データ送信
        recode_data = []
        # rcv_data = self.socket.recv(DATE_SIZE)  # データ受信
        # recode_data.append(rcv_data)
        while True:
            rcv_data = self.socket.recv(DATE_SIZE)  # データ受信
            # print('[{0}] recv data : {1}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), rcv_data))
            if rcv_data == in_data:
                print('[{0}] recv data : {1}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), rcv_data))
                break
            else:
                recode_data.append(rcv_data)
        print('OK')
        recode_data = b''.join(recode_data)
        self.wave_save(recode_data, channels=8, wave_file=out_put_name)

    def close(self):
        self.socket.close()  # ソケットクローズ
        self.socket = None

    @staticmethod
    def wave_save(data, channels=1, width=2, sampling=44100, wave_file='./out_put.wav'):
        wf = wave.Wave_write(wave_file)
        wf.setnchannels(channels)
        wf.setsampwidth(width)
        wf.setframerate(sampling)
        wf.writeframes(data)
        wf.close()
