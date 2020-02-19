import socket
import time
from datetime import datetime
import wave
# from _recode_func import RecodeFunc

HOST_IP_1 = '163.221.126.64'  # 接続するサーバーのIPアドレス
HOST_IP_2 = "163.221.44.120"
PORT_1 = 12345  # 接続するサーバーのポート
PORT_2 = 12345  # 接続するサーバーのポート
DATE_SIZE = 1024  # 受信データバイト数
INTERVAL = 3  # ソケット接続時のリトライ待ち時間
RETRY_TIMES = 5  # ソケット接続時のリトライ回数


class SocketClient:
    def __init__(self, host_1, host_2, port_1, port_2, data_size, interval, retry_times):
        self.host_1 = host_1
        self.host_2 = host_2
        self.port_1 = port_1
        self.port_2 = port_2
        self.data_size = data_size
        self.interval = interval
        self.retry_times = retry_times
        self.socket_1 = None
        self.socket_2 = None

    def connect(self):
        client_socket_1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket_2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        for x in range(RETRY_TIMES):  # RETRYTIMESの回数だけリトライ
            try:
                client_socket_1.connect((self.host_1, self.port_1))  # サーバーとの接続
                self.socket_1 = client_socket_1
                print('[{0}] server connect -> address : {1}:{2}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                                                         self.host_1, self.port_1))
                break
            except socket.error:
                print('[{0}] retry after wait{1}s'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), str(INTERVAL)))
                time.sleep(INTERVAL)  # 接続を確立できない場合、INTERVAL秒待ってリトライ

        for x in range(RETRY_TIMES):  # RETRYTIMESの回数だけリトライ
            try:
                client_socket_2.connect((self.host_2, self.port_2))  # サーバーとの接続
                self.socket_2 = client_socket_2
                print('[{0}] server connect -> address : {1}:{2}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                                                         self.host_2, self.port_2))
                break
            except socket.error:
                print('[{0}] retry after wait{1}s'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), str(INTERVAL)))
                time.sleep(INTERVAL)  # 接続を確立できない場合、INTERVAL秒待ってリトライ

    def send(self, socket_no, in_data):  # サーバーへデータ送信関数
        # input_data = input("send data:")  # ターミナルから入力された文字を取得
        # print('[{0}] input data : {1}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), input_data))
        # input_data = input_data.encode('utf-8')
        in_data = in_data.encode('utf-8')
        print(in_data)
        if socket_no == 1:
            self.socket_1.send(in_data)  # データ送信
        else:
            self.socket_2.send(in_data)

    def recv(self, socket_no):  # サーバーからデータ受信関数
        if socket_no == 1:
            rcv_data = self.socket_1.recv(DATE_SIZE)  # データ受信
        else:
            rcv_data = self.socket_2.recv(DATE_SIZE)  # データ受信
        rcv_data = rcv_data.decode('utf-8')
        # print('[{0}] recv data : {1}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), rcv_data))
        return rcv_data

    def send_rcv(self, in_data):
        self.send(1, in_data)
        return self.recv(1)

    def send_GO(self, out_put_name):
        in_data = 'go'
        in_data = in_data.encode('utf-8')
        # print(in_data)
        self.socket_1.send(in_data)  # データ送信
        recode_data = []
        # rcv_data = self.socket.recv(DATE_SIZE)  # データ受信
        # recode_data.append(rcv_data)
        while True:
            rcv_data = self.socket_1.recv(DATE_SIZE)  # データ受信
            # print('[{0}] recv data : {1}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), rcv_data))
            if rcv_data == in_data:
                # print('[{0}] recv data : {1}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), rcv_data.decode('utf-8')))
                break
            else:
                recode_data.append(rcv_data)
        # print('OK')
        recode_data = b''.join(recode_data)
        self.wave_save(recode_data, channels=8, wave_file=out_put_name)

    def send_INIT(self):
        in_data = 'initialize'
        in_data = in_data.encode('utf-8')
        # print(in_data)
        self.socket_2.send(in_data)  # データ送信
        print('-----------------------------------------------')
        print("Setting Robot")
        while True:
            if self.recv(2) == 'done_initialization':
                print("Set Up Robot")
                self.socket_2.close()  # ソケットクローズ
                self.socket_2 = None
                break

    def send_SET_DIS(self, dis):
        self.reconnect()
        in_data = str(dis)
        in_data = in_data.encode('utf-8')
        # print(in_data)
        self.socket_2.send(in_data)  # データ送信
        print()
        print('---------------------------------------------')
        while True:
            if self.recv(2) == 'SET_OK' + '_' + str(dis):
                print(str(dis), "[mm] Set")
                print('↓↓↓')
                self.socket_2.close()  # ソケットクローズ
                self.socket_2 = None
                break

    def send_FINI(self):
        self.reconnect()
        in_data = 'finalize'
        in_data = in_data.encode('utf-8')
        # print(in_data)
        self.socket_2.send(in_data)  # データ送信
        print('-----------------------------------------------')
        print("Shut Down Robot")
        while True:
            if self.recv(2) == 'done_finalization':
                print("Finish Shut Down Robot")
                break

    def close(self):
        self.socket_1.close()  # ソケットクローズ
        self.socket_1 = None
        self.send_FINI()
        self.socket_2.close()  # ソケットクローズ
        self.socket_2 = None

    def reconnect(self):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        for x in range(RETRY_TIMES):  # RETRYTIMESの回数だけリトライ
            try:
                client_socket.connect((self.host_2, self.port_2))  # サーバーとの接続
                self.socket_2 = client_socket
                print('[{0}] server connect -> address : {1}:{2}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                                                       self.host_2, self.port_2))
                break
            except socket.error:
                print('[{0}] retry after wait{1}s'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                                          str(INTERVAL)))
                time.sleep(INTERVAL)  # 接続を確立できない場合、INTERVAL秒待ってリトライ

    @staticmethod
    def wave_save(data, channels=1, width=2, sampling=44100, wave_file='./out_put.wav'):
        wf = wave.Wave_write(wave_file)
        wf.setnchannels(channels)
        wf.setsampwidth(width)
        wf.setframerate(sampling)
        wf.writeframes(data)
        wf.close()

