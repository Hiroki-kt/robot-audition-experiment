import socket
import time
from datetime import datetime

HOST_IP_1 = "163.221.44.120"  # 接続するサーバーのIPアドレス
HOST_IP_2 = '163.221.126.64'
PORT_1 = 12345  # 接続するサーバーのポート
PORT_2 = 12345  # 接続するサーバーのポート
DATE_SIZE = 1024  # 受信データバイト数
INTERVAL = 3  # ソケット接続時のリトライ待ち時間
RETRY_TIMES = 5  # ソケット接続時のリトライ回数


class SocketClient:
    def __init__(self, host_1, host_2, port_1, port_2):
        self.host_1 = host_1
        self.host_2 = host_2
        self.port_1 = port_1
        self.port_2 = port_2
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

    def send(self):  # サーバーへデータ送信関数
        input_data = input("send data:")  # ターミナルから入力された文字を取得
        print('[{0}] input data : {1}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), input_data))
        input_data = input_data.encode('utf-8')
        self.socket_1.send(input_data)  # データ送信
        time.sleep(5)
        self.socket_2.send(input_data)

    def recv(self):  # サーバーからデータ受信関数
        rcv_data_1 = self.socket_1.recv(DATE_SIZE)  # データ受信
        rcv_data_1 = rcv_data_1.decode('utf-8')
        print('[{0}] recv data : {1}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), rcv_data_1))
        time.sleep(5)
        rcv_data_2 = self.socket_2.recv(DATE_SIZE)
        rcv_data_2 = rcv_data_2.decode('utf-8')
        print('[{0}] recv data : {1}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), rcv_data_2))
        return rcv_data_1, rcv_data_2

    def send_rcv(self):
        self.send()
        return self.recv()

    def close(self):
        self.socket_1.close()  # ソケットクローズ
        self.socket_1 = None


if __name__ == '__main__':
    client = SocketClient(HOST_IP_1, HOST_IP_2, PORT_1, PORT_2)
    client.connect()  # はじめの1回だけソケットをオープン

    while True:
        if client.socket_1 is not None:
            if client.send_rcv() == 'quit':  # quitが戻ってくるとソケットをクローズして終了
                client.close()
        else:
            break
