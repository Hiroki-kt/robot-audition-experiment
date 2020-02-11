import socket
import time
from datetime import datetime

HOST_IP = "163.221.44.237"  # 接続するサーバーのIPアドレス
PORT = 12345  # 接続するサーバーのポート
DATE_SIZE = 1024  # 受信データバイト数
INTERVAL = 3  # ソケット接続時のリトライ待ち時間
RETRY_TIMES = 5  # ソケット接続時のリトライ回数


class SocketClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
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

    def send(self):  # サーバーへデータ送信関数
        input_data = input("send data:")  # ターミナルから入力された文字を取得
        print('[{0}] input data : {1}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), input_data))
        input_data = input_data.encode('utf-8')
        self.socket.send(input_data)  # データ送信

    def recv(self):  # サーバーからデータ受信関数
        rcv_data = self.socket.recv(DATE_SIZE)  # データ受信
        rcv_data = rcv_data.decode('utf-8')
        print('[{0}] recv data : {1}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), rcv_data))
        return rcv_data

    def send_rcv(self):
        self.send()
        return self.recv()

    def close(self):
        self.socket.close()  # ソケットクローズ
        self.socket = None


if __name__ == '__main__':
    client = SocketClient(HOST_IP, PORT)
    client.connect()  # はじめの1回だけソケットをオープン

    while True:
        if client.socket is not None:
            if client.send_rcv() == 'quit':  # quitが戻ってくるとソケットをクローズして終了
                client.close()
        else:
            break
