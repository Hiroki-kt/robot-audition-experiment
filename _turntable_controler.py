# coding:utf-8
import time
import serial
from serial.tools import list_ports
import sys
from msvcrt import getch


class StageControl:
    def __init__(self):
        self.ser = self.select_port
        # print(self.ser)

    def isReady(self):
        while self.ser.is_open:
            flag = self.gCheck("!:")
            # print(flag)
            if flag == "R":
                return True
            else:
                return False

    def gCheck(self, command):
        while self.ser.is_open:
            s = str(command) + '\r\n'  # CRLF
            sb = s.encode('utf-8')  # Byte
            self.ser.write(sb)
            s = self.ser.readline()
            if s != '':
                rep = s.decode().strip()
                return rep
            time.sleep(0.5)

    def gSend(self,  command, dtime=0.1):
        # print("Send", command)
        while self.ser.is_open:
            if self.isReady():
                s = str(command) + '\r\n'  # CRLF
                sb = s.encode('utf-8')  # Byte
                # print(sb.decode(), end='')
                self.ser.write(sb)
                s = self.ser.readline()
                time.sleep(dtime)
                if s != '':
                    rep = s.decode().strip()
                    if rep == 'OK':
                        # print(rep)
                        return rep
                else:
                    print(".")

    @property
    def select_port(self):
        ser = serial.Serial()
        ser.baudrate = 9600
        ser.timeout = 0

        ports = list_ports.comports()
        devices = [info.device for info in ports]

        if len(devices) == 0:
            print("error: device not found")
            return None
        elif len(devices) == 1:
            print("Only Found %s" % devices[0])
            ser.port = devices[0]
        else:
            for i in range(len(devices)):
                print("input %3d: open %s" % (i, devices[i]))
            print("input number of target port >> ", end="")
            num = int(input())
            ser.port = devices[num]

        try:
            ser.open()
            print("Complete Turn Table Setting: Port " + ser.port)
            return ser
        except:
            print("error when opening serial")
            return None

    def sendA(self, x, n="1", m="+"):
        msg = "A:" + n + m + "P" + str(x)
        self.gSend(msg)

    def calibrate(self):
        self.gSend("V:1,8000,100000,600,50000")
        self.gSend("H:1")
        self.gSend("D:1S5000F100000R600")

    def move_table(self, deg):
        deg_per_plus = 72000 * 20 / 360
        self.gSend("C:11")  # モータON
        self.sendA(int(deg * deg_per_plus))
        self.gSend("G:", )  # 実行

    def main(self):

        if self.ser is None:
            return

        # 初期設定 V:原点復帰速度 H:原点復帰 D:移動速度設定
        self.gSend("V:1,8000,100000,600,50000")
        self.gSend("H:1")
        self.gSend("D:1S5000F100000R600")

        deg_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        deg_list = [22, 32, 42, 24]
        deg_list = [k + 50 for k in deg_list]
        deg_per_plus = 72000 * 20 / 360

        NEXT = 110  # n
        TEN_SKIP = 78  # N
        EXIT = 3  # ctrl c

        while True:
            print("First position set up finish!, please press 「n」")
            key = ord(getch())
            if key == NEXT:  # ctrl c
                print("NEXT")
                break
            else:
                message = 'input, {0}'.format(chr(key))
                print("You mistake, Please retake", message)
        try:
            while self.ser.isOpen:
                for i, deg in enumerate(deg_list):
                    print(deg)
                    wait_time = 3.0
                    self.gSend("C:11", wait_time)  # モータON
                    self.sendA(int(deg * deg_per_plus))
                    self.gSend("G:", )  # 実行
                    print("***************************************************")
                    print("IF you do next 10 deg! Please press key n")
                    print("IF you do New step! Please press key N")
                    print("If you exit this script, PLease press key ctrl+c")
                    print("***************************************************")
                    key = ord(getch())
                    if key == TEN_SKIP:
                        print("Skip the Next ten step 1~10")
                        break
                    elif key == EXIT:
                        print("Program shut down")
                        sys.exit()
                    else:
                        while True:
                            if key == NEXT:  # n
                                print("NEXT")
                                break
                            else:
                                message = 'input, {0}'.format(chr(key))
                                print("You mistake!! Please Retake", message)
                                key = ord(getch())
                deg_list = [k + 1 for k in deg_list]
                # self.gSend("C:11", 5.0)
        except KeyboardInterrupt:
            self.ser.close()
            print("serial connection closed")
            sys.exit(0)


if __name__ == "__main__":
    stage = StageControl()
    stage.main()