# -*- coding: utf_8 -*-
"""
 Modbus TestKit: Implementation of Modbus protocol in python

 (C)2009 - Luc Jean - luc.jean@gmail.com
 (C)2009 - Apidev - http://www.apidev.fr

 This is distributed under GNU LGPL license, see license.txt
"""

import serial

import modbus_tk
import modbus_tk.defines as cst
from modbus_tk import modbus_rtu

from time import sleep

#PORT = '/dev/ttyp5'

class PconCB(object):
    """
    リニアアクチュエータ[iai製 RCP4]をmodbusプロトコルを使って制御するためのクラス
    """

    # ロガーとマスターの設定を初期化
    def __init__(self, port="COM3", baudrate=38400):
        # ロガーをコンソール表示設定で初期化
        self.logger = modbus_tk.utils.create_logger("console")
        #Connect to the slave
        self.master = modbus_rtu.RtuMaster(
            serial.Serial(port=port, baudrate=baudrate, bytesize=8, parity='N', stopbits=1, xonxoff=0)
        )
        self.master.set_timeout(5.0)
        self.master.set_verbose(False)
        self.logger.info("connected")
        sleep(2)
        self.initialize()


    # 終了時の処理
    def close(self):
        # サーボをオフ
        self.logger.info(self.master.execute(1, cst.WRITE_SINGLE_COIL, int("0x0403", 16), output_value=int("0x0000", 16)))

    # PCON CB側の初期化設定＆原点復帰動作
    def initialize(self):
        # セーフティーモードを有効に設定
        self.logger.info(self.master.execute(1, cst.WRITE_SINGLE_COIL, int("0x0401", 16), output_value=int("0xFF00", 16)))
        sleep(0.5)
        # サーボをオン
        self.logger.info(self.master.execute(1, cst.WRITE_SINGLE_COIL, int("0x0403", 16), output_value=int("0xFF00", 16)))
        sleep(0.5)
        self.disp_current_moving_state()
        self.move_to_regression_point()

    # 現在の位置を取得
    def get_current_position(self):
        # 現在位置の読み取り
        ret = self.master.execute(1, cst.READ_HOLDING_REGISTERS, int("0x9000", 16), int("0x0002", 16))
        sleep(0.05)
        current_pos = int(ret[0]*2^8 + ret[1]) * 0.01 # [mm]
        return current_pos

    # 指定ビットの値を確認
    def _extract_bit(self, data, num):
        return int((data >> num) & 0b1)

    # 原点復帰完了フラグを取得
    def get_home_regression_flag(self):
        # 原点回帰完了フラグの読み取り
        ret = self.master.execute(1, cst.READ_HOLDING_REGISTERS, int("0x9005", 16) , int("0x0001", 16))
        sleep(0.05)
        return self._extract_bit(ret[0], 4)

    # 移動完了フラグを取得
    def get_complete_positioning_flag(self):
        # 原点回帰完了フラグの読み取り
        ret = self.master.execute(1, cst.READ_HOLDING_REGISTERS, int("0x9005", 16) , int("0x0001", 16))
        sleep(0.05)
        return self._extract_bit(ret[0], 3)

    # 指定した位置に移動
    ## 原点復帰位置のオフセット設定箇所からの位置
    def move_to_position(self, pos):
        # 位置決めデータの直接書き込み
        position = str(hex(round(pos*100)))
        self.master.execute(1, cst.WRITE_MULTIPLE_REGISTERS,
                                   int("0x9900", 16), int("0x0002", 16),
                                   output_value=(int("0x0000", 16), int(position, 16)))
        sleep(0.05)
        while not self.get_complete_positioning_flag() == 1:
            self.disp_current_moving_state()
            pass

    # 原点復帰
    def move_to_regression_point(self):
        # 原点復帰動作
        self.master.execute(1, cst.WRITE_SINGLE_COIL, int("0x040B", 16), output_value=int("0x0000", 16))
        sleep(0.05)
        self.master.execute(1, cst.WRITE_SINGLE_COIL, int("0x040B", 16), output_value=int("0xFF00", 16))
        sleep(0.05)
        while not self.get_home_regression_flag() == 1:
            pass

    # 現在の位置情報を表示
    def disp_current_moving_state(self):
        print("Current position: ", self.get_current_position(), " [mm]"
              " Home regression flag: ", self.get_home_regression_flag(),
              " Complete positioning flag: ", self.get_complete_positioning_flag())


if __name__ == '__main__':
    pc = PconCB()

