from msvcrt import getch


def main3():
    CTRL_C = 3

    while True:
        key = ord(getch())
        if key == CTRL_C:  # ctrl c
            print("ctrl c")
            break
        else:
            message = 'input, {0}'.format(chr(key))
            print(message)


if __name__ == '__main__':
    main3()