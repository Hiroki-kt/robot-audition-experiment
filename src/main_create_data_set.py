from _create_data_set import CreateDataSet

if __name__ == '__main__':
    config_ini = './config_'
    date = ['200128_PTs07', '200210_PTs09']
    distance = [200, 300, 400]
    for d in date:
        for dis in distance:
            config_path = config_ini + d + '_kuka_distance_' + str(dis) + '.ini'
            cd = CreateDataSet(config_path)
            cd()
