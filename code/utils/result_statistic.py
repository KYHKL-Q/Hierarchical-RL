import numpy as np
import json
import os
import xlwt

def statistics(city,population):
    path = os.path.join('../../result', city)

    FILEs = []#FILEs be to analysed
    result = dict()

    for filename in FILEs:
        temp_dict = dict()
        temp_dict['Infections'] = list()
        temp_dict['Death'] = list()
        temp_dict['Max_Infections_increase'] = list()
        temp_dict['Max_Death_increase'] = list()
        for I in range(10):
            with open(os.path.join(path, filename + '_' + str(I) + '.json'), 'r') as f:
                data = np.array(json.load(f))

            temp_dict['Infections'].append(int(np.sum(data[-1,:,:]) - np.sum(data[-1,:, 0])))
            temp_dict['Death'].append(int(np.sum(data[-1,:, -1])))

            max_death = np.sum(data[47,:, 7])
            max_incr = np.sum(data[47,:,:]) - np.sum(data[47,:, 0])
            for i in range(1,int(np.shape(data)[0] / 48)):
                death = np.sum(data[i * 48 + 47,:, 7]) - np.sum(data[i * 48 - 1,:, 7])
                incr = (np.sum(data[i * 48 + 47,:,:]) - np.sum(data[i * 48 + 47,:, 0])) - (np.sum(data[i * 48 - 1,:,:]) - np.sum(data[i * 48 - 1,:, 0]))
                if death > max_death:
                    max_death = death
                if incr > max_incr:
                    max_incr = incr

            temp_dict['Max_Infections_increase'].append(int(max_incr))
            temp_dict['Max_Death_increase'].append(int(max_death))

            print('Success:' + filename + '_' + str(I))

        result[filename] = temp_dict

    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet('Infections')
    col = 0
    for key in result.keys():
        worksheet.write(0, col, key)
        temp = result[key]['Infections']
        row = 1
        for i in range(len(temp)):
            worksheet.write(row, col, temp[i])
            row += 1

        temp = np.array(temp)
        mu = np.mean(temp)
        std = np.std(temp)
        upper = mu + 1.96 * std
        lower = mu - 1.96 * std

        worksheet.write(row + 1, col, '%.2f' % (100 * mu / population))
        worksheet.write(row + 2, col, '(%.2f,%.2f)' % (100 * lower / population, 100 * upper / population))
        worksheet.write(row + 3, col, '%d' % (mu))
        worksheet.write(row + 4, col, '(%d,%d)' % (lower, upper))
        col += 1

    worksheet = workbook.add_sheet('Death')
    col = 0
    for key in result.keys():
        worksheet.write(0, col, key)
        temp = result[key]['Death']
        row = 1
        for i in range(len(temp)):
            worksheet.write(row, col, temp[i])
            row += 1

        temp = np.array(temp)
        mu = np.mean(temp)
        std = np.std(temp)
        upper = mu + 1.96 * std
        lower = mu - 1.96 * std

        worksheet.write(row + 1, col, '%.2f' % (1000 * mu / population))
        worksheet.write(row + 2, col, '(%.2f,%.2f)' % (1000 * lower / population, 1000 * upper / population))
        worksheet.write(row + 3, col, '%d' % (mu))
        worksheet.write(row + 4, col, '(%d,%d)' % (lower, upper))
        col += 1

    worksheet = workbook.add_sheet('Max_Infections_increase')
    col = 0
    for key in result.keys():
        worksheet.write(0, col, key)
        temp = result[key]['Max_Infections_increase']
        row = 1
        for i in range(len(temp)):
            worksheet.write(row, col, temp[i])
            row += 1

        temp = np.array(temp)
        mu = np.mean(temp)
        std = np.std(temp)
        upper = mu + 1.96 * std
        lower = mu - 1.96 * std

        worksheet.write(row + 1, col, '%.2f' % (1000 * mu / population))
        worksheet.write(row + 2, col, '(%.2f,%.2f)' % (1000 * lower / population, 1000 * upper / population))
        worksheet.write(row + 3, col, '%d' % (mu))
        worksheet.write(row + 4, col, '(%d,%d)' % (lower, upper))
        col += 1

    worksheet = workbook.add_sheet('Max_Death_increase')
    col = 0
    for key in result.keys():
        worksheet.write(0, col, key)
        temp = result[key]['Max_Death_increase']
        row = 1
        for i in range(len(temp)):
            worksheet.write(row, col, temp[i])
            row += 1

        temp = np.array(temp)
        mu = np.mean(temp)
        std = np.std(temp)
        upper = mu + 1.96 * std
        lower = mu - 1.96 * std

        worksheet.write(row + 1, col, '%.2f' % (10000 * mu / population))
        worksheet.write(row + 2, col, '(%.2f,%.2f)' % (10000 * lower / population, 10000 * upper / population))
        worksheet.write(row + 3, col, '%d' % (mu))
        worksheet.write(row + 4, col, '(%d,%d)' % (lower, upper))
        col += 1

    workbook.save(os.path.join(path, 'statistics.xls'))

if __name__ == '__main__':
    os.chdir(os.path.split(os.path.realpath(__file__))[0])
    statistics('sample_city', 11000000)