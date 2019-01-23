path = '~/Documents/stomata-complex-network-analysis/data/' 
t = '500.0'

for c in ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10']:
    for d in range(1, 5):
        f = open('input/segunda/controle/' + c + '/' + str(d) + '.in', 'w')
        f.write(path + 'segunda/controle/' + c + '/' + str(d) + '.txt')
        f.write('\n' + t)
        f.close()


for a in ['terca', 'quarta', 'quinta']:
    for b in ['controle', 'quente']:
        for c in ['F1', 'F2', 'F3', 'F4', 'F5']:
            for d in range(1, 5):
                f = open('input/' + a + '/' + b + '/' + c + '/' + str(d) + '.in', 'w')
                f.write(path + a + '/' + b + '/' + c + '/' + str(d) + '.txt')
                f.write('\n' + t)
                f.close()
