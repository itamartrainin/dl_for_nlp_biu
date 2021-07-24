import os

if __name__ == '__main__':
    for fname in os.listdir('.'):
        if fname.split('_')[-1] == 'fix':
            with open(fname, 'r', encoding='utf-8') as f, open(fname + '_org', 'w', encoding='utf-8') as f_org:
                for line in f.readlines():
                    line_split = line.strip().split('\t')
                    f_org.write(line_split[-1] + '\n')
