import os

if __name__ == '__main__':
    for fname in os.listdir('.'):
            if fname.split('.')[-1] != 'py' and fname != 'test':
                with open('test', 'r', encoding='utf-8') as test:
                    test.seek(0)
                    with open(fname, 'r', encoding='utf-8') as f, open(fname + '_fix', 'w', encoding='utf-8') as f_fix:
                        for line1, line2 in zip(test, f):
                            line1 = line1.strip()
                            line2 = line2.strip()
                            f_fix.write('{} {}\n'.format(line1, line2))
