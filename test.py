from multiprocessing import Pool
def job(num):
    return num * 2

if __name__ == '__main__':
    p = Pool(processes=2)
    data = p.map(job, [i for i in range(20)])
    p.close()
    print(data)