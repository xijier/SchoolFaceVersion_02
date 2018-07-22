# coding=utf-8
import threading
import time


class MyThread(threading.Thread):
    def run(self):
        global num
        time.sleep(1)

        if mutex.acquire():
            time.sleep(0.7)
            num = num + 1
            msg = self.name + ' set num to ' + str(num)
            print (msg)
            mutex.release()


num = 0
mutex = threading.RLock()
from queue import Queue
q_thread = Queue()
def test():
    index = 0
    while True:
        if not q_thread.empty():
            thread_current = q_thread.get()
            thread_current.start()
            print(q_thread.unfinished_tasks)
        else:
            t = MyThread()
            t.setName(index)
            q_thread.put(t)
            index = index + 1
        time.sleep(0.015)


    # for i in range(5):
    #     t = MyThread()
    #     t.start()

if __name__ == '__main__':
    test()
