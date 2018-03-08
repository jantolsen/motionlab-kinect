abc = 3
def test_func(value):
    global abc
    abc = value
    

def print_func():
    print(abc)
if __name__ == '__main__':
    # detect_init = True
    test_func(2)
    print_func()