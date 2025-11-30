import time

def print_section(title:str):
    print("\n" + "="*50)
    print(f"=> {title}")
    print("="*50)

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"실행 시간: {time.time() - start:.2f}초")
        return result
    return wrapper
