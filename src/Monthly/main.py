from api import ApiRequest

def main():
    req = ApiRequest()
    req.get_sm('01', '55')

if __name__ == "__main__":
    main()