from api import ApiRequest

def main():
    req = ApiRequest()
    req.get_sm('01', '55')
    # req.get(['CES0500000002'])

if __name__ == "__main__":
    main()