from re import T
import requests
import json
import pandas as pd
import config

class ApiRequest():
    def __init__(self):
        self.data_type = {
            '01': 'All_Employees',
            '02': 'Average_Weekly_Hours',
            '03': 'Average_Hourly_Earnings'
        }
        self.state_code = None
        self.industry_code = None

    def get_sm(self, state_code, sector):
        series_dict = {}
        for key, val in self.data_type.items():
            s = 'SMU' + state_code + '00000' + sector + '000000' + key
            series_dict[s] = val
        print(series_dict)
        # API key in config.py which contains: bls_key = 'key'
        # plz register your own api to avoid max request
        key = '?registrationkey={}'.format(config.bls_key)

        # The url for BLS API v2
        url = 'https://api.bls.gov/publicAPI/v2/timeseries/data/'

        print('{}{}'.format(url, key))

        # Start year and end year
        dates = ('2007', '2022')

        # Specify json as content type to return
        headers = {'Content-type': 'application/json'}

        # Submit the list of series as data
        data = json.dumps({"seriesid": list(series_dict.keys()),
                            "startyear": dates[0], 
                            "endyear": dates[1]})

        p = requests.post(
            '{}{}'.format(url, key),
            headers=headers,
            data=data)
        json_data = json.loads(p.text)
        print('test', json_data)

        # Post request for the data
        # p = requests.post(
        #     '{}{}'.format(url, key),
        #     headers=headers,
        #     data=data).json()['Results']['series']
        # # Empty dataframe to fill with values
        # df = pd.DataFrame()

        # # Date index from first series
        # date_list = [f"{i['year']}-{i['period'][1:]}-01" for i in p[0]['data']]

        # # Build a pandas series from the API results, p
        # for s in p:
        #     df[series_dict[s['seriesID']]] = pd.Series(
        #         index = pd.to_datetime(date_list),
        #         data = [i['value'] for i in s['data']]
        #         ).astype(float).iloc[::-1]
        
        # print(df)
        # file_name = '../../data/' + state_code + '_' + sector
        #df.to_csv(file_name, sep='\t', encoding='utf-8')
