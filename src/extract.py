from pandas import concat


def extract_data_by_occ_title(df_dict, target_occ_title='Software Developers'):
    target_occ_title = target_occ_title.lower()
    filter_dfs = []
    for file_key, file_df in df_dict.items():
        filter_dfs.append(
            file_df[file_df['OCC_TITLE'].str.lower().str.contains(target_occ_title)])
    return concat(filter_dfs)
