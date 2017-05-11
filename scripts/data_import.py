import pandas as pd
import numpy as np
import timeit
import data_mappings


def process_data(df):
    df['CIC'] = df['CIC'].astype(np.uint32)

    df['BIRTH_DATE'] = pd.to_datetime(df['BIRTH_DATE'], errors='coerce', format='%Y-%m-%d')
    df['AGE'] = (df.DAX.dt.year * 12 + df.DAX.dt.month -
                 (df.BIRTH_DATE.dt.year * 12 + df.BIRTH_DATE.dt.month))
    df['AGE'] = df['AGE'].fillna(-1).astype(np.int16)
    df.drop('BIRTH_DATE', axis = 1, inplace = True)

    df['CLIENT_START_DATE'] = pd.to_datetime(df['CLIENT_START_DATE'], errors='coerce', format='%Y-%m-%d')
    df['TENOR'] = ((df.DAX.dt.year - df.CLIENT_START_DATE.dt.year) * 12 +
                   (df.DAX.dt.month - df.CLIENT_START_DATE.dt.month))
    df['TENOR'] = df['TENOR'].fillna(-1).astype(np.int16)
    df.drop('CLIENT_START_DATE', axis = 1, inplace = True)
    
    df['GENDER'] = df['GENDER'].map(lambda x: data_mappings.gender_dict[x]).astype(np.int8)
    df['MARITAL_STATUS'] = df['MARITAL_STATUS'].map(lambda x: data_mappings.marital_status_dict[x]).astype(np.int8)
    df['SEGMENT'] = df['SEGMENT'].map(lambda x: data_mappings.segment_dict[x]).astype(np.int8)
    df['SUBSEGMENT'] = df['SUBSEGMENT'].fillna('XNA').map(lambda x: data_mappings.subsegment_dict[x]).astype(np.int8)
    df['EDUCATION'] = df['EDUCATION'].fillna('XNA').map(lambda x: data_mappings.education_dict[x]).astype(np.int8)
    df['PROFESSION'] = df['PROFESSION'].astype(np.uint16)
        
    df['BCR_EMPLOYEE'] = df['BCR_EMPLOYEE'].map(lambda x: data_mappings.bcr_employee_dict[x]).astype(np.int8)
    df['WORKOUT_FLAG'] = df['WORKOUT_FLAG'].map(lambda x: data_mappings.workout_flag_dict[x]).astype(np.int8)

    df['RATING_VALUE'] = df['RATING_VALUE'].map(lambda x: data_mappings.rating_value_dict[x]).astype(np.int8)
   
    df['MARKETING_AGREEMENT'] = df['MARKETING_AGREEMENT'].fillna('X').map(lambda x: data_mappings.marketing_agreement_dict[x]).astype(np.int8)

    df['ACCOUNT'] = 4
    df.loc[df['JUNIOR'] != 0, 'ACCOUNT'] = 1
    df.loc[df['CAMPUS'] != 0, 'ACCOUNT'] = 2
    df.loc[df['COMOD'] != 0, 'ACCOUNT']  = 3
    df.loc[df['CLASIC'] != 0, 'ACCOUNT'] = 5
    df.loc[df['TOTAL'] != 0 ,'ACCOUNT']  = 6    
    df['ACCOUNT'] = df['ACCOUNT'].astype(np.int8)
    df.drop(['JUNIOR', 'CAMPUS', 'COMOD', 'CLASIC', 'TOTAL'], axis=1, inplace=True)

    df['PBS_TYPE'] = df['PBS_TYPE'].map(lambda x: -1 if x == 'XNA' else 1).astype(np.int8)

    c_int8 = ['FLAG_ACTIVE_34', 'UNSECURED', 'SECURED', 'CREDITCARD', 'OVERDRAFT', 
              'DEPOZIT', 'SAVING_PLAN', 'MAXICONT',
              'PPI', 'UL_KI', 'INDX_LINK', 'HEALTH', 'ACP',
              'ASSET', 'TITLURI', 'AUR', 'PENSII',
              'DIRECT_DEBIT', 'STANDING_ORDER', 'TRANZACTII_POS', 'NET_BANKING']   
    df[c_int8] = df[c_int8].apply(np.int8)

    df['PAD'] = df['PAD'].fillna(-1).astype(np.int8)
    
    c_int16 = ['BRANCH_CODE', 'CLIENT_DPD']
    df[c_int16] = df[c_int16].apply(np.uint16)
          
    c_float16 = ['CM1_A', 'CM1_L', 'NFC', 'BALANCE_MAX_CAS', 'BALANCE_AVG_CAS']   
    df[c_float16] = df[c_float16].apply(np.float16)

    df['BALANCE_MIN_CA_3MONTHS'] = df['BALANCE_MIN_CA_3MONTHS'].astype(np.float16)
    df['BALANCE_MAX_DEPOSITS_3MONTHS'] = df['BALANCE_MAX_DEPOSITS_3MONTHS'].astype(np.float16)
    
    df['NO_SALARY'] = df['NO_SALARY'].fillna(-1).astype(np.int8)
    df['NO_CASH_LAST_MONTH'] = df['NO_CASH_LAST_MONTH'].fillna(-1).astype(np.int8)
    df['NO_OUTGOINGS_MONTH'] = df['NO_OUTGOINGS_MONTH'].fillna(-1).astype(np.int8)

    df['SALARY'] = df['SALARY'].astype(np.float16)
    df['CASH_LAST_MONTH'] = df['CASH_LAST_MONTH'].astype(np.float16)
    df['OUTGOINGS_MONTH'] = df['OUTGOINGS_MONTH'].astype(np.float16)
    
    return df


def main():
       
    tic0 = timeit.default_timer()
    
    reader = pd.read_csv('../data/C_CLIENTS_V_DATA_VIEW.dsv', 
                         sep = ';',
                         chunksize = 100000,
                         parse_dates = ['DAX'])
                         #infer_datetime_format = True)
    
    if __name__ == '__main__':
        df = pd.concat([process_data(chunk) for chunk in reader])
    else:
        for chunk in reader:
            df = process_data(chunk)
            df.info()
            break
    
    print('Load time: ', timeit.default_timer() - tic0)
  
    
    tic0 = timeit.default_timer()
    
    for d in df['DAX'].unique():
        df[df['DAX'] == d].to_hdf('../cache/c_{file_name}.hdf'
                                      .format(file_name=pd.to_datetime(str(d)).strftime('%Y%m')),
                                  key='dump_months', format='fixed', mode='w',
                                  complevel=5, complib='zlib')

    df.to_hdf('../cache/c_clients.hdf', key='dump_whole', mode='w', format='fixed', 
              complevel=5, complib='zlib')
    
    print('Save time: ', timeit.default_timer() - tic0)


main()
