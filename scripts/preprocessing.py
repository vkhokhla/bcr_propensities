import pandas as pd
import numpy as np
import timeit

segment_dict = {'MM':1, 'MA':2, 'PB':3 }

subsegment_dict = {'XNA':-1,
                   'EMPL':1, 
                   'RETIRED':2, 
                   'STUD':3, 
                   'UNEMPL':4, 
                   'CHILD_ADOL':5, 
                   'SELF_EMPL':6, 
                   'PUBL_SERV':7, 
                   'DOCTOR':8, 
                   'LAWYER':9, 
                   'OWN_EMPL_ACT':99, 
                   'ASSOC_FAM':99, 
                   'CIV_ENG':99, 
                   'FREELANCER':99, 
                   'PUBL_ACC':99, 
                   'NOTAR':99, 
                   'DENTIST':99, 
                   'VETERINARIES':99, 
                   'PHARM':99, 
                   'MEDIATORI':99, 
                   'PUBL_ACC':99, 
                   'OWN_EMPL_RET':99, 
                   'OTHERS':99}

gender_dict = {'F':1, 'M':2}

marital_status_dict = {'X': -1, 'C':1, 'N':2, 'D':3, 'V':4} # Married, not married, divorced, widow

education_dict = {'XNA':-1, 
                  'Alte forme de invatamant':1,
                  'Scoala primara':2, 
                  'Scoala profesionala':3, 
                  'Scoala postliceala':4,
                  'Gimnaziu':5,
                  'Colegiu':6,
                  'Liceu':7,
                  'Master':8,
                  'Universitate':9
                 }

bcr_employee_dict = {'N':0, 'Y':1}
workout_flag_dict = {'N':0, 'Y':1}

rating_value_dict = {'N':-1, 'R':1, 'D2':2, 'D1':3, 'C2':4, 'C1':5, 'B2':6, 'B1':7, 'A2':8, 'A1':9}

marketing_agreement_dict = {'X':-1, 'N':0, 'Y':1}


def process_data(df):
    df['CIC'] = df['CIC'].astype(np.uint32)
    
    df['AGE'] = (df.apply(lambda x: (x['DAX'].year - int(x['BIRTH_DATE'][:4])) * 12 +
                                     x['DAX'].month - int(x['BIRTH_DATE'][5:7])
                                     if not pd.isnull(x['BIRTH_DATE'])
                                        and int(x['BIRTH_DATE'][:4]) > 1900
                                     else -1                 # data quality
                          , axis=1).astype(np.int8))
    df.drop('BIRTH_DATE', axis = 1, inplace = True)

    df['TENOR'] = (df.apply(lambda x: (x['DAX'].year - int(x['CLIENT_START_DATE'][:4])) * 12 +
                                       x['DAX'].month - int(x['CLIENT_START_DATE'][5:7])
                                       if not pd.isnull(x['CLIENT_START_DATE'])
                                          and int(x['CLIENT_START_DATE'][:4]) >= 1990   # data quality
                                       else -1
                          , axis=1).astype(np.int8))
    df.drop('CLIENT_START_DATE', axis = 1, inplace = True)
    
    df['GENDER'] = df['GENDER'].map(lambda x: gender_dict[x]).astype(np.int8)
    df['MARITAL_STATUS'] = df['MARITAL_STATUS'].map(lambda x: marital_status_dict[x]).astype(np.int8)
    df['SEGMENT'] = df['SEGMENT'].map(lambda x: segment_dict[x]).astype(np.int8)
    df['SUBSEGMENT'] = df['SUBSEGMENT'].map(lambda x: subsegment_dict['XNA' if pd.isnull(x)
                                                                      else x]).astype(np.int8)
    df['EDUCATION'] = df['EDUCATION'].map(lambda x: education_dict['XNA' if pd.isnull(x) else x]).astype(np.int8)
        
    df['BCR_EMPLOYEE'] = df['BCR_EMPLOYEE'].map(lambda x: bcr_employee_dict[x]).astype(np.int8)
    df['WORKOUT_FLAG'] = df['WORKOUT_FLAG'].map(lambda x: workout_flag_dict[x]).astype(np.int8)

    df['RATING_VALUE'] = df['RATING_VALUE'].map(lambda x: rating_value_dict[x]).astype(np.int8)
   
    df['MARKETING_AGREEMENT'] = df['MARKETING_AGREEMENT'].map(lambda x: marketing_agreement_dict['X' if pd.isnull(x)
                                                                                                 else x]).astype(np.int8)

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

    df['PAD'] = df['PAD'].map(lambda x: -1 if np.isnan(x) else x).astype(np.int8)
    
    c_int16 = ['BRANCH_CODE', 'CLIENT_DPD']
    df[c_int16] = df[c_int16].apply(np.uint16)
    
    df['PROFESSION'] = df['PROFESSION'].astype(np.int16)
       
    c_float16 = ['CM1_A', 'CM1_L', 'NFC', 'BALANCE_MAX_CAS', 'BALANCE_AVG_CAS']   
    df[c_float16] = df[c_float16].apply(np.float16)

    df['BALANCE_MIN_CA_3MONTHS'] = df['BALANCE_MIN_CA_3MONTHS'].map(lambda x: -1 if np.isnan(x) else x).astype(np.float16)
    df['BALANCE_MAX_DEPOSITS_3MONTHS'] = df['BALANCE_MAX_DEPOSITS_3MONTHS'].map(lambda x: -1 if np.isnan(x) else x).astype(np.float16)
    
    df['NO_SALARY'] = df['NO_SALARY'].map(lambda x: -1 if np.isnan(x) else x).astype(np.int8)
    df['NO_CASH_LAST_MONTH'] = df['NO_CASH_LAST_MONTH'].map(lambda x: -1 if np.isnan(x) else x).astype(np.int8)
    df['NO_OUTGOINGS_MONTH'] = df['NO_OUTGOINGS_MONTH'].map(lambda x: -1 if np.isnan(x) else x).astype(np.int8)

    df['SALARY'] = df['SALARY'].map(lambda x: -1 if np.isnan(x) else x).astype(np.float16)
    df['CASH_LAST_MONTH'] = df['CASH_LAST_MONTH'].map(lambda x: -1 if np.isnan(x) else x).astype(np.float16)
    df['OUTGOINGS_MONTH'] = df['OUTGOINGS_MONTH'].map(lambda x: -1 if np.isnan(x) else x).astype(np.float16)
    
    return df


tic0 = timeit.default_timer()

reader = pd.read_csv('../data/C_CLIENTS_V_DATA_VIEW.dsv', 
                     sep=';', chunksize=100000,
                     parse_dates=['DAX'])

#for chunk in reader:
#    df = process_data(chunk)
#    break
    
df = pd.concat([process_data(chunk) for chunk in reader])

print('Load time: ', timeit.default_timer() - tic0)

df.info()


tic0 = timeit.default_timer()

df.to_pickle('../cache/c_clients.pkl')
#df.to_hdf('../data/processed/c_clients.hdf', 'dump', mode = 'w')

for d in df['DAX'].unique():
    df[df['DAX'] == d].to_pickle('../cache/c_' + pd.to_datetime(str(d)).strftime('%Y%m') + '.pkl')

print('Save time: ', timeit.default_timer() - tic0)

