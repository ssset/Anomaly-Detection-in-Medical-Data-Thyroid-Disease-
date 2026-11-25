import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

binary_data = ['on thyroxine', 'query on thyroxine', 'on antithyroid medication', 'sick', 'pregnant', 'thyroid surgery', 'I131 treatment', 
                'query hypothyroid','query hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH measured', 'T3 measured', 'TT4 measured', 'T4U measured',  'FTI measured']


def feature_engeneering(
            df: pd.DataFrame, qt: QuantileTransformer,
            train_median: pd.Series, train_mad: pd.Series,
            tsh_quantile: float, fti_quantile: float, tt4_quantile: float
            ) -> pd.DataFrame:
        eps = 1e-9
        trans_array = qt.transform(df[['TSH', 'TT4', 'T4U', 'FTI']])
        df_transformed = pd.DataFrame(
            trans_array, 
            columns=['TSH_qt', 'TT4_qt', 'T4U_qt', 'FTI_qt'], 
            index=df.index 
        )

        df_transformed['tsh_tt4_div'] = df['TSH'] / (df['TT4'] + eps)
        df_transformed['tsh_fti_div'] = df['TSH'] / (df['FTI'] + eps)
        df_transformed['tsh_tt4_log_diff'] = np.log1p(df['TSH']) - np.log1p(df['TT4'])
        df_transformed['tsh_fti_log_diff'] = np.log1p(df['TSH']) - np.log1p(df['FTI'])

        df_transformed['is_tsh_extreme_high'] = (df['TSH'] > tsh_quantile).astype(int)
        df_transformed['is_tt4_extreme_low'] = (df['TT4'] < tt4_quantile).astype(int)
        df_transformed['is_fti_extreme_low'] = (df['FTI'] < fti_quantile).astype(int)
        df_transformed['bad_analysis_count'] = df_transformed['is_tsh_extreme_high'] +  df_transformed['is_tt4_extreme_low'] + df_transformed['is_fti_extreme_low']


        df_transformed['z_tsh'] = (df['TSH'] - train_median['TSH']) / (train_mad['TSH'] + eps)
        df_transformed['z_tt4'] = (df['TT4'] - train_median['TT4']) / (train_mad['TT4'] + eps)
        df_transformed['z_fti'] = (df['FTI'] - train_median['FTI']) / (train_mad['FTI'] + eps)
        df_transformed['bin_sum'] = df[binary_data].sum(axis=1)

        df_final = pd.concat([df_transformed, df], axis=1)

        cols_to_drop = ['TSH', 'TT4', 'FTI', 'T4U']
        df_final = df_final.drop(columns=cols_to_drop)

        return df_final


def get_fe_parametrs(df: pd.DataFrame) -> tuple[QuantileTransformer, pd.Series, pd.Series, float, float, float]:
    qt = QuantileTransformer()
    qt.fit(df[['TSH', 'TT4', 'T4U', 'FTI']])

    train_median = df[['FTI', 'TSH', 'TT4']].median()
    train_mad = (df[['FTI', 'TSH', 'TT4']] - train_median).abs().median()

    tsh_q = df['TSH'].quantile(0.99)
    tt4_q = df['TT4'].quantile(0.01)
    fti_q = df['FTI'].quantile(0.01)

    return qt, train_median, train_mad, tsh_q, fti_q, tt4_q
