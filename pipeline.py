import pandas as pd
import joblib
import numpy as np
import os
from sklearn.metrics import mean_squared_log_error


def prever_precos(caminho_arquivo_teste):
    """
    Função obrigatória para o corretor automático.
    Lê o arquivo de teste, aplica todo o pré-processamento e retorna as predições.

    Parâmetros:
    caminho_arquivo_teste (str): Caminho local para o arquivo CSV de teste.

    Retorna:
    np.array: Predições de preços em dólares (escala original, não log).
    """

    # ------------------------------------------------------------------ #
    # 1. LEITURA
    # ------------------------------------------------------------------ #
    df = pd.read_csv(caminho_arquivo_teste)

    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])

    # ------------------------------------------------------------------ #
    # 2. REMOÇÃO DE COLUNAS DE BAIXO VALOR PREDITIVO
    # Mesmas colunas removidas no notebook de tratamento
    # ------------------------------------------------------------------ #
    cols_to_drop = [
        'PoolArea', 'PoolQC', 'MiscFeature', 'MiscVal',
        'Street', 'Utilities', 'LowQualFinSF',
        '3SsnPorch', 'KitchenAbvGr', 'GarageYrBlt',
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # ------------------------------------------------------------------ #
    # 3. TRATAR NULOS COM SIGNIFICADO DE NEGÓCIO
    # NaN = ausência da feature, não dado perdido
    # ------------------------------------------------------------------ #
    cols_none = [
        'Alley', 'Fence', 'FireplaceQu',
        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    ]
    for col in cols_none:
        if col in df.columns:
            df[col] = df[col].fillna('None')

    cols_zero = [
        'GarageArea', 'GarageCars',
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
        'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea',
    ]
    for col in cols_zero:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # ------------------------------------------------------------------ #
    # 4. TRATAR NULOS GENUINAMENTE FALTANTES
    # ------------------------------------------------------------------ #
    # LotFrontage: mediana do bairro
    if 'LotFrontage' in df.columns:
        df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )
        # Caso algum bairro inteiro seja nulo, usa mediana global
        df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())

    if 'MasVnrType' in df.columns:
        df['MasVnrType'] = df['MasVnrType'].fillna('None')

    if 'Electrical' in df.columns:
        df['Electrical'] = df['Electrical'].fillna('SBrkr')  # moda do treino

    if 'Functional' in df.columns:
        df['Functional'] = df['Functional'].fillna('Typ')    # moda do treino

    # ------------------------------------------------------------------ #
    # 5. CORRIGIR TIPOS — códigos numéricos que são categorias
    # ------------------------------------------------------------------ #
    for col in ['MSSubClass', 'MoSold', 'YrSold']:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # ------------------------------------------------------------------ #
    # 6. FEATURE ENGINEERING
    # ------------------------------------------------------------------ #
    yr_sold_int = df['YrSold'].astype(int)

    df['house_age']         = yr_sold_int - df['YearBuilt']
    df['years_since_remod'] = yr_sold_int - df['YearRemodAdd']
    df['was_remodeled']     = (df['YearBuilt'] != df['YearRemodAdd']).astype(int)
    df['total_sf']          = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['total_bathrooms']   = (df['FullBath'] + df['BsmtFullBath']
                               + 0.5 * df['HalfBath'] + 0.5 * df['BsmtHalfBath'])
    df['total_porch_sf']    = (df['OpenPorchSF'] + df['EnclosedPorch']
                               + df['ScreenPorch'])
    df['has_fireplace']     = (df['Fireplaces'] > 0).astype(int)
    df['has_garage']        = (df['GarageArea'] > 0).astype(int)
    df['has_pool']          = 0  # coluna removida, todas as casas valem 0

    # ------------------------------------------------------------------ #
    # 7. ENCODING ORDINAL — variáveis com hierarquia clara
    # ------------------------------------------------------------------ #
    qual_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    qual_cols = [
        'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
        'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond',
    ]
    for col in qual_cols:
        if col in df.columns:
            df[col] = df[col].map(qual_map).fillna(0)

    if 'BsmtExposure' in df.columns:
        df['BsmtExposure'] = df['BsmtExposure'].map(
            {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
        ).fillna(0)

    bsmt_fin_map = {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
    for col in ['BsmtFinType1', 'BsmtFinType2']:
        if col in df.columns:
            df[col] = df[col].map(bsmt_fin_map).fillna(0)

    if 'GarageFinish' in df.columns:
        df['GarageFinish'] = df['GarageFinish'].map(
            {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
        ).fillna(0)

    if 'Functional' in df.columns:
        func_map = {'Sal': 0, 'Sev': 1, 'Maj2': 2, 'Maj1': 3,
                    'Mod': 4, 'Min2': 5, 'Min1': 6, 'Typ': 7}
        df['Functional'] = df['Functional'].map(func_map).fillna(7)

    if 'PavedDrive' in df.columns:
        df['PavedDrive'] = df['PavedDrive'].map({'N': 0, 'P': 1, 'Y': 2}).fillna(0)

    if 'LandSlope' in df.columns:
        df['LandSlope'] = df['LandSlope'].map({'Gtl': 0, 'Mod': 1, 'Sev': 2}).fillna(0)

    if 'CentralAir' in df.columns:
        df['CentralAir'] = df['CentralAir'].map({'N': 0, 'Y': 1}).fillna(0)

    # ------------------------------------------------------------------ #
    # 8. ONE-HOT ENCODING — variáveis nominais sem hierarquia
    # ------------------------------------------------------------------ #
    onehot_cols = [
        'MSSubClass', 'MSZoning', 'Alley', 'LotShape', 'LandContour',
        'LotConfig', 'Neighborhood', 'Condition1', 'Condition2',
        'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
        'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation',
        'Heating', 'Electrical', 'GarageType', 'Fence',
        'SaleType', 'SaleCondition', 'MoSold', 'YrSold',
    ]
    onehot_cols = [c for c in onehot_cols if c in df.columns]
    df = pd.get_dummies(df, columns=onehot_cols, drop_first=False, dtype=int)

    # ------------------------------------------------------------------ #
    # 9. LOG1P NAS FEATURES ASSIMÉTRICAS
    # Mesmas colunas transformadas no treino (skew > 0.75)
    # ------------------------------------------------------------------ #
    skewed_cols = [
        'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
        'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea',
        'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
        'ScreenPorch', 'total_sf', 'total_porch_sf',
    ]
    for col in skewed_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))

    # ------------------------------------------------------------------ #
    # 10. CARREGAMENTO DO MODELO
    # ------------------------------------------------------------------ #
    caminho_modelo = 'modelo_baseline.joblib'
    if not os.path.exists(caminho_modelo):
        raise FileNotFoundError(
            f"Modelo '{caminho_modelo}' não encontrado. "
            "Certifique-se de que o arquivo está na raiz do repositório."
        )
    modelo = joblib.load(caminho_modelo)

    # ------------------------------------------------------------------ #
    # 11. ALINHAMENTO DE COLUNAS
    # Garante que o teste tenha exatamente as colunas que o modelo espera
    # ------------------------------------------------------------------ #
    if hasattr(modelo, 'feature_names_in_'):
        df = df.reindex(columns=modelo.feature_names_in_, fill_value=0)

    # ------------------------------------------------------------------ #
    # 12. PREDIÇÃO E PÓS-PROCESSAMENTO
    # Se o modelo foi treinado com log1p(SalePrice), reverter com expm1
    # Se usou TransformedTargetRegressor, a reversão é automática
    # ------------------------------------------------------------------ #
    predicoes = modelo.predict(df)

    # Reverte log1p caso o modelo não use TransformedTargetRegressor
    # Comente esta linha se o seu modelo já retorna valores em dólares
    predicoes = np.expm1(predicoes)

    # Garante que não haja valores negativos
    predicoes_finais = np.clip(predicoes, a_min=0, a_max=None)

    return predicoes_finais


# ------------------------------------------------------------------ #
# TESTE LOCAL
# Execute: python pipeline.py
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    arquivo_teste_exemplo = 'teste_publico.csv'

    print("--- Executando Validação Local do Pipeline ---")

    if not os.path.exists(arquivo_teste_exemplo):
        print(f"[Aviso] Arquivo '{arquivo_teste_exemplo}' não encontrado.")
    else:
        try:
            resultados = prever_precos(arquivo_teste_exemplo)

            print("\n✅ Sucesso! O pipeline rodou corretamente.")
            print("-" * 40)
            print(f"Total de predições: {len(resultados)}")
            print(f"Primeiras 5 predições: {resultados[:5]}")
            print(f"Mín: ${resultados.min():,.0f} | Máx: ${resultados.max():,.0f}")
            print("-" * 40)

            df_val = pd.read_csv(arquivo_teste_exemplo)
            if 'SalePrice' in df_val.columns:
                y_true = df_val['SalePrice'].values
                rmsle = np.sqrt(mean_squared_log_error(y_true, resultados))
                mae   = np.mean(np.abs(y_true - resultados))
                print(f"RMSLE local : {rmsle:.5f}")
                print(f"MAE local   : ${mae:,.0f}")
            else:
                print("[Nota] 'SalePrice' não encontrado — RMSLE não calculado.")

        except Exception as e:
            print(f"\n❌ Erro no pipeline: {e}")