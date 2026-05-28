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
    # ------------------------------------------------------------------ #
    cols_to_drop = [
        'PoolArea', 'PoolQC', 'MiscFeature', 'MiscVal',
        'Street', 'Utilities', 'LowQualFinSF',
        '3SsnPorch', 'KitchenAbvGr', 'GarageYrBlt',
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # ------------------------------------------------------------------ #
    # 3. TRATAR NULOS COM SIGNIFICADO DE NEGÓCIO
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
    if 'LotFrontage' in df.columns:
        df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )
        df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())

    if 'MasVnrType' in df.columns:
        df['MasVnrType'] = df['MasVnrType'].fillna('None')

    if 'Electrical' in df.columns:
        df['Electrical'] = df['Electrical'].fillna('SBrkr')

    if 'Functional' in df.columns:
        df['Functional'] = df['Functional'].fillna('Typ')

    # ------------------------------------------------------------------ #
    # 5. CORRIGIR TIPOS
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
    df['has_pool']          = 0

    # ------------------------------------------------------------------ #
    # 7. ENCODING ORDINAL
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
    # 8. ONE-HOT ENCODING
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
    # Usa o diretório do próprio pipeline.py como âncora para que o
    # script funcione independentemente de onde é invocado.
    # ------------------------------------------------------------------ #
    base_dir = os.path.dirname(os.path.abspath(__file__))
    caminho_modelo = os.path.join(base_dir, 'models', 'modelo_final.pkl')
    if not os.path.exists(caminho_modelo):
        raise FileNotFoundError(
            f"Modelo '{caminho_modelo}' não encontrado."
        )
    modelo = joblib.load(caminho_modelo)

    # ------------------------------------------------------------------ #
    # 11. ALINHAMENTO DE COLUNAS
    # Garante exatamente as mesmas colunas que o modelo viu no treino.
    # Colunas a mais são removidas; colunas faltando viram 0.
    # ------------------------------------------------------------------ #
    colunas_treino_path = os.path.join(base_dir, 'colunas_treino.joblib')

    if os.path.exists(colunas_treino_path):
        # Usa a lista de colunas salva no treinamento — mais confiável
        colunas_treino = joblib.load(colunas_treino_path)
        df = df.reindex(columns=colunas_treino, fill_value=0)
    elif hasattr(modelo, 'feature_names_in_'):
        # Fallback: usa as colunas gravadas no próprio modelo
        df = df.reindex(columns=modelo.feature_names_in_, fill_value=0)
    else:
        # Último recurso: mantém o que tiver
        pass

    # ------------------------------------------------------------------ #
    # 12. PREDIÇÃO
    # O modelo foi treinado com log1p(SalePrice) (confirmado pelo MSE de
    # treino ~0.002, que só faz sentido em escala logarítmica).
    # Revertemos com expm1 para devolver preços em dólares.
    # ------------------------------------------------------------------ #
    predicoes_log = modelo.predict(df)
    predicoes = np.expm1(predicoes_log)

    # Garante apenas valores finitos, não-negativos e na escala de dólares.
    # Se algum valor for inválido (NaN/inf), substitui pela mediana das
    # predições válidas — evita retornar casas de $0.
    mascara_validos = np.isfinite(predicoes) & (predicoes > 0)
    if mascara_validos.any():
        mediana_valida = float(np.median(predicoes[mascara_validos]))
    else:
        mediana_valida = 150_000.0  # fallback absoluto

    predicoes = np.where(mascara_validos, predicoes, mediana_valida)
    predicoes_finais = np.clip(predicoes, a_min=1_000.0, a_max=None)

    return predicoes_finais


# ------------------------------------------------------------------ #
# TESTE LOCAL — execute: python pipeline.py
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    arquivo_teste_exemplo = 'treino.csv'

    print("--- Executando Validação Local do Pipeline ---")

    if not os.path.exists(arquivo_teste_exemplo):
        print(f"[Aviso] Arquivo '{arquivo_teste_exemplo}' não encontrado.")
    else:
        try:
            resultados = prever_precos(arquivo_teste_exemplo)

            print("\n✅ Sucesso! O pipeline rodou corretamente.")
            print("-" * 40)
            print(f"Total de predições : {len(resultados)}")
            print(f"Primeiras 5        : {resultados[:5]}")
            print(f"Mín: ${resultados.min():,.0f} | Máx: ${resultados.max():,.0f}")
            print(f"Infinitos/NaN      : {np.sum(~np.isfinite(resultados))}")
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