import gc
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import pandas as pd


# ================================================================
# UTILITÁRIOS E TRATAMENTO DE TEXTO / PLACAS
# ================================================================
class DataUtils:
    NUMERO_PARA_LETRA = {
        "0": "A",
        "1": "B",
        "2": "C",
        "3": "D",
        "4": "E",
        "5": "F",
        "6": "G",
        "7": "H",
        "8": "I",
        "9": "J",
    }
    LETRA_PARA_NUMERO = {v: k for k, v in NUMERO_PARA_LETRA.items()}

    @staticmethod
    def normalizar_texto(valor) -> str:
        if pd.isna(valor):
            return ""
        texto = str(valor).strip().upper()
        texto = (
            unicodedata.normalize("NFKD", texto)
            .encode("ASCII", "ignore")
            .decode("ASCII")
        )
        return re.sub(r"\s+", " ", texto)

    @staticmethod
    def padronizar_placa(valor) -> str:
        if pd.isna(valor):
            return ""
        return re.sub(r"[^A-Z0-9]", "", str(valor).upper().strip())

    @classmethod
    def coluna_canonica(cls, nome: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", cls.normalizar_texto(nome))

    @classmethod
    def encontrar_coluna(
        cls, df: pd.DataFrame, alternativas: list
    ) -> Optional[str]:
        mapa = {cls.coluna_canonica(col): col for col in df.columns}
        for alt in alternativas:
            chave = cls.coluna_canonica(alt)
            if chave in mapa:
                return mapa[chave]
        return None

    @staticmethod
    def converter_numero(valor) -> float:
        if pd.isna(valor):
            return np.nan
        if isinstance(valor, (int, float, np.integer, np.floating)):
            return float(valor)
        texto = str(valor).strip().replace("R$", "").replace(" ", "")
        if not texto:
            return np.nan
        if "." in texto and "," in texto:
            texto = texto.replace(".", "").replace(",", ".")
        elif "," in texto:
            texto = texto.replace(",", ".")
        try:
            return float(texto)
        except ValueError:
            return np.nan

    @classmethod
    def placa_equivalente_mercosul(cls, placa: str) -> Optional[str]:
        placa = cls.padronizar_placa(placa)
        if len(placa) != 7:
            return None
        char = placa[4]
        if char in cls.NUMERO_PARA_LETRA:
            novo = cls.NUMERO_PARA_LETRA[char]
        elif char in cls.LETRA_PARA_NUMERO:
            novo = cls.LETRA_PARA_NUMERO[char]
        else:
            return None
        return placa[:4] + novo + placa[5:]


# ================================================================
# LEITURA DE PLANILHAS (SUPORTA CAMINHO E UPLOAD)
# ================================================================
class DataLoader:

    def __init__(
        self, path_precos, path_frota, path_motoristas, path_abastecimentos
    ):
        self.path_precos = path_precos
        self.path_frota = path_frota
        self.path_motoristas = path_motoristas
        self.path_abastecimentos = path_abastecimentos

    def carregar_precos(self) -> pd.DataFrame:
        df = pd.read_excel(self.path_precos, sheet_name="Planilha1")
        col_tipo = DataUtils.encontrar_coluna(
            df, ["TIPO", "TIPO VEICULO", "TIPO DE VEICULO"]
        )
        col_media = DataUtils.encontrar_coluna(df, ["MEDIA", "MÉDIA"])
        col_premio = DataUtils.encontrar_coluna(
            df, ["TOTAL", "PREMIO", "PRÊMIO", "VALOR"]
        )

        if None in (col_tipo, col_media, col_premio) and df.shape[1] == 5:
            df = df.iloc[1:].copy()
            df.columns = ["DEL", "TIPO", "FATOR", "MEDIA", "PREMIO"]
            col_tipo, col_media, col_premio = "TIPO", "MEDIA", "PREMIO"

        resultado = pd.DataFrame({
            "TIPO": df[col_tipo].apply(DataUtils.normalizar_texto),
            "MEDIA": df[col_media].apply(DataUtils.converter_numero),
            "PREMIO": df[col_premio].apply(DataUtils.converter_numero),
        }).dropna(subset=["MEDIA", "PREMIO"])

        resultado["TIPO"] = resultado["TIPO"].replace({"TOCO": "TRUCK"})
        return resultado[resultado["TIPO"] != ""].reset_index(drop=True)

    def carregar_frota(self) -> Tuple[pd.DataFrame, dict]:
        df = pd.read_excel(self.path_frota)
        col_placa = DataUtils.encontrar_coluna(
            df, ["CAVALO", "PLACA", "PLACA CAVALO", "PLACA DO CAVALO"]
        )
        col_tipo = DataUtils.encontrar_coluna(
            df, ["TIPO", "TIPO VEICULO", "TIPO DE VEICULO", "CATEGORIA"]
        )

        resultado = pd.DataFrame({
            "PLACA_PADRONIZADA": df[col_placa].apply(DataUtils.padronizar_placa),
            "TIPO": df[col_tipo].apply(DataUtils.normalizar_texto),
        })
        resultado = resultado[resultado["PLACA_PADRONIZADA"] != ""].copy()
        resultado["TIPO"] = resultado["TIPO"].replace({"TOCO": "TRUCK"})
        resultado = resultado.drop_duplicates("PLACA_PADRONIZADA", keep="last")

        mapa = dict(zip(resultado["PLACA_PADRONIZADA"], resultado["TIPO"]))
        return resultado, mapa

    def carregar_cadastro_motoristas(self) -> pd.DataFrame:
        bruto = pd.read_excel(self.path_motoristas, header=None)
        cab_idx, linha_cab = None, None

        for i in range(min(len(bruto), 15)):
            vals = [str(x).strip().upper() for x in bruto.iloc[i].tolist()]
            if "MOTORISTAS" in vals and "TIPO" in vals:
                cab_idx, linha_cab = i, vals
                break

        if cab_idx is None:
            raise ValueError(
                "Cabeçalho não encontrado no arquivo de cadastro de"
                " motoristas."
            )

        idx_mot = linha_cab.index("MOTORISTAS")
        idx_tipo = linha_cab.index("TIPO")
        idx_base = linha_cab.index("BASE") if "BASE" in linha_cab else None

        cadastro = bruto.iloc[cab_idx + 1 :].copy()
        cadastro["MOTORISTA_CADASTRO"] = cadastro.iloc[:, idx_mot].apply(
            DataUtils.normalizar_texto
        )
        cadastro["TIPO_CADASTRO"] = (
            cadastro.iloc[:, idx_tipo]
            .apply(DataUtils.normalizar_texto)
            .replace({"TOCO": "TRUCK"})
        )
        cadastro["BASE_CADASTRO"] = (
            cadastro.iloc[:, idx_base].apply(DataUtils.normalizar_texto)
            if idx_base is not None
            else ""
        )
        cadastro["EH_FOLGUISTA"] = cadastro["TIPO_CADASTRO"].eq("FOLGUISTA")

        cadastro = cadastro[
            (cadastro["MOTORISTA_CADASTRO"] != "")
            & (cadastro["TIPO_CADASTRO"] != "")
        ]
        return cadastro.drop_duplicates("MOTORISTA_CADASTRO", keep="last")

    def carregar_abastecimentos(self, mapa_frota: dict) -> pd.DataFrame:
        df = pd.read_excel(self.path_abastecimentos)

        col_placa = DataUtils.encontrar_coluna(df, ["PLACA", "CAVALO"])
        col_km = DataUtils.encontrar_coluna(
            df, ["KM ATUAL", "KM", "KM_1", "QUILOMETRAGEM"]
        )
        col_litros = DataUtils.encontrar_coluna(
            df, ["QTDE", "LITROS", "QUANTIDADE", "QTD"]
        )
        col_valor = DataUtils.encontrar_coluna(
            df,
            [
                "VALOR TOTAL",
                "VALOR",
                "TOTAL",
                "VALOR_TOTAL",
                "VR TOTAL",
                "VLR TOTAL",
                "VALOR COMBUSTIVEL",
                "VALOR (R$)",
            ],
        )
        col_motorista = DataUtils.encontrar_coluna(df, ["CONDUTOR", "MOTORISTA"])
        col_data = DataUtils.encontrar_coluna(df, ["DATA", "Data"])

        resultado = df.copy()
        resultado["_ORDEM_ORIGINAL"] = np.arange(len(resultado))
        resultado["PLACA_PADRONIZADA"] = resultado[col_placa].apply(
            DataUtils.padronizar_placa
        )
        resultado["KM_ATUAL_NUM"] = resultado[col_km].apply(
            DataUtils.converter_numero
        )
        resultado["QTDE_NUM"] = resultado[col_litros].apply(
            DataUtils.converter_numero
        )
        resultado["VALOR_NUM"] = (
            resultado[col_valor].apply(DataUtils.converter_numero).fillna(0.0)
            if col_valor
            else 0.0
        )
        resultado["CONDUTOR_NORMALIZADO"] = (
            resultado[col_motorista]
            .fillna("SEM MOTORISTA")
            .apply(DataUtils.normalizar_texto)
        )
        resultado["DATA_NUM"] = (
            pd.to_datetime(resultado[col_data], errors="coerce", dayfirst=True)
            if col_data
            else pd.NaT
        )

        resultado["TIPO"] = resultado["PLACA_PADRONIZADA"].map(mapa_frota)

        sem_tipo_mask = resultado["TIPO"].isna()
        placas_sem_tipo = resultado.loc[
            sem_tipo_mask, "PLACA_PADRONIZADA"
        ].unique()
        mapa_mercosul = {}

        for p in placas_sem_tipo:
            eq = DataUtils.placa_equivalente_mercosul(p)
            if eq and eq in mapa_frota:
                mapa_mercosul[p] = mapa_frota[eq]

        for idx in resultado.index[sem_tipo_mask]:
            placa = resultado.at[idx, "PLACA_PADRONIZADA"]
            if placa in mapa_mercosul:
                resultado.at[idx, "TIPO"] = mapa_mercosul[placa]

        resultado["REGISTRO_VALIDO"] = (
            (resultado["PLACA_PADRONIZADA"] != "")
            & resultado["KM_ATUAL_NUM"].notna()
            & (resultado["KM_ATUAL_NUM"] > 0)
            & resultado["QTDE_NUM"].notna()
            & (resultado["QTDE_NUM"] > 0)
        )

        return resultado


# ================================================================
# CÁLCULO DE PREMIAÇÕES
# ================================================================
class RewardEngine:

    @staticmethod
    def calcular_eventos_consumo(abastecimentos: pd.DataFrame) -> pd.DataFrame:
        base = abastecimentos[abastecimentos["REGISTRO_VALIDO"]].copy()
        base["_DATA_ORDENACAO"] = base["DATA_NUM"].fillna(
            pd.Timestamp("1900-01-01")
        )
        base = base.sort_values(
            ["PLACA_PADRONIZADA", "_DATA_ORDENACAO", "_ORDEM_ORIGINAL"],
            kind="stable",
        ).copy()

        base["KM_ANTERIOR"] = base.groupby("PLACA_PADRONIZADA")[
            "KM_ATUAL_NUM"
        ].shift(1)
        base["KM_RODADO_EVENTO"] = (
            base["KM_ATUAL_NUM"] - base["KM_ANTERIOR"]
        )

        base["REGISTRO_CONSUMO_VALIDO"] = (
            base["KM_ANTERIOR"].notna()
            & base["KM_RODADO_EVENTO"].notna()
            & (base["KM_RODADO_EVENTO"] > 0)
            & base["QTDE_NUM"].notna()
            & (base["QTDE_NUM"] > 0)
        )

        base["KM_CONSUMO"] = np.where(
            base["REGISTRO_CONSUMO_VALIDO"], base["KM_RODADO_EVENTO"], np.nan
        )
        base["LITROS_CONSUMO"] = np.where(
            base["REGISTRO_CONSUMO_VALIDO"], base["QTDE_NUM"], np.nan
        )
        base["TIPO_CALCULO"] = (
            base["TIPO"].fillna("GERAL").replace({"TOCO": "TRUCK"})
        )

        return base.reset_index(drop=True)

    def faixa_mais_proxima(
        self, media: float, tipo: str, precos: pd.DataFrame
    ) -> dict:
        if pd.isna(media):
            return {
                "MEDIA_FAIXA": np.nan,
                "PREMIO": 0.0,
                "STATUS_PREMIO": "SEM MEDIA",
            }

        tabela = (
            precos[precos["TIPO"] == tipo].copy()
            if tipo in precos["TIPO"].values
            else precos.copy()
        )
        if tabela.empty:
            return {
                "MEDIA_FAIXA": np.nan,
                "PREMIO": 0.0,
                "STATUS_PREMIO": "SEM FAIXA",
            }

        tabela = tabela.sort_values("MEDIA").copy()
        media_min, media_max = float(tabela["MEDIA"].min()), float(
            tabela["MEDIA"].max()
        )
        media_class = round(float(media), 2)

        if media_class < media_min:
            return {
                "MEDIA_FAIXA": np.nan,
                "PREMIO": 0.0,
                "STATUS_PREMIO": "DESCLASSIFICADO",
            }

        if media_class > media_max:
            linha_max = tabela.iloc[-1]
            return {
                "MEDIA_FAIXA": float(linha_max["MEDIA"]),
                "PREMIO": float(linha_max["PREMIO"]),
                "STATUS_PREMIO": "TETO",
            }

        medias = tabela["MEDIA"].astype(float).to_numpy()
        pos = max(
            0,
            min(
                int(np.searchsorted(medias, media_class, side="right") - 1),
                len(tabela) - 1,
            ),
        )
        linha = tabela.iloc[pos]

        return {
            "MEDIA_FAIXA": float(linha["MEDIA"]),
            "PREMIO": float(linha["PREMIO"]),
            "STATUS_PREMIO": "OK",
        }

    def calcular_premios(
        self,
        eventos: pd.DataFrame,
        precos: pd.DataFrame,
        cadastro: pd.DataFrame,
    ) -> pd.DataFrame:
        base = eventos[eventos["REGISTRO_CONSUMO_VALIDO"]].copy()
        base["CATEGORIA_ABASTECIMENTO"] = (
            base["TIPO"].fillna("GERAL").replace({"TOCO": "TRUCK"})
        )
        base["MOTORISTA_CHAVE"] = (
            base["CONDUTOR_NORMALIZADO"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )

        cad = cadastro.copy()
        cad["MOTORISTA_CHAVE"] = (
            cad["MOTORISTA_CADASTRO"].astype(str).str.strip().str.upper()
        )

        base = base.merge(
            cad[[
                "MOTORISTA_CHAVE",
                "TIPO_CADASTRO",
                "BASE_CADASTRO",
                "EH_FOLGUISTA",
            ]],
            on="MOTORISTA_CHAVE",
            how="left",
        )
        base["TIPO_CADASTRO"] = base["TIPO_CADASTRO"].fillna("")
        base["EH_FOLGUISTA"] = base["EH_FOLGUISTA"].fillna(False)

        registros = []
        for _, grupo in base.groupby("MOTORISTA_CHAVE", sort=False):
            eh_folguista = grupo["EH_FOLGUISTA"].iloc[0]
            tipo_cad = grupo["TIPO_CADASTRO"].iloc[0]

            if eh_folguista:
                soma = grupo.groupby("CATEGORIA_ABASTECIMENTO")[
                    "KM_CONSUMO"
                ].sum()
                cat_elegivel = soma.idxmax() if not soma.empty else tipo_cad
            else:
                cat_elegivel = (
                    tipo_cad
                    if tipo_cad
                    else grupo["CATEGORIA_ABASTECIMENTO"].iloc[0]
                )

            sub = grupo[grupo["CATEGORIA_ABASTECIMENTO"] == cat_elegivel]
            if not sub.empty:
                registros.append(sub)

        if not registros:
            return pd.DataFrame()

        df_eleg = pd.concat(registros, ignore_index=True)

        resumo = df_eleg.groupby(
            ["MOTORISTA_CHAVE", "CATEGORIA_ABASTECIMENTO"], as_index=False
        ).agg(
            MOTORISTA=("CONDUTOR_NORMALIZADO", "first"),
            BASE=("BASE_CADASTRO", "first"),
            CATEGORIA=("CATEGORIA_ABASTECIMENTO", "first"),
            KM_TOTAL=("KM_CONSUMO", "sum"),
            LITROS_TOTAL=("LITROS_CONSUMO", "sum"),
            QTD_ABASTECIMENTOS=("LITROS_CONSUMO", "count"),
            PLACAS=("PLACA_PADRONIZADA", lambda s: " | ".join(sorted(set(s)))),
        )

        resumo["MEDIA_CALCULADA"] = np.where(
            resumo["LITROS_TOTAL"] > 0,
            resumo["KM_TOTAL"] / resumo["LITROS_TOTAL"],
            np.nan,
        )
        faixas = resumo.apply(
            lambda r: self.faixa_mais_proxima(
                r["MEDIA_CALCULADA"], r["CATEGORIA"], precos
            ),
            axis=1,
        )

        resumo_df = pd.concat([resumo, pd.DataFrame(list(faixas))], axis=1)
        resumo_df["PREMIO_BRUTO"] = resumo_df["PREMIO"]
        resumo_df["DIAS_AUSENCIA"] = 0
        resumo_df["DIAS_EFETIVOS"] = 30
        return resumo_df


# ================================================================
# RECALCULO DE AUSÊNCIAS (ATESTADOS / FÉRIAS)
# ================================================================
def aplicar_ausencias_no_resumo(
    df_resumo_original: pd.DataFrame, df_ausencias: pd.DataFrame
) -> pd.DataFrame:
    res = df_resumo_original.copy()
    if res.empty:
        return res
    if df_ausencias.empty:
        res["DIAS_AUSENCIA"] = 0
        res["DIAS_EFETIVOS"] = 30
        res["PREMIO"] = res["PREMIO_BRUTO"]
        return res

    soma_dias = df_ausencias.groupby("MOTORISTA")["DIAS"].sum().to_dict()
    res["DIAS_AUSENCIA"] = (
        res["MOTORISTA"].map(soma_dias).fillna(0).astype(int)
    )
    res["DIAS_EFETIVOS"] = np.maximum(0, 30 - res["DIAS_AUSENCIA"])

    res["PREMIO"] = res.apply(
        lambda r: max(0.0, r["PREMIO_BRUTO"] * (r["DIAS_EFETIVOS"] / 30.0)),
        axis=1,
    )
    return res


# ================================================================
# FILTRAGEM DO DASHBOARD
# ================================================================
def aplicar_filtros(
    motorista, placa, categoria, filial, df_resumo, df_eventos
):
    res_f = df_resumo.copy()
    evt_f = df_eventos.copy()

    if res_f.empty or evt_f.empty:
        return (
            "R$ 0,00",
            "R$ 0,00",
            "0 km",
            "0 L",
            "0,00 km/L",
            "0",
            pd.DataFrame(),
            pd.DataFrame(),
        )

    if motorista and motorista != "TODOS":
        m_norm = DataUtils.normalizar_texto(motorista)
        res_f = res_f[
            res_f["MOTORISTA"]
            .apply(DataUtils.normalizar_texto)
            .str.contains(m_norm, na=False)
        ]
        evt_f = evt_f[
            evt_f["CONDUTOR_NORMALIZADO"]
            .apply(DataUtils.normalizar_texto)
            .str.contains(m_norm, na=False)
        ]

    if placa and placa.strip():
        p_norm = DataUtils.padronizar_placa(placa)
        res_f = res_f[
            res_f["PLACAS"]
            .apply(DataUtils.padronizar_placa)
            .str.contains(p_norm, na=False)
        ]
        evt_f = evt_f[evt_f["PLACA_PADRONIZADA"] == p_norm]

    if categoria and categoria != "TODAS":
        c_norm = DataUtils.normalizar_texto(categoria)
        res_f = res_f[res_f["CATEGORIA"] == c_norm]
        evt_f = evt_f[evt_f["TIPO_CALCULO"] == c_norm]

    if filial and filial != "TODAS":
        f_norm = DataUtils.normalizar_texto(filial)
        res_f = res_f[
            res_f["BASE"].apply(DataUtils.normalizar_texto) == f_norm
        ]
        mots_da_filial = (
            res_f["MOTORISTA"].apply(DataUtils.normalizar_texto).unique()
        )
        evt_f = evt_f[
            evt_f["CONDUTOR_NORMALIZADO"]
            .apply(DataUtils.normalizar_texto)
            .isin(mots_da_filial)
        ]

    tot_premio = res_f["PREMIO"].sum() if "PREMIO" in res_f.columns else 0.0
    tot_km = res_f["KM_TOTAL"].sum() if "KM_TOTAL" in res_f.columns else 0.0
    tot_litros = (
        res_f["LITROS_TOTAL"].sum() if "LITROS_TOTAL" in res_f.columns else 0.0
    )
    tot_gasto_combustivel = (
        evt_f["VALOR_NUM"].sum() if "VALOR_NUM" in evt_f.columns else 0.0
    )
    tot_media_geral = (tot_km / tot_litros) if tot_litros > 0 else 0.0
    tot_mots = len(res_f)

    res_view = res_f.copy()
    if "PREMIO" in res_view.columns:
        res_view["PREMIO"] = res_view["PREMIO"].map(
            lambda x: (
                f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            )
        )
    if "MEDIA_CALCULADA" in res_view.columns:
        res_view["MEDIA_CALCULADA"] = res_view["MEDIA_CALCULADA"].map(
            lambda x: f"{x:.2f}" if pd.notna(x) else "-"
        )

    f_premio = (
        f"R$ {tot_premio:,.2f}".replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )
    f_gasto_comb = (
        f"R$ {tot_gasto_combustivel:,.2f}".replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )
    f_km = f"{tot_km:,.1f} km".replace(",", "X").replace(".", ",").replace("X", ".")
    f_litros = (
        f"{tot_litros:,.1f} L".replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )
    f_media = f"{tot_media_geral:.2f} km/L".replace(".", ",")
    f_mots = f"{tot_mots}"

    return (
        f_premio,
        f_gasto_comb,
        f_km,
        f_litros,
        f_media,
        f_mots,
        res_view,
        evt_f,
    )


# ================================================================
# GERADOR DE RECIBOS
# ================================================================
def gerar_html_unico_recibo(
    row_data: pd.Series,
    motorista_sel: str,
    periodo_ini: str,
    periodo_fim: str,
    fator_c: str,
) -> str:
    base_val = row_data.get("BASE", "")
    if pd.isna(base_val) or str(base_val).strip() == "":
        base_val = "CIANORTE"

    tipo_val = row_data.get("CATEGORIA", "")
    km_val = f"{row_data.get('KM_TOTAL', 0):,.0f}".replace(",", ".")
    dias_efetivos_val = row_data.get("DIAS_EFETIVOS", 30)

    media_raw = row_data.get("MEDIA_CALCULADA", np.nan)
    media_val = (
        f"{media_raw:.2f}".replace(".", ",")
        if pd.notna(media_raw) and isinstance(media_raw, (int, float))
        else str(media_raw)
    )

    premio_raw = row_data.get("PREMIO", 0.0)
    val_total_str = (
        f"R$ {premio_raw:,.2f}".replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )

    return f"""
    <div class="recibo-card" style="background-color: #FFFFFF; padding: 28px; border-radius: 12px; max-width: 650px; margin: 0 auto; font-family: Arial, sans-serif; color: #000000; border: 1px solid #CBD5E1; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); page-break-after: always; break-after: page;">
        <div style="text-align: center; margin-bottom: 12px;">
            <div style="display: inline-block; width: 190px;">
                <div style="width: 190px; height: 46px; position: relative; border-radius: 2px; overflow: hidden; display: flex; flex-direction: column; border: 1px solid #CBD5E1;">
                    <div style="height: 33.3%; background-color: #0099DA;"></div>
                    <div style="height: 33.3%; background-color: #FFD700;"></div>
                    <div style="height: 33.3%; background-color: #1E2B7A;"></div>
                    <svg viewBox="0 0 24 24" style="position: absolute; top: 50%; left: 20px; transform: translateY(-50%); width: 24px; height: 24px;">
                        <path fill="#0099DA" stroke="#FFFFFF" stroke-width="1.5" d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"/>
                        <path fill="#FFFFFF" d="M10 8.5a2.5 2.5 0 0 0 2.5 2.5 0.7 0.7 0 0 0-1.4 0z"/>
                    </svg>
                    <span style="position: absolute; top: 50%; left: 105px; transform: translate(-50%, -50%); font-family: 'Arial Black', sans-serif; font-size: 20px; font-weight: 900; color: #1E2B7A;">Ciapetro</span>
                </div>
            </div>
        </div>

        <h2 style="text-align: center; margin: 10px 0 16px 0; font-size: 18px; font-weight: bold; color: #000000;">Recibo de Premiação</h2>

        <table style="width: 100%; border-collapse: collapse; border: 2px solid #000000; font-size: 12px; font-weight: bold;">
            <tr style="border-bottom: 1px solid #000000;">
                <td style="width: 45%; background-color: #D0E0F0; padding: 4px 8px; border-right: 1px solid #000000; text-align: center;">MOTORISTA</td>
                <td style="width: 55%; padding: 4px 8px; text-align: center; background-color: #FFFFFF;">{motorista_sel}</td>
            </tr>
            <tr style="border-bottom: 1px solid #000000;">
                <td style="background-color: #D0E0F0; padding: 4px 8px; border-right: 1px solid #000000; text-align: center;">BASE</td>
                <td style="padding: 4px 8px; text-align: center;">{base_val}</td>
            </tr>
            <tr style="border-bottom: 1px solid #000000;">
                <td style="background-color: #D0E0F0; padding: 4px 8px; border-right: 1px solid #000000; text-align: center;">TIPO</td>
                <td style="padding: 4px 8px; text-align: center;">{tipo_val}</td>
            </tr>
            <tr style="border-bottom: 1px solid #000000;">
                <td style="background-color: #D0E0F0; padding: 4px 8px; border-right: 1px solid #000000; text-align: center;">DIAS EFETIVOS</td>
                <td style="padding: 4px 8px; text-align: center;">{dias_efetivos_val}</td>
            </tr>
            <tr style="border-bottom: 1px solid #000000;">
                <td style="background-color: #D0E0F0; padding: 4px 8px; border-right: 1px solid #000000; text-align: center;">OUTROS CONTROLES</td>
                <td style="padding: 4px 8px; text-align: center;">0</td>
            </tr>
            <tr style="border-bottom: 1px solid #000000;">
                <td style="background-color: #D0E0F0; padding: 4px 8px; border-right: 1px solid #000000; text-align: center;">JORNADA</td>
                <td style="padding: 4px 8px; text-align: center;">0</td>
            </tr>
            <tr style="border-bottom: 1px solid #000000;">
                <td style="background-color: #D0E0F0; padding: 4px 8px; border-right: 1px solid #000000; text-align: center;">EXCESSO DE VELOCIDADE</td>
                <td style="padding: 4px 8px; text-align: center;">0</td>
            </tr>
            <tr style="border-bottom: 1px solid #000000;">
                <td style="background-color: #D0E0F0; padding: 4px 8px; border-right: 1px solid #000000; text-align: center;">KM RODADO</td>
                <td style="padding: 4px 8px; text-align: center;">{km_val}</td>
            </tr>
            <tr style="border-bottom: 1px solid #000000;">
                <td style="background-color: #D0E0F0; padding: 4px 8px; border-right: 1px solid #000000; text-align: center;">MEDIA</td>
                <td style="padding: 4px 8px; text-align: center;">{media_val}</td>
            </tr>
            <tr style="border-bottom: 1px solid #000000;">
                <td style="background-color: #D0E0F0; padding: 4px 8px; border-right: 1px solid #000000; text-align: center;">FATOR CARGA</td>
                <td style="padding: 4px 8px; text-align: center;">{fator_c}</td>
            </tr>
            <tr style="border-bottom: 1px solid #000000;">
                <td style="background-color: #D0E0F0; padding: 4px 8px; border-right: 1px solid #000000; text-align: center;">PONTOS NEG</td>
                <td style="padding: 4px 8px; text-align: center;">0</td>
            </tr>
            <tr style="border-bottom: 1px solid #000000;">
                <td style="background-color: #D0E0F0; padding: 4px 8px; border-right: 1px solid #000000; text-align: center;">CONTROLES</td>
                <td style="padding: 4px 8px; text-align: center;">130</td>
            </tr>
            <tr style="border-bottom: 1px solid #000000;">
                <td style="background-color: #D0E0F0; padding: 4px 8px; border-right: 1px solid #000000; text-align: center;">VALOR TOTAL CONTROLES</td>
                <td style="padding: 4px 8px; text-align: center;">R$ 0,00</td>
            </tr>
            <tr style="border-bottom: 1px solid #000000;">
                <td style="background-color: #D0E0F0; padding: 4px 8px; border-right: 1px solid #000000; text-align: center;">% CONTROLES</td>
                <td style="padding: 4px 8px; text-align: center;">100%</td>
            </tr>
            <tr style="border-bottom: 1px solid #000000;">
                <td style="background-color: #D0E0F0; padding: 4px 8px; border-right: 1px solid #000000; text-align: center;">R$ CONTROLES</td>
                <td style="padding: 4px 8px; text-align: center;">R$ 0,00</td>
            </tr>
            <tr style="border-bottom: 1px solid #000000;">
                <td style="background-color: #D0E0F0; padding: 4px 8px; border-right: 1px solid #000000; text-align: center;">VALOR MÉDIA</td>
                <td style="padding: 4px 8px; text-align: center;">{val_total_str}</td>
            </tr>
            <tr>
                <td style="background-color: #00A3E0; color: #FFFFFF; padding: 6px 8px; border-right: 1px solid #000000; text-align: center; font-size: 13px;">VALOR TOTAL</td>
                <td style="padding: 6px 8px; text-align: center; font-size: 13px; background-color: #FFFFFF; font-weight: bold;">{val_total_str}</td>
            </tr>
        </table>

        <p style="text-align: center; margin-top: 18px; margin-bottom: 20px; font-size: 11px; font-weight: normal; line-height: 1.4; color: #000000;">
            Eu, <strong>{motorista_sel}</strong> ,Conferi e concordo com as informações, pois estão de acordo com a Política de Premiação dos Motoristas.
        </p>

        <div style="margin-top: 25px; text-align: left; font-size: 11px; color: #000000;">
            <span>Ass.: __________________________________________________</span><br/>
            <div style="margin-left: 35px; margin-top: 4px; font-weight: bold;">{motorista_sel}</div>
        </div>

        <div style="margin-top: 20px; text-align: left; font-size: 11px; color: #000000;">
            LOCAL/DATA _____________, ______/______/ 2026
        </div>

        <div style="margin-top: 12px; text-align: center; font-size: 11px; font-weight: bold; color: #000000;">
            Período de Controle: {periodo_ini} a {periodo_fim}
        </div>
    </div>
    """


def gerar_recibos_lote(
    filial_sel: str,
    motorista_sel: str,
    periodo_ini: str,
    periodo_fim: str,
    fator_c: str,
    df_resumo: pd.DataFrame,
) -> str:
    if df_resumo.empty:
        return (
            "<div style='text-align: center; padding: 40px; color:"
            " #64748B;'>Nenhum dado processado ainda. Faça o upload das"
            " planilhas para gerar os recibos.</div>"
        )

    if not motorista_sel or motorista_sel in ("SELECIONE...", ""):
        return (
            "<div style='text-align: center; padding: 40px; color: #64748B;"
            " font-size: 15px;'>👉 Por favor, selecione um motorista ou a opção"
            " de TODOS DA FILIAL para gerar os recibos.</div>"
        )

    res_f = df_resumo.copy()

    if str(motorista_sel).startswith("TODOS"):
        if filial_sel and filial_sel != "TODAS":
            f_norm = DataUtils.normalizar_texto(filial_sel)
            res_f = res_f[
                res_f["BASE"].apply(DataUtils.normalizar_texto) == f_norm
            ]

        lista_mots = sorted(list(res_f["MOTORISTA"].dropna().unique()))
        if not lista_mots:
            return (
                "<div style='text-align: center; padding: 40px; color:"
                f" #EF4444;'>Nenhum motorista encontrado na filial '{filial_sel}'.</div>"
            )
    else:
        lista_mots = [motorista_sel]

    recibos_html = []

    recibos_html.append(f"""
    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 12px 20px; border-radius: 8px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center;">
        <span style="font-size: 14px; font-weight: bold; color: #1E293B;">
            📄 Total de Recibos Prontos: <span style="color: #2563EB;">{len(lista_mots)}</span>
        </span>
        <button onclick="window.print()" style="background-color: #2563EB; color: #FFFFFF; border: none; padding: 8px 18px; border-radius: 6px; font-weight: bold; cursor: pointer; font-size: 13px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            🖨️ Imprimir Todos os Recibos ({len(lista_mots)})
        </button>
    </div>
    """)

    for m_nome in lista_mots:
        row = df_resumo[df_resumo["MOTORISTA"] == m_nome]
        if not row.empty:
            card_html = gerar_html_unico_recibo(
                row.iloc[0], m_nome, periodo_ini, periodo_fim, fator_c
            )
            recibos_html.append(card_html)

    return (
        "<div style='display: flex; flex-direction: column; gap: 30px;'>"
        + "".join(recibos_html)
        + "</div>"
    )


# ================================================================
# FUNÇÃO PRINCIPAL DE PROCESSAMENTO DAS PLANILHAS
# ================================================================
def processar_arquivos(
    file_precos, file_frota, file_motoristas, file_abastecimentos
):
    path_precos = file_precos.name if file_precos else "Pasta2.xlsx"
    path_frota = file_frota.name if file_frota else "frota.xlsx"
    path_motoristas = file_motoristas.name if file_motoristas else "Pasta4.xlsx"
    path_abastecimentos = (
        file_abastecimentos.name
        if file_abastecimentos
        else "uah abastecimentos_3.xlsx"
    )

    faltantes = [
        p
        for p in (
            path_precos,
            path_frota,
            path_motoristas,
            path_abastecimentos,
        )
        if not os.path.isfile(p)
    ]
    if faltantes:
        msg = (
            "⚠️ Por favor, envie os arquivos necessários pelo painel acima."
            " Faltando:\n"
            + "\n".join(f"- {f}" for f in faltantes)
        )
        return (
            msg,
            pd.DataFrame(),
            pd.DataFrame(),
            gr.Dropdown(choices=["TODOS"]),
            gr.Dropdown(choices=["TODAS"]),
            gr.Dropdown(choices=["TODAS"]),
            gr.Dropdown(choices=["TODOS"]),
            "Aguardando envio das planilhas...",
        )

    try:
        loader = DataLoader(
            path_precos, path_frota, path_motoristas, path_abastecimentos
        )
        engine = RewardEngine()

        precos = loader.carregar_precos()
        _, mapa_frota = loader.carregar_frota()
        cadastro = loader.carregar_cadastro_motoristas()
        abastecimentos = loader.carregar_abastecimentos(mapa_frota)

        eventos = engine.calcular_eventos_consumo(abastecimentos)
        resumo_base = engine.calcular_premios(eventos, precos, cadastro)

        mots_lista = ["TODOS"] + sorted(
            list(resumo_base["MOTORISTA"].dropna().unique())
        )
        cats_lista = ["TODAS"] + sorted(
            list(resumo_base["CATEGORIA"].dropna().unique())
        )
        filiais_lista = ["TODAS"] + sorted([
            str(b)
            for b in resumo_base["BASE"].dropna().unique()
            if str(b).strip() != ""
        ])
        opcoes_recibo = [
            "SELECIONE...",
            "TODOS OS MOTORISTAS (TODAS AS FILIAIS)",
        ] + sorted(list(resumo_base["MOTORISTA"].dropna().unique()))

        msg_sucesso = "✅ Planilhas processadas com sucesso!"

        # Limpeza forçada de memória RAM
        gc.collect()

        return (
            msg_sucesso,
            resumo_base,
            eventos,
            gr.Dropdown(choices=mots_lista, value="TODOS"),
            gr.Dropdown(choices=cats_lista, value="TODAS"),
            gr.Dropdown(choices=filiais_lista, value="TODAS"),
            gr.Dropdown(
                choices=opcoes_recibo,
                value="TODOS OS MOTORISTAS (TODAS AS FILIAIS)",
            ),
            "Pronto para gerar recibos.",
        )
    except Exception as e:
        return (
            f"❌ Erro ao processar planilhas: {str(e)}",
            pd.DataFrame(),
            pd.DataFrame(),
            gr.Dropdown(),
            gr.Dropdown(),
            gr.Dropdown(),
            gr.Dropdown(),
            "Erro ao processar.",
        )


# ================================================================
# MONTAGEM DA INTERFACE GRADIO
# ================================================================
with gr.Blocks(title="Dashboard do Prêmio de Motoristas") as app:
    gr.Markdown("# 🚚 Dashboard do Prêmio de Motoristas")

    # ESTADOS DA APLICAÇÃO
    state_resumo = gr.State(value=pd.DataFrame())
    state_eventos = gr.State(value=pd.DataFrame())

    with gr.Tab("📁 Upload e Processamento"):
        gr.Markdown(
            "### Envie as planilhas para processar os cálculos do mês"
        )
        with gr.Row():
            file_precos = gr.File(
                label="Planilha de Preços (Pasta2.xlsx)", file_count="single"
            )
            file_frota = gr.File(
                label="Planilha de Frota (frota.xlsx)", file_count="single"
            )
            file_motoristas = gr.File(
                label="Cadastro de Motoristas (Pasta4.xlsx)",
                file_count="single",
            )
            file_abastecimentos = gr.File(
                label="Abastecimentos (uah abastecimentos_3.xlsx)",
                file_count="single",
            )

        btn_processar = gr.Button("🚀 Processar Planilhas", variant="primary")
        lbl_status = gr.Textbox(
            label="Status do Processamento", interactive=False
        )

    with gr.Tab("📊 Dashboard de Resultados"):
        with gr.Row():
            kpi_premio = gr.Textbox(
                label="Total Prêmio", value="R$ 0,00", interactive=False
            )
            kpi_gasto = gr.Textbox(
                label="Gasto Combustível", value="R$ 0,00", interactive=False
            )
            kpi_km = gr.Textbox(
                label="Total KM", value="0 km", interactive=False
            )
            kpi_litros = gr.Textbox(
                label="Total Litros", value="0 L", interactive=False
            )
            kpi_media = gr.Textbox(
                label="Média Geral", value="0,00 km/L", interactive=False
            )
            kpi_mots = gr.Textbox(
                label="Motoristas", value="0", interactive=False
            )

        gr.Markdown("### 🔍 Filtros")
        with gr.Row():
            drop_mot = gr.Dropdown(
                choices=["TODOS"], value="TODOS", label="Motorista"
            )
            txt_placa = gr.Textbox(
                label="Placa", placeholder="Digite a placa..."
            )
            drop_cat = gr.Dropdown(
                choices=["TODAS"], value="TODAS", label="Categoria"
            )
            drop_filial = gr.Dropdown(
                choices=["TODAS"], value="TODAS", label="Filial / Base"
            )

        btn_filtrar = gr.Button("🔍 Aplicar Filtros", variant="primary")

        with gr.Tab("Resumo Premiações"):
            grid_resumo = gr.Dataframe(interactive=False)

        with gr.Tab("Eventos Abastecimento"):
            grid_eventos = gr.Dataframe(interactive=False)

    with gr.Tab("📄 Gerador de Recibos"):
        with gr.Row():
            recibo_filial = gr.Dropdown(
                choices=["TODAS"], value="TODAS", label="Filial"
            )
            recibo_mot = gr.Dropdown(
                choices=["TODOS OS MOTORISTAS (TODAS AS FILIAIS)"],
                value="TODOS OS MOTORISTAS (TODAS AS FILIAIS)",
                label="Motorista",
            )
        with gr.Row():
            recibo_ini = gr.Textbox(label="Data Início", value="01/08/2026")
            recibo_fim = gr.Textbox(label="Data Fim", value="31/08/2026")
            recibo_fator = gr.Textbox(label="Fator Carga", value="1,00")

        btn_recibos = gr.Button("🖨️ Gerar Recibos", variant="primary")
        html_recibos = gr.HTML()

    # CONEXÕES DE EVENTOS
    btn_processar.click(
        fn=processar_arquivos,
        inputs=[file_precos, file_frota, file_motoristas, file_abastecimentos],
        outputs=[
            lbl_status,
            state_resumo,
            state_eventos,
            drop_mot,
            drop_cat,
            drop_filial,
            recibo_mot,
            lbl_status,
        ],
    )

    btn_filtrar.click(
        fn=aplicar_filtros,
        inputs=[
            drop_mot,
            txt_placa,
            drop_cat,
            drop_filial,
            state_resumo,
            state_eventos,
        ],
        outputs=[
            kpi_premio,
            kpi_gasto,
            kpi_km,
            kpi_litros,
            kpi_media,
            kpi_mots,
            grid_resumo,
            grid_eventos,
        ],
    )

    btn_recibos.click(
        fn=gerar_recibos_lote,
        inputs=[
            recibo_filial,
            recibo_mot,
            recibo_ini,
            recibo_fim,
            recibo_fator,
            state_resumo,
        ],
        outputs=[html_recibos],
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"--- INICIANDO GRADIO NA PORTA {port} ---", flush=True)
    app.launch(server_name="0.0.0.0", server_port=port)
