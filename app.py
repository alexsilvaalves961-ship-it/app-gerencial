import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import openpyxl
import pandas as pd


# ================================================================
# CONFIGURAÇÃO DE CAMINHOS E PERSISTÊNCIA (LOCAL / RAILWAY)
# ================================================================
DATA_DIR = os.getenv("DATA_DIR", ".")
ARQUIVO_AUSENCIAS = os.path.join(DATA_DIR, "ausencias.csv")
ARQUIVO_DESCLASSIFICACOES = os.path.join(DATA_DIR, "desclassificacoes.csv")
ARQUIVO_CATEGORIAS_CUSTOM = os.path.join(DATA_DIR, "categorias_customizadas.csv")
ARQUIVO_FROTA_CUSTOM = os.path.join(DATA_DIR, "frota_customizada.csv")
ARQUIVO_MOTORISTAS_CUSTOM = os.path.join(DATA_DIR, "motoristas_customizados.csv")
ARQUIVO_INATIVOS = os.path.join(DATA_DIR, "inativos.csv")


def garantir_diretorio():
  """Garante a existência do diretório de dados antes de operações de escrita."""
  if DATA_DIR and DATA_DIR != "." and not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)


def carregar_ausencias() -> pd.DataFrame:
  """Carrega as ausências salvas em disco ou retorna DataFrame vazio."""
  if os.path.exists(ARQUIVO_AUSENCIAS):
    try:
      df = pd.read_csv(ARQUIVO_AUSENCIAS, dtype=str, encoding="utf-8-sig")
      if "DATA_FIM" not in df.columns:
        df["DATA_FIM"] = ""
      if "DIAS" in df.columns:
        df["DIAS"] = pd.to_numeric(df["DIAS"], errors="coerce").fillna(1)
      else:
        df["DIAS"] = 1
      colunas = ["MOTORISTA", "TIPO_AUSENCIA", "DATA_INICIO", "DATA_FIM", "DIAS", "OBSERVACAO"]
      for coluna in colunas:
        if coluna not in df.columns:
          df[coluna] = ""
      return df[colunas]
    except Exception as e:
      print(f"Erro ao carregar ausências: {e}")
  return pd.DataFrame(
      columns=["MOTORISTA", "TIPO_AUSENCIA", "DATA_INICIO", "DIAS", "OBSERVACAO"]
  )


def salvar_ausencias(df: pd.DataFrame):
  """Salva as ausências no arquivo CSV com codificação UTF-8."""
  try:
    garantir_diretorio()
    df.to_csv(ARQUIVO_AUSENCIAS, index=False, encoding="utf-8-sig")
  except Exception as e:
    print(f"Erro ao salvar ausências: {e}")


def carregar_desclassificacoes() -> pd.DataFrame:
  """Carrega as desclassificações salvas em disco ou retorna DataFrame vazio."""
  if os.path.exists(ARQUIVO_DESCLASSIFICACOES):
    try:
      df = pd.read_csv(
          ARQUIVO_DESCLASSIFICACOES, dtype=str, encoding="utf-8-sig"
      )
      if "PONTOS" in df.columns:
        df["PONTOS"] = pd.to_numeric(df["PONTOS"], errors="coerce").fillna(1)
      return df
    except Exception as e:
      print(f"Erro ao carregar desclassificações: {e}")
  return pd.DataFrame(
      columns=["MOTORISTA", "CRITERIO", "PONTOS", "TIPO_IMPACTO", "OBSERVACAO"]
  )


def salvar_desclassificacoes(df: pd.DataFrame):
  """Salva as desclassificações no arquivo CSV com codificação UTF-8."""
  try:
    garantir_diretorio()
    df.to_csv(ARQUIVO_DESCLASSIFICACOES, index=False, encoding="utf-8-sig")
  except Exception as e:
    print(f"Erro ao salvar desclassificações: {e}")


def normalizar_chave_categoria_customizada(motorista: str, placa: str = "") -> str:
  """Gera chave persistente para categoria manual por motorista + placa."""
  mot = DataUtils.normalizar_texto(motorista)
  plc = DataUtils.padronizar_placa(placa)
  return f"{mot}|||{plc}" if plc else mot


def carregar_categorias_customizadas() -> dict:
  """Carrega os mapeamentos manuais de categoria.

  Compatibilidade:
  - MOTORISTA_CHAVE = MOTORISTA|||PLACA: regra por motorista e placa.
  - MOTORISTA_CHAVE = MOTORISTA: regra antiga por motorista.
  """
  if os.path.exists(ARQUIVO_CATEGORIAS_CUSTOM):
    try:
      df = pd.read_csv(
          ARQUIVO_CATEGORIAS_CUSTOM, dtype=str, encoding="utf-8-sig"
      )
      if "MOTORISTA_CHAVE" in df.columns and "CATEGORIA_ESCOLHIDA" in df.columns:
        df = df.fillna("")
        return {
            str(k).strip().upper(): DataUtils.normalizar_texto(v)
            for k, v in zip(df["MOTORISTA_CHAVE"], df["CATEGORIA_ESCOLHIDA"])
            if str(k).strip() and str(v).strip()
        }
    except Exception as e:
      print(f"Erro ao carregar categorias customizadas: {e}")
  return {}


def salvar_categorias_customizadas(mapa: dict):
  """Salva as seleções manuais de categorias por motorista e placa."""
  try:
    garantir_diretorio()
    df = pd.DataFrame(
        [
            [str(k).strip().upper(), DataUtils.normalizar_texto(v)]
            for k, v in (mapa or {}).items()
            if str(k).strip() and str(v).strip()
        ],
        columns=["MOTORISTA_CHAVE", "CATEGORIA_ESCOLHIDA"],
    )
    df.to_csv(ARQUIVO_CATEGORIAS_CUSTOM, index=False, encoding="utf-8-sig")
  except Exception as e:
    print(f"Erro ao salvar categorias customizadas: {e}")


def carregar_frota_customizada() -> pd.DataFrame:
  if os.path.exists(ARQUIVO_FROTA_CUSTOM):
    try:
      df = pd.read_csv(ARQUIVO_FROTA_CUSTOM, dtype=str, encoding="utf-8-sig")
      if "CAVALO" in df.columns and "TIPO" in df.columns:
        return df
    except Exception as e:
      print(f"Erro ao carregar frota customizada: {e}")
  return pd.DataFrame(columns=["CAVALO", "TIPO"])


def salvar_frota_customizada(df: pd.DataFrame):
  try:
    garantir_diretorio()
    df.to_csv(ARQUIVO_FROTA_CUSTOM, index=False, encoding="utf-8-sig")
  except Exception as e:
    print(f"Erro ao salvar frota customizada: {e}")


def carregar_motoristas_customizados() -> pd.DataFrame:
  if os.path.exists(ARQUIVO_MOTORISTAS_CUSTOM):
    try:
      df = pd.read_csv(
          ARQUIVO_MOTORISTAS_CUSTOM, dtype=str, encoding="utf-8-sig"
      )
      if (
          "MOTORISTAS" in df.columns
          and "TIPO" in df.columns
          and "BASE" in df.columns
      ):
        return df
    except Exception as e:
      print(f"Erro ao carregar motoristas customizados: {e}")
  return pd.DataFrame(columns=["MOTORISTAS", "TIPO", "BASE"])


def salvar_motoristas_customizados(df: pd.DataFrame):
  try:
    garantir_diretorio()
    df.to_csv(ARQUIVO_MOTORISTAS_CUSTOM, index=False, encoding="utf-8-sig")
  except Exception as e:
    print(f"Erro ao salvar motoristas customizados: {e}")


def carregar_inativos() -> dict:
  """Carrega as placas e motoristas marcados como INATIVOS do arquivo CSV."""
  if os.path.exists(ARQUIVO_INATIVOS):
    try:
      df = pd.read_csv(ARQUIVO_INATIVOS, dtype=str, encoding="utf-8-sig")
      if "DATA_INATIVACAO" not in df.columns:
        df["DATA_INATIVACAO"] = ""

      df_mot = df[df["TIPO"] == "MOTORISTA"].dropna(subset=["VALOR"])
      df_placa = df[df["TIPO"] == "PLACA"].dropna(subset=["VALOR"])

      mots = dict(zip(df_mot["VALOR"], df_mot["DATA_INATIVACAO"].fillna("")))
      placas = dict(
          zip(df_placa["VALOR"], df_placa["DATA_INATIVACAO"].fillna(""))
      )
      return {"MOTORISTA": mots, "PLACA": placas}
    except Exception as e:
      print(f"Erro ao carregar inativos: {e}")
  return {"MOTORISTA": {}, "PLACA": {}}


def alternar_inativo(tipo: str, valor: str, inativar: bool = True):
  """Grava ou remove o motorista/placa do arquivo de inativos."""
  garantir_diretorio()
  if tipo == "MOTORISTA":
    valor = DataUtils.normalizar_texto(valor)
  else:
    valor = DataUtils.padronizar_placa(valor)

  df = pd.DataFrame(columns=["TIPO", "VALOR", "DATA_INATIVACAO"])
  if os.path.exists(ARQUIVO_INATIVOS):
    try:
      df = pd.read_csv(ARQUIVO_INATIVOS, dtype=str, encoding="utf-8-sig")
      if "DATA_INATIVACAO" not in df.columns:
        df["DATA_INATIVACAO"] = ""
    except Exception:
      pass

  mask = (df["TIPO"] == tipo) & (df["VALOR"] == valor)
  if inativar:
    if not mask.any():
      data_atual = datetime.now().strftime("%d/%m/%Y")
      novo_reg = pd.DataFrame(
          [{"TIPO": tipo, "VALOR": valor, "DATA_INATIVACAO": data_atual}]
      )
      df = pd.concat([df, novo_reg], ignore_index=True)
  else:
    df = df[~mask]

  try:
    df.to_csv(ARQUIVO_INATIVOS, index=False, encoding="utf-8-sig")
  except Exception as e:
    print(f"Erro ao salvar inativos: {e}")


@dataclass
class AppConfig:
  CAMINHO_PRECOS: str = "Pasta2.xlsx"
  CAMINHO_FROTA: str = "frota.xlsx"
  CAMINHO_MOTORISTAS: str = "Pasta4.xlsx"
  CAMINHO_ABASTECIMENTOS: str = "uah_abastecimentos_3.xlsx"

  def _resolver_caminho_real(self, nome_arquivo: str) -> str:
    if os.path.isfile(nome_arquivo):
      return nome_arquivo

    if os.path.exists("."):
      for arquivo_real in os.listdir("."):
        if arquivo_real.lower() == nome_arquivo.lower():
          return arquivo_real
    return nome_arquivo

  def criar_arquivos_teste_se_ausentes(self):
    if not os.path.isfile(self.CAMINHO_PRECOS):
      df_p = pd.DataFrame({
          "TIPO": ["TRUCK", "BITRUCK", "BITRUCK UBERABA", "CARRETA"],
          "MEDIA": [2.5, 2.2, 2.1, 1.8],
          "PREMIO": [500.0, 600.0, 650.0, 700.0],
      })
      df_p.to_excel(self.CAMINHO_PRECOS, index=False)

    if not os.path.isfile(self.CAMINHO_FROTA):
      df_f = pd.DataFrame({
          "CAVALO": [
              "ABC1234",
              "DEF5678",
              "GHI9012",
              "WES0001",
              "WES0002",
              "WES0003",
              "WES0004",
          ],
          "TIPO": [
              "TRUCK",
              "BITRUCK",
              "CARRETA",
              "TRUCK",
              "BITRUCK",
              "CARRETA",
              "BITRUCK UBERABA",
          ],
      })
      df_f.to_excel(self.CAMINHO_FROTA, index=False)

    if not os.path.isfile(self.CAMINHO_MOTORISTAS):
      df_m = pd.DataFrame([
          ["MOTORISTAS", "TIPO", "BASE"],
          ["JOAO SILVA", "TRUCK", "CIANORTE"],
          ["MARIA SOUZA", "BITRUCK", "UBERABA"],
          ["CARLOS ALVES", "CARRETA", "MARINGA"],
          ["WESLEI", "MULTIPLASCATEGORIAS", "CIANORTE"],
      ])
      df_m.to_excel(self.CAMINHO_MOTORISTAS, header=False, index=False)

    if not os.path.isfile(self.CAMINHO_ABASTECIMENTOS):
      df_a = pd.DataFrame({
          "DATA": [
              "01/08/2026",
              "05/08/2026",
              "10/08/2026",
              "15/08/2026",
              "01/08/2026",
              "05/08/2026",
              "10/08/2026",
              "15/08/2026",
          ],
          "PLACA": [
              "ABC1234",
              "ABC1234",
              "DEF5678",
              "DEF5678",
              "WES0001",
              "WES0002",
              "WES0003",
              "WES0004",
          ],
          "MOTORISTA": [
              "JOAO SILVA",
              "JOAO SILVA",
              "MARIA SOUZA",
              "MARIA SOUZA",
              "WESLEI",
              "WESLEI",
              "WESLEI",
              "WESLEI",
          ],
          "KM": [1000, 1500, 2000, 2600, 100, 500, 1000, 1600],
          "QTDE": [200, 180, 280, 270, 50, 100, 150, 200],
          "VALOR": [
              1200.0,
              1080.0,
              1680.0,
              1620.0,
              300.0,
              600.0,
              900.0,
              1200.0,
          ],
      })
      df_a.to_excel(self.CAMINHO_ABASTECIMENTOS, index=False)

  def verificar_arquivos(self):
    self.criar_arquivos_teste_se_ausentes()
    self.CAMINHO_PRECOS = self._resolver_caminho_real(self.CAMINHO_PRECOS)
    self.CAMINHO_FROTA = self._resolver_caminho_real(self.CAMINHO_FROTA)
    self.CAMINHO_MOTORISTAS = self._resolver_caminho_real(
        self.CAMINHO_MOTORISTAS
    )
    self.CAMINHO_ABASTECIMENTOS = self._resolver_caminho_real(
        self.CAMINHO_ABASTECIMENTOS
    )


# ================================================================
# UTILITÁRIOS E TRATAMENTO DE TEXTO / PLACAS / DATAS
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
  def encontrar_coluna(cls, df: pd.DataFrame, alternativas: list) -> Optional[str]:
    mapa = {cls.coluna_canonica(col): col for col in df.columns}
    for alt in alternativas:
      chave = cls.coluna_canonica(alt)
      if chave in mapa:
        return mapa[chave]
    return None

  @staticmethod
  def converter_data(valor):
    if pd.isna(valor):
      return pd.NaT

    if isinstance(valor, (pd.Timestamp, datetime, np.datetime64)):
      try:
        return pd.Timestamp(valor).normalize()
      except Exception:
        return pd.NaT

    if isinstance(valor, (int, float, np.integer, np.floating)):
      try:
        if float(valor) > 20000:
          return pd.to_datetime(
              float(valor), unit="D", origin="1899-12-30", errors="coerce"
          ).normalize()
      except Exception:
        pass

    texto = str(valor).strip()
    if not texto:
      return pd.NaT

    try:
      if re.fullmatch(r"\d{2}/\d{2}/\d{4}(?:\s+.*)?", texto):
        return pd.to_datetime(
            texto[:10], format="%d/%m/%Y", errors="coerce"
        ).normalize()
      if re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:.*)?", texto):
        return pd.to_datetime(
            texto[:10], format="%Y-%m-%d", errors="coerce"
        ).normalize()
      if re.fullmatch(r"\d{8}", texto[:8]):
        return pd.Timestamp(
            year=int(texto[:4]), month=int(texto[4:6]), day=int(texto[6:8])
        )
      return pd.NaT
    except Exception:
      return pd.NaT

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


def parse_data_filtro(val) -> Optional[pd.Timestamp]:
  if val is None:
    return None
  try:
    if pd.isna(val):
      return None
  except Exception:
    pass

  if isinstance(val, (pd.Timestamp, datetime, np.datetime64)):
    try:
      return pd.Timestamp(val).normalize()
    except Exception:
      return None

  if isinstance(val, (int, float, np.integer, np.floating)):
    try:
      n = float(val)
      if n > 20000:
        return pd.to_datetime(
            n, unit="D", origin="1899-12-30", errors="coerce"
        ).normalize()
    except Exception:
      return None

  texto = str(val).strip()
  if not texto:
    return None

  try:
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})", texto)
    if m:
      return pd.Timestamp(
          year=int(m.group(3)), month=int(m.group(2)), day=int(m.group(1))
      )

    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", texto)
    if m:
      return pd.Timestamp(
          year=int(m.group(1)), month=int(m.group(2)), day=int(m.group(3))
      )

    if re.fullmatch(r"\d{8}", texto[:8]):
      return pd.Timestamp(
          year=int(texto[:4]), month=int(texto[4:6]), day=int(texto[6:8])
      )
    return pd.NaT
  except Exception:
    return None


def criar_data_filtro(valor) -> pd.Timestamp:
  parsed = parse_data_filtro(valor)
  return parsed if parsed is not None else pd.NaT


def calcular_dias_ausencia(data_inicio, data_fim):
  """Calcula dias corridos inclusivos entre a data inicial e final."""
  dt_ini = parse_data_filtro(data_inicio)
  dt_fim = parse_data_filtro(data_fim)
  if dt_ini is None or dt_fim is None:
    return 0
  if pd.isna(dt_ini) or pd.isna(dt_fim) or pd.Timestamp(dt_fim) < pd.Timestamp(dt_ini):
    return 0
  return int((pd.Timestamp(dt_fim) - pd.Timestamp(dt_ini)).days + 1)


# ================================================================
# LEITURA DE PLANILHAS
# ================================================================
class DataLoader:

  def __init__(self, config: AppConfig):
    self.config = config

  def carregar_precos(self) -> pd.DataFrame:
    df = pd.read_excel(self.config.CAMINHO_PRECOS, sheet_name=0)

    col_tipo = DataUtils.encontrar_coluna(
        df, ["TIPO", "TIPO VEICULO", "TIPO DE VEICULO", "CATEGORIA"]
    )
    col_media = DataUtils.encontrar_coluna(
        df, ["MEDIA", "MÉDIA", "MEDIA KM/L", "MÉDIA KM/L"]
    )
    col_premio = DataUtils.encontrar_coluna(
        df,
        [
            "TOTAL",
            "PREMIO",
            "PRÊMIO",
            "VALOR",
            "PREMIAÇÃO",
            "PREMIACAO",
            "VALOR PREMIO",
            "BONUS",
            "R$",
        ],
    )

    if None in (col_tipo, col_media, col_premio):
      for i in range(min(len(df), 10)):
        vals = [str(x).strip().upper() for x in df.iloc[i].tolist()]
        if any("TIPO" in v or "CATEGORIA" in v for v in vals) and any(
            "MEDIA" in v for v in vals
        ):
          df.columns = [DataUtils.normalizar_texto(c) for c in df.iloc[i]]
          df = df.iloc[i + 1 :].reset_index(drop=True)
          col_tipo = DataUtils.encontrar_coluna(
              df, ["TIPO", "TIPO VEICULO", "TIPO DE VEICULO", "CATEGORIA"]
          )
          col_media = DataUtils.encontrar_coluna(df, ["MEDIA", "MÉDIA"])
          col_premio = DataUtils.encontrar_coluna(
              df,
              [
                  "TOTAL",
                  "PREMIO",
                  "PRÊMIO",
                  "VALOR",
                  "PREMIAÇÃO",
                  "BONUS",
                  "R$",
              ],
          )
          break

    if None in (col_tipo, col_media, col_premio) and df.shape[1] >= 3:
      df.columns = [
          "TIPO",
          "MEDIA",
          "PREMIO",
      ] + list(df.columns[3:])
      col_tipo, col_media, col_premio = "TIPO", "MEDIA", "PREMIO"

    resultado = pd.DataFrame({
        "TIPO": df[col_tipo].apply(DataUtils.normalizar_texto),
        "MEDIA": df[col_media].apply(DataUtils.converter_numero),
        "PREMIO": df[col_premio].apply(DataUtils.converter_numero),
    }).dropna(subset=["MEDIA", "PREMIO"])

    resultado["TIPO"] = resultado["TIPO"].replace({"TOCO": "TRUCK"})
    return resultado[resultado["TIPO"] != ""].reset_index(drop=True)

  def carregar_frota(self) -> Tuple[pd.DataFrame, dict]:
    df_excel = pd.read_excel(self.config.CAMINHO_FROTA, sheet_name=0)
    col_placa = DataUtils.encontrar_coluna(
        df_excel, ["CAVALO", "PLACA", "PLACA CAVALO", "PLACA DO CAVALO"]
    )
    col_tipo = DataUtils.encontrar_coluna(
        df_excel, ["TIPO", "TIPO VEICULO", "TIPO DE VEICULO", "CATEGORIA"]
    )

    df_base = pd.DataFrame({
        "CAVALO": df_excel[col_placa].apply(DataUtils.padronizar_placa),
        "TIPO": df_excel[col_tipo].apply(DataUtils.normalizar_texto),
    })

    df_custom = carregar_frota_customizada()
    if not df_custom.empty:
      df_custom["CAVALO"] = df_custom["CAVALO"].apply(
          DataUtils.padronizar_placa
      )
      df_custom["TIPO"] = df_custom["TIPO"].apply(DataUtils.normalizar_texto)
      df_base = pd.concat([df_base, df_custom], ignore_index=True)

    resultado = pd.DataFrame({
        "PLACA_PADRONIZADA": df_base["CAVALO"],
        "TIPO": df_base["TIPO"],
    })
    resultado = resultado[resultado["PLACA_PADRONIZADA"] != ""].copy()
    resultado["TIPO"] = resultado["TIPO"].replace({"TOCO": "TRUCK"})
    resultado = resultado.drop_duplicates("PLACA_PADRONIZADA", keep="last")

    inativos_dict = carregar_inativos().get("PLACA", {})
    resultado["STATUS"] = resultado["PLACA_PADRONIZADA"].apply(
        lambda x: "INATIVO" if x in inativos_dict else "ATIVO"
    )
    resultado["DATA_INATIVACAO"] = resultado["PLACA_PADRONIZADA"].apply(
        lambda x: inativos_dict.get(x, "")
    )

    mapa = dict(zip(resultado["PLACA_PADRONIZADA"], resultado["TIPO"]))
    return resultado, mapa

  def carregar_cadastro_motoristas(self) -> pd.DataFrame:
    bruto = pd.read_excel(
        self.config.CAMINHO_MOTORISTAS, sheet_name=0, header=None
    )
    cab_idx, linha_cab = None, None

    for i in range(min(len(bruto), 15)):
      vals = [str(x).strip().upper() for x in bruto.iloc[i].tolist()]
      if "MOTORISTAS" in vals and "TIPO" in vals:
        cab_idx, linha_cab = i, vals
        break

    if cab_idx is None:
      cadastro = pd.DataFrame({
          "MOTORISTA_CADASTRO": bruto.iloc[:, 0].apply(
              DataUtils.normalizar_texto
          ),
          "TIPO_CADASTRO": bruto.iloc[:, 1]
          .apply(DataUtils.normalizar_texto)
          .replace({"TOCO": "TRUCK"}),
          "BASE_CADASTRO": (
              bruto.iloc[:, 2].apply(DataUtils.normalizar_texto)
              if bruto.shape[1] > 2
              else ""
          ),
      })
    else:
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
      cadastro = cadastro[
          (cadastro["MOTORISTA_CADASTRO"] != "")
          & (cadastro["TIPO_CADASTRO"] != "")
      ][["MOTORISTA_CADASTRO", "TIPO_CADASTRO", "BASE_CADASTRO"]]

    df_custom = carregar_motoristas_customizados()
    if not df_custom.empty:
      df_c_fmt = pd.DataFrame({
          "MOTORISTA_CADASTRO": df_custom["MOTORISTAS"].apply(
              DataUtils.normalizar_texto
          ),
          "TIPO_CADASTRO": df_custom["TIPO"]
          .apply(DataUtils.normalizar_texto)
          .replace({"TOCO": "TRUCK"}),
          "BASE_CADASTRO": df_custom["BASE"].apply(DataUtils.normalizar_texto),
      })
      cadastro = pd.concat([cadastro, df_c_fmt], ignore_index=True)

    cadastro["EH_FOLGUISTA"] = cadastro["TIPO_CADASTRO"].eq("FOLGUISTA")
    cadastro = cadastro.drop_duplicates("MOTORISTA_CADASTRO", keep="last")

    inativos_dict = carregar_inativos().get("MOTORISTA", {})
    cadastro["STATUS"] = cadastro["MOTORISTA_CADASTRO"].apply(
        lambda x: "INATIVO" if x in inativos_dict else "ATIVO"
    )
    cadastro["DATA_INATIVACAO"] = cadastro["MOTORISTA_CADASTRO"].apply(
        lambda x: inativos_dict.get(x, "")
    )

    return cadastro

  def carregar_abastecimentos(self, mapa_frota: dict) -> pd.DataFrame:
    df = pd.read_excel(
        self.config.CAMINHO_ABASTECIMENTOS,
        sheet_name=0,
        dtype=str,
        keep_default_na=False,
    )

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
    col_motorista = DataUtils.encontrar_coluna(
        df, ["CONDUTOR", "MOTORISTA", "MOTORISTAS"]
    )
    col_data = DataUtils.encontrar_coluna(
        df,
        [
            "DATA",
            "Data",
            "DATA ABASTECIMENTO",
            "DATA_ABASTECIMENTO",
            "DATA DO ABASTECIMENTO",
            "DATA/HORA",
            "DATA HORA",
            "DATA EMISSAO",
            "DT_ABASTECIMENTO",
            "DT ABAST",
        ],
    )

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

    if col_data:
      resultado["DATA_ORIGINAL"] = resultado[col_data]
      resultado["DATA_FILTRO"] = resultado[col_data].apply(criar_data_filtro)
      resultado["DATA_NUM"] = resultado["DATA_FILTRO"]
      resultado["DATA"] = resultado["DATA_FILTRO"]
    else:
      resultado["DATA_ORIGINAL"] = pd.NaT
      resultado["DATA_NUM"] = pd.NaT
      resultado["DATA_FILTRO"] = pd.NaT

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
    base["KM_RODADO_EVENTO"] = base["KM_ATUAL_NUM"] - base["KM_ANTERIOR"]

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
      self,
      media: float,
      tipo: str,
      precos: pd.DataFrame,
      base_motorista: str = "",
  ) -> dict:
    if pd.isna(media) or media <= 0:
      return {
          "MEDIA_FAIXA": np.nan,
          "PREMIO": 0.0,
          "STATUS_PREMIO": "SEM MEDIA",
      }

    tipo_norm = DataUtils.normalizar_texto(tipo)
    base_norm = DataUtils.normalizar_texto(base_motorista)

    tabela = pd.DataFrame()

    if ("BITRUCK" in tipo_norm or tipo_norm == "BITRUCK") and (
        "UBERABA" in base_norm or "UBERABA" in tipo_norm
    ):
      tabela = precos[precos["TIPO"].str.contains("UBERABA", na=False)].copy()

    if tabela.empty:
      tabela = precos[precos["TIPO"] == tipo_norm].copy()

    if tabela.empty:
      tabela = precos[precos["TIPO"].str.contains(tipo_norm, na=False)].copy()

    if tabela.empty and " " in tipo_norm:
      primeira_palavra = tipo_norm.split()[0]
      tabela = precos[
          precos["TIPO"].str.contains(primeira_palavra, na=False)
      ].copy()

    if tabela.empty:
      tabela = precos.copy()

    if tabela.empty:
      return {
          "MEDIA_FAIXA": np.nan,
          "PREMIO": 0.0,
          "STATUS_PREMIO": "SEM FAIXA",
      }

    tabela = tabela.sort_values("MEDIA").copy()
    media_min = float(tabela["MEDIA"].min())
    media_max = float(tabela["MEDIA"].max())
    media_class = round(float(media), 2)

    if media_class < media_min:
      return {
          "MEDIA_FAIXA": media_min,
          "PREMIO": 0.0,
          "STATUS_PREMIO": "DESCLASSIFICADO",
      }

    if media_class >= media_max:
      linha_max = tabela.iloc[-1]
      return {
          "MEDIA_FAIXA": float(linha_max["MEDIA"]),
          "PREMIO": float(linha_max["PREMIO"]),
          "STATUS_PREMIO": "TETO",
      }

    faixa_atingida = tabela[tabela["MEDIA"] <= media_class]
    if not faixa_atingida.empty:
      linha = faixa_atingida.iloc[-1]
      return {
          "MEDIA_FAIXA": float(linha["MEDIA"]),
          "PREMIO": float(linha["PREMIO"]),
          "STATUS_PREMIO": "OK",
      }

    return {
        "MEDIA_FAIXA": np.nan,
        "PREMIO": 0.0,
        "STATUS_PREMIO": "DESCLASSIFICADO",
    }

  def calcular_premios(
      self,
      eventos: pd.DataFrame,
      precos: pd.DataFrame,
      cadastro: pd.DataFrame,
      categorias_customizadas: dict = None,
  ) -> pd.DataFrame:
    colunas_saida = [
        "MOTORISTA_CHAVE",
        "CATEGORIA_ABASTECIMENTO",
        "MOTORISTA",
        "BASE",
        "CATEGORIA",
        "STATUS_MOTORISTA",
        "KM_TOTAL",
        "LITROS_TOTAL",
        "KM_CAT",
        "LITROS_CAT",
        "QTD_ABASTECIMENTOS",
        "PLACAS",
        "MEDIA_CALCULADA",
        "MEDIA_FAIXA",
        "PREMIO",
        "STATUS_PREMIO",
        "PREMIO_BRUTO",
        "DIAS_AUSENCIA",
        "DIAS_EFETIVOS",
        "MOTIVO_DESCLASSIFICACAO",
    ]

    if categorias_customizadas is None:
      categorias_customizadas = {}

    if eventos.empty:
      return pd.DataFrame(columns=colunas_saida)

    base = eventos[eventos["REGISTRO_CONSUMO_VALIDO"]].copy()
    if base.empty:
      return pd.DataFrame(columns=colunas_saida)

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
            "STATUS",
        ]],
        on="MOTORISTA_CHAVE",
        how="left",
    )
    base["TIPO_CADASTRO"] = base["TIPO_CADASTRO"].fillna("")
    base["EH_FOLGUISTA"] = base["EH_FOLGUISTA"].fillna(False)
    base["STATUS"] = base["STATUS"].fillna("ATIVO")

    registros = []
    for _, grupo in base.groupby("MOTORISTA_CHAVE", sort=False):
      mot_chave = grupo["MOTORISTA_CHAVE"].iloc[0]
      eh_folguista = grupo["EH_FOLGUISTA"].iloc[0]
      tipo_cad = grupo["TIPO_CADASTRO"].iloc[0]
      custom_legacy = categorias_customizadas.get(mot_chave, "")

      # Regra nova: a categoria manual pode ser definida por MOTORISTA + PLACA.
      # Regra antiga continua aceita quando só houver MOTORISTA_CHAVE.
      grupo = grupo.copy()
      grupo["CHAVE_CAT_MANUAL"] = (
          grupo.apply(
              lambda r: normalizar_chave_categoria_customizada(
                  r["MOTORISTA_CHAVE"], r["PLACA_PADRONIZADA"]
              ),
              axis=1,
          )
      )
      grupo["CATEGORIA_MANUAL"] = grupo["CHAVE_CAT_MANUAL"].map(
          categorias_customizadas
      ).fillna("")

      if custom_legacy:
        grupo["CATEGORIA_MANUAL"] = grupo["CATEGORIA_MANUAL"].replace("", custom_legacy)

      if eh_folguista and not custom_legacy and not grupo["CATEGORIA_MANUAL"].astype(bool).any():
        soma = grupo.groupby("CATEGORIA_ABASTECIMENTO")["KM_CONSUMO"].sum()
        cat_elegivel = (
            soma.idxmax()
            if not soma.empty
            else (tipo_cad if tipo_cad else "TRUCK")
        )
        grupo["CATEGORIA_ELEGIVEL"] = cat_elegivel
        grupo["USA_CATEGORIA_MANUAL"] = False
      else:
        grupo["CATEGORIA_ELEGIVEL"] = np.where(
            grupo["CATEGORIA_MANUAL"].astype(str).str.strip() != "",
            grupo["CATEGORIA_MANUAL"],
            tipo_cad if tipo_cad else grupo["CATEGORIA_ABASTECIMENTO"].iloc[0],
        )
        grupo["USA_CATEGORIA_MANUAL"] = (
            grupo["CATEGORIA_MANUAL"].astype(str).str.strip() != ""
        )

      registros.append(grupo)

    if not registros:
      return pd.DataFrame(columns=colunas_saida)

    df_eleg = pd.concat(registros, ignore_index=True)

    df_eleg["EH_CATEGORIA_ELEGIVEL"] = np.where(
        df_eleg["USA_CATEGORIA_MANUAL"],
        True,
        df_eleg["CATEGORIA_ABASTECIMENTO"] == df_eleg["CATEGORIA_ELEGIVEL"],
    )
    df_eleg["KM_CAT_ELEGIVEL"] = np.where(
        df_eleg["EH_CATEGORIA_ELEGIVEL"], df_eleg["KM_CONSUMO"], 0
    )
    df_eleg["LITROS_CAT_ELEGIVEL"] = np.where(
        df_eleg["EH_CATEGORIA_ELEGIVEL"], df_eleg["LITROS_CONSUMO"], 0
    )

    resumo = df_eleg.groupby("MOTORISTA_CHAVE", as_index=False).agg(
        MOTORISTA=("CONDUTOR_NORMALIZADO", "first"),
        BASE=("BASE_CADASTRO", "first"),
        CATEGORIA=("CATEGORIA_ELEGIVEL", "first"),
        STATUS_MOTORISTA=("STATUS", "first"),
        KM_TOTAL=("KM_CONSUMO", "sum"),
        LITROS_TOTAL=("LITROS_CONSUMO", "sum"),
        KM_CAT=("KM_CAT_ELEGIVEL", "sum"),
        LITROS_CAT=("LITROS_CAT_ELEGIVEL", "sum"),
        QTD_ABASTECIMENTOS=("LITROS_CONSUMO", "count"),
        PLACAS=("PLACA_PADRONIZADA", lambda s: " | ".join(sorted(set(s)))),
    )

    resumo["MEDIA_CALCULADA"] = np.where(
        resumo["LITROS_CAT"] > 0,
        resumo["KM_CAT"] / resumo["LITROS_CAT"],
        np.nan,
    )

    faixas = resumo.apply(
        lambda r: self.faixa_mais_proxima(
            r["MEDIA_CALCULADA"],
            r["CATEGORIA"],
            precos,
            base_motorista=r["BASE"],
        ),
        axis=1,
    )

    resumo_df = pd.concat(
        [resumo.reset_index(drop=True), pd.DataFrame(list(faixas))], axis=1
    )
    resumo_df["PREMIO_BRUTO"] = resumo_df["PREMIO"]
    resumo_df["DIAS_AUSENCIA"] = 0
    resumo_df["DIAS_EFETIVOS"] = 30
    resumo_df["MOTIVO_DESCLASSIFICACAO"] = np.where(
        resumo_df["STATUS_PREMIO"] == "DESCLASSIFICADO",
        "Média de consumo abaixo do limite mínimo da categoria",
        "Elegível / Em conformidade",
    )

    return resumo_df


# ================================================================
# GERADOR DE DADOS DE MÚLTIPLAS PLACAS
# ================================================================
def gerar_dados_multiplas_placas(eventos: pd.DataFrame):
  if eventos.empty:
    return pd.DataFrame(), []

  base = eventos[eventos["REGISTRO_CONSUMO_VALIDO"]].copy()
  if base.empty:
    return pd.DataFrame(), []

  base["MOTORISTA_CHAVE"] = (
      base["CONDUTOR_NORMALIZADO"]
      .fillna("")
      .astype(str)
      .str.strip()
      .str.upper()
  )
  base["CATEGORIA_ABASTECIMENTO"] = (
      base["TIPO"].fillna("GERAL").replace({"TOCO": "TRUCK"})
  )

  placas_por_motorista = base.groupby("MOTORISTA_CHAVE")[
      "PLACA_PADRONIZADA"
  ].nunique()
  mots_multi = placas_por_motorista[placas_por_motorista > 1].index.tolist()

  base_multi = base[base["MOTORISTA_CHAVE"].isin(mots_multi)].copy()
  if base_multi.empty:
    return pd.DataFrame(), []

  resumo_placa = base_multi.groupby(
      [
          "MOTORISTA_CHAVE",
          "CONDUTOR_NORMALIZADO",
          "PLACA_PADRONIZADA",
          "CATEGORIA_ABASTECIMENTO",
      ],
      as_index=False,
  ).agg(
      KM_TOTAL=("KM_CONSUMO", "sum"),
      LITROS_TOTAL=("LITROS_CONSUMO", "sum"),
      QTD_ABASTECIMENTOS=("LITROS_CONSUMO", "count"),
  )

  resumo_placa["MEDIA_CALCULADA"] = np.where(
      resumo_placa["LITROS_TOTAL"] > 0,
      resumo_placa["KM_TOTAL"] / resumo_placa["LITROS_TOTAL"],
      np.nan,
  )

  df_exibicao = resumo_placa.rename(columns={
      "CONDUTOR_NORMALIZADO": "MOTORISTA",
      "PLACA_PADRONIZADA": "PLACA",
      "CATEGORIA_ABASTECIMENTO": "CATEGORIA_VEICULO",
      "KM_TOTAL": "KM TOTAL",
      "LITROS_TOTAL": "LITROS TOTAL",
      "MEDIA_CALCULADA": "MÉDIA (KM/L)",
      "QTD_ABASTECIMENTOS": "QTD ABASTECIMENTOS",
  })[[
      "MOTORISTA",
      "PLACA",
      "CATEGORIA_VEICULO",
      "KM TOTAL",
      "LITROS TOTAL",
      "MÉDIA (KM/L)",
      "QTD ABASTECIMENTOS",
  ]]

  df_exibicao["MÉDIA (KM/L)"] = df_exibicao["MÉDIA (KM/L)"].apply(
      lambda x: f"{x:.2f}".replace(".", ",") if pd.notna(x) else "-"
  )

  lista_nomes_mots = sorted(
      base_multi["CONDUTOR_NORMALIZADO"].unique().tolist()
  )
  return df_exibicao, lista_nomes_mots


# ================================================================
# CONSTANTES DE CRITÉRIOS DE DESCLASSIFICAÇÃO (PILAR 1)
# ================================================================
CRITERIOS_PILAR_1 = [
    (
        "1 - Controle de velocidade, limite máx de 80 km/h (1 Ponto/evento até"
        " 129)"
    ),
    (
        "2 - Controle de jornada, macros e intervalos incorretos (1"
        " Ponto/evento até 129)"
    ),
    "3 - Não realização correta do check list diário (1 Ponto/evento até 129)",
    (
        "4 - Deszelo com documentação e comprovantes de carga/descarga (1"
        " Ponto/evento até 129)"
    ),
    (
        "5 - Distração com freadas bruscas, risco de colisão ou manobra perigosa"
        " [DESCLASSIFICADO]"
    ),
    "6 - Uso do celular em direção [DESCLASSIFICADO]",
    "7 - Estacionar ou parada com veículo em L [DESCLASSIFICADO]",
    "8 - Identificação de ausência do motorista [DESCLASSIFICADO]",
    "9 - Paradas em locais proibidos [DESCLASSIFICADO]",
    "10 - Ausência do cinto de segurança em direção [DESCLASSIFICADO]",
    "11 - Vedar as câmeras [DESCLASSIFICADO]",
    (
        "12 - Comportamento inadequado do motorista dentro e fora do veículo"
        " [DESCLASSIFICADO]"
    ),
    (
        "13 - Picos acima de 80 km/h, com mais de 30 infrações por período"
        " [DESCLASSIFICADO]"
    ),
    "14 - Não cumprir determinações e escalas estipuladas [DESCLASSIFICADO]",
    (
        "15 - Erros operacionais (Carregamento/Descarregamento incorreto ou"
        " Derramamento/Contaminação) [DESCLASSIFICADO]"
    ),
]


# ================================================================
# RECALCULO DE AUSÊNCIAS E DESCLASSIFICAÇÕES
# ================================================================
def aplicar_regras_gerais(
    df_resumo_original: pd.DataFrame,
    df_ausencias: pd.DataFrame,
    df_desclassificacoes: pd.DataFrame,
) -> pd.DataFrame:
  if df_resumo_original.empty:
    return df_resumo_original.copy()

  res = df_resumo_original.copy()

  motivos_iniciais = []
  for idx in res.index:
    st = res.at[idx, "STATUS_PREMIO"]
    if st == "DESCLASSIFICADO":
      motivos_iniciais.append(
          "Média de consumo abaixo do limite mínimo da categoria"
      )
    else:
      motivos_iniciais.append("Elegível / Em conformidade")
  res["MOTIVO_DESCLASSIFICACAO"] = motivos_iniciais

  if not df_ausencias.empty and "MOTORISTA" in df_ausencias.columns:
    df_aus_tmp = df_ausencias.copy()
    df_aus_tmp["DIAS"] = pd.to_numeric(
        df_aus_tmp["DIAS"], errors="coerce"
    ).fillna(0)
    soma_dias = df_aus_tmp.groupby("MOTORISTA")["DIAS"].sum().to_dict()

    res["DIAS_AUSENCIA"] = (
        res["MOTORISTA"].map(soma_dias).fillna(0).astype(int)
    )
    res["DIAS_EFETIVOS"] = np.maximum(0, 30 - res["DIAS_AUSENCIA"])
    res["PREMIO"] = res.apply(
        lambda r: max(0.0, r["PREMIO_BRUTO"] * (r["DIAS_EFETIVOS"] / 30.0)),
        axis=1,
    )
  else:
    res["DIAS_AUSENCIA"] = 0
    res["DIAS_EFETIVOS"] = 30
    res["PREMIO"] = res["PREMIO_BRUTO"]

  if (
      not df_desclassificacoes.empty
      and "MOTORISTA" in df_desclassificacoes.columns
  ):
    for idx in res.index:
      m_nome = res.at[idx, "MOTORISTA"]
      g = df_desclassificacoes[df_desclassificacoes["MOTORISTA"] == m_nome]
      if not g.empty:
        motivos_pilar1 = []
        diretos = g[g["TIPO_IMPACTO"] == "DESCLASSIFICADO"]
        if not diretos.empty:
          crit_limpos = [
              c.split("[")[0].strip() for c in diretos["CRITERIO"].unique()
          ]
          motivos_pilar1.append(f"Infração Pilar 1: {', '.join(crit_limpos)}")

        tot_pontos = pd.to_numeric(g["PONTOS"], errors="coerce").fillna(0).sum()
        if tot_pontos > 129:
          motivos_pilar1.append(
              f"Excesso de Pontos Pilar 1 ({int(tot_pontos)} pts - máx 129)"
          )

        if motivos_pilar1:
          res.at[idx, "PREMIO"] = 0.0
          res.at[idx, "STATUS_PREMIO"] = "DESCLASSIFICADO"

          mot_existente = res.at[idx, "MOTIVO_DESCLASSIFICACAO"]
          novo_mot = " | ".join(motivos_pilar1)
          if mot_existente and "Média" in mot_existente:
            res.at[idx, "MOTIVO_DESCLASSIFICACAO"] = (
                f"{mot_existente} + {novo_mot}"
            )
          else:
            res.at[idx, "MOTIVO_DESCLASSIFICACAO"] = novo_mot

  return res


# ================================================================
# GERADOR DE DATAFRAME EXCLUSIVO DO RH
# ================================================================
def gerar_tabela_rh(df_resumo: pd.DataFrame) -> pd.DataFrame:
  if df_resumo.empty or "MOTORISTA" not in df_resumo.columns:
    return pd.DataFrame(columns=["NOME", "FILIAL", "VALOR PAGO"])

  if "STATUS_MOTORISTA" in df_resumo.columns:
    df_rh = df_resumo[df_resumo["STATUS_MOTORISTA"] == "ATIVO"].copy()
  else:
    df_rh = df_resumo.copy()

  rh_df = pd.DataFrame()
  rh_df["NOME"] = df_rh["MOTORISTA"]
  rh_df["FILIAL"] = df_rh["BASE"].fillna("CIANORTE")

  def formatar_valor_pago(x):
    if pd.isna(x):
      return "R$ 0,00"
    try:
      return (
          f"R$ {float(x):,.2f}".replace(",", "X")
          .replace(".", ",")
          .replace("X", ".")
      )
    except Exception:
      return "R$ 0,00"

  rh_df["VALOR PAGO"] = df_rh["PREMIO"].map(formatar_valor_pago)
  return rh_df


# ================================================================
# FILTRAGEM E RECÁLCULO DO DASHBOARD POR DATA E DEMAIS CRITÉRIOS
# ================================================================
def aplicar_filtros(
    dt_ini,
    dt_fim,
    motorista,
    placa,
    categoria,
    filial,
    df_ausencias,
    df_desclassificacoes,
    mapa_cat_custom=None,
):
  if mapa_cat_custom is None:
    mapa_cat_custom = carregar_categorias_customizadas()

  evt_f = eventos.copy()
  det_f = abastecimentos.copy()

  d_i = parse_data_filtro(dt_ini)
  d_f = parse_data_filtro(dt_fim)

  def aplicar_periodo(df):
    if df.empty:
      return df

    out = df.copy()
    if "DATA_FILTRO" in out.columns:
      datas = pd.to_datetime(out["DATA_FILTRO"], errors="coerce").dt.normalize()
    elif "DATA_NUM" in out.columns:
      datas = pd.to_datetime(out["DATA_NUM"], errors="coerce").dt.normalize()
    elif "DATA_ORIGINAL" in out.columns:
      datas = out["DATA_ORIGINAL"].apply(parse_data_filtro)
    else:
      return out

    mascara = datas.notna()

    if d_i is not None and pd.notna(d_i):
      ini = pd.Timestamp(d_i).normalize()
      mascara &= datas >= ini

    if d_f is not None and pd.notna(d_f):
      fim_exclusivo = pd.Timestamp(d_f).normalize() + pd.Timedelta(days=1)
      mascara &= datas < fim_exclusivo

    out["DATA_FILTRO"] = datas
    return out.loc[mascara].copy()

  evt_f = aplicar_periodo(evt_f)
  det_f = aplicar_periodo(det_f)

  if motorista and motorista != "TODOS":
    m_norm = DataUtils.normalizar_texto(motorista)
    evt_f = evt_f[
        evt_f["CONDUTOR_NORMALIZADO"]
        .apply(DataUtils.normalizar_texto)
        .str.contains(m_norm, na=False)
    ]
    det_f = det_f[
        det_f["CONDUTOR_NORMALIZADO"]
        .apply(DataUtils.normalizar_texto)
        .str.contains(m_norm, na=False)
    ]

  if placa and placa.strip():
    p_norm = DataUtils.padronizar_placa(placa)
    evt_f = evt_f[evt_f["PLACA_PADRONIZADA"] == p_norm]
    det_f = det_f[det_f["PLACA_PADRONIZADA"] == p_norm]

  if categoria and categoria != "TODAS":
    c_norm = DataUtils.normalizar_texto(categoria)
    evt_f = evt_f[evt_f["TIPO_CALCULO"] == c_norm]
    det_f = det_f[
        det_f["TIPO"]
        .fillna("")
        .replace({"TOCO": "TRUCK"})
        .apply(DataUtils.normalizar_texto)
        == c_norm
    ]

  if filial and filial != "TODAS":
    f_norm = DataUtils.normalizar_texto(filial)
    mapa_filial = {}
    if not cadastro.empty:
      cad_tmp = cadastro.copy()
      for _, row in cad_tmp.iterrows():
        chave = DataUtils.normalizar_texto(row.get("MOTORISTA_CADASTRO", ""))
        base = DataUtils.normalizar_texto(row.get("BASE_CADASTRO", ""))
        if chave:
          mapa_filial[chave] = base

    evt_f = evt_f[
        evt_f["CONDUTOR_NORMALIZADO"].map(
            lambda x: mapa_filial.get(DataUtils.normalizar_texto(x), "")
        )
        == f_norm
    ]
    det_f = det_f[
        det_f["CONDUTOR_NORMALIZADO"].map(
            lambda x: mapa_filial.get(DataUtils.normalizar_texto(x), "")
        )
        == f_norm
    ]

  resumo_periodo = engine.calcular_premios(
      evt_f, precos, cadastro, categorias_customizadas=mapa_cat_custom
  )
  res_f = aplicar_regras_gerais(
      resumo_periodo, df_ausencias, df_desclassificacoes
  )

  df_multi, mots_multi = gerar_dados_multiplas_placas(evt_f)

  tot_premio = (
      float(res_f["PREMIO"].sum())
      if (not res_f.empty and "PREMIO" in res_f.columns)
      else 0.0
  )
  if pd.isna(tot_premio):
    tot_premio = 0.0

  tot_km = (
      float(res_f["KM_TOTAL"].sum())
      if (not res_f.empty and "KM_TOTAL" in res_f.columns)
      else 0.0
  )
  if pd.isna(tot_km):
    tot_km = 0.0

  tot_litros = (
      float(res_f["LITROS_TOTAL"].sum())
      if (not res_f.empty and "LITROS_TOTAL" in res_f.columns)
      else 0.0
  )
  if pd.isna(tot_litros):
    tot_litros = 0.0

  tot_gasto_combustivel = (
      float(det_f["VALOR_NUM"].sum())
      if (not det_f.empty and "VALOR_NUM" in det_f.columns)
      else 0.0
  )
  if pd.isna(tot_gasto_combustivel):
    tot_gasto_combustivel = 0.0

  tot_media_geral = (tot_km / tot_litros) if tot_litros > 0 else 0.0
  if pd.isna(tot_media_geral):
    tot_media_geral = 0.0

  tot_mots = len(res_f)

  res_view = res_f.copy()
  if not res_view.empty:

    def fmt_premio(x):
      if pd.isna(x):
        return "R$ 0,00"
      try:
        return (
            f"R$ {float(x):,.2f}".replace(",", "X")
            .replace(".", ",")
            .replace("X", ".")
        )
      except Exception:
        return "R$ 0,00"

    def fmt_media(x):
      if pd.isna(x):
        return "-"
      try:
        return f"{float(x):.2f}".replace(".", ",")
      except Exception:
        return str(x)

    if "PREMIO" in res_view.columns:
      res_view["PREMIO"] = res_view["PREMIO"].map(fmt_premio)
    if "PREMIO_BRUTO" in res_view.columns:
      res_view["PREMIO_BRUTO"] = res_view["PREMIO_BRUTO"].map(fmt_premio)
    if "MEDIA_CALCULADA" in res_view.columns:
      res_view["MEDIA_CALCULADA"] = res_view["MEDIA_CALCULADA"].map(fmt_media)
    if "MEDIA_FAIXA" in res_view.columns:
      res_view["MEDIA_FAIXA"] = res_view["MEDIA_FAIXA"].map(fmt_media)

  col_data_detalhe = (
      "DATA_FILTRO" if "DATA_FILTRO" in det_f.columns else "DATA_NUM"
  )
  det_view = det_f.sort_values(
      [col_data_detalhe, "_ORDEM_ORIGINAL"], kind="stable"
  ).copy()
  det_view = det_view.drop(
      columns=[c for c in ["_DATA_ORDENACAO"] if c in det_view.columns],
      errors="ignore",
  )

  rh_view = gerar_tabela_rh(res_f)

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
  f_km = (
      f"{tot_km:,.1f} km".replace(",", "X")
      .replace(".", ",")
      .replace("X", ".")
  )
  f_litros = (
      f"{tot_litros:,.1f} L".replace(",", "X")
      .replace(".", ",")
      .replace("X", ".")
  )
  f_media = f"{tot_media_geral:.2f} km/L".replace(".", ",")
  f_mots = f"{tot_mots}"

  dropdown_mots_multi = gr.Dropdown(
      choices=mots_multi if mots_multi else ["NENHUM MOTORISTA"],
      value=mots_multi[0] if mots_multi else None,
      interactive=True,
  )

  return (
      f_premio,
      f_gasto_comb,
      f_km,
      f_litros,
      f_media,
      f_mots,
      res_view,
      rh_view,
      det_view,
      df_multi,
      dropdown_mots_multi,
      res_f,
  )


# ================================================================
# GERADOR DE RECIBOS DE PREMIAÇÃO
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

  km_raw = row_data.get("KM_CAT", row_data.get("KM_TOTAL", 0))
  try:
    km_val = (
        "0" if pd.isna(km_raw) else f"{float(km_raw):,.0f}".replace(",", ".")
    )
  except Exception:
    km_val = "0"

  dias_ef = row_data.get("DIAS_EFETIVOS", 30)
  dias_efetivos_val = 30 if pd.isna(dias_ef) else int(dias_ef)

  media_raw = row_data.get("MEDIA_CALCULADA", np.nan)
  if pd.isna(media_raw):
    media_val = "-"
  else:
    try:
      media_val = f"{float(media_raw):.2f}".replace(".", ",")
    except Exception:
      media_val = str(media_raw)

  premio_raw = row_data.get("PREMIO", 0.0)
  premio_float = 0.0 if pd.isna(premio_raw) else float(premio_raw)
  val_total_str = (
      f"R$ {premio_float:,.2f}".replace(",", "X")
      .replace(".", ",")
      .replace("X", ".")
  )

  motivo_descl = row_data.get(
      "MOTIVO_DESCLASSIFICACAO", "Elegível / Em conformidade"
  )
  if pd.isna(motivo_descl) or not str(motivo_descl).strip():
    motivo_descl = "Elegível / Em conformidade"

  status_premio = row_data.get("STATUS_PREMIO", "OK")
  eh_desclassificado = (status_premio == "DESCLASSIFICADO") or (
      premio_float == 0.0 and motivo_descl != "Elegível / Em conformidade"
  )

  bg_motivo = "#FEE2E2" if eh_desclassificado else "#FFFFFF"
  fg_motivo = "#991B1B" if eh_desclassificado else "#000000"
  hdr_motivo_bg = "#EF4444" if eh_desclassificado else "#D0E0F0"
  hdr_motivo_fg = "#FFFFFF" if eh_desclassificado else "#000000"

  return f"""
    <div class="recibo-card" style="background-color: #FFFFFF; padding: 28px; border-radius: 12px; max-width: 650px; margin: 0 auto 20px auto; font-family: Arial, sans-serif; color: #000000; border: 1px solid #CBD5E1; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); page-break-after: always; break-after: page;">
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
            <tr style="border-bottom: 1px solid #000000; background-color: {bg_motivo};">
                <td style="background-color: {hdr_motivo_bg}; color: {hdr_motivo_fg}; padding: 4px 8px; border-right: 1px solid #000000; text-align: center;">MOTIVO DESCLASSIFICAÇÃO</td>
                <td style="padding: 4px 8px; text-align: center; color: {fg_motivo}; font-weight: bold;">{motivo_descl}</td>
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
  if not motorista_sel or motorista_sel in ("SELECIONE...", ""):
    return (
        "<div style='text-align: center; padding: 40px; color: #64748B; "
        "font-size: 15px;'>👉 Por favor, selecione um motorista ou a opção de "
        "TODOS DA FILIAL para gerar os recibos.</div>"
    )

  res_f = df_resumo.copy()

  if "STATUS_MOTORISTA" in res_f.columns:
    res_f = res_f[res_f["STATUS_MOTORISTA"] == "ATIVO"]

  if str(motorista_sel).startswith("TODOS"):
    if filial_sel and filial_sel != "TODAS":
      f_norm = DataUtils.normalizar_texto(filial_sel)
      res_f = res_f[res_f["BASE"].apply(DataUtils.normalizar_texto) == f_norm]

    lista_mots = sorted(list(res_f["MOTORISTA"].dropna().unique()))
    if not lista_mots:
      return (
          f"<div style='text-align: center; padding: 40px; color:"
          f" #EF4444;'>Nenhum motorista encontrado na filial '{filial_sel}'.</div>"
      )
  else:
    lista_mots = [motorista_sel]

  recibos_html = [f"""
    <style>
    @media print {{
        body * {{ visibility: hidden; }}
        .recibo-container, .recibo-container * {{ visibility: visible; }}
        .recibo-container {{ position: absolute; left: 0; top: 0; width: 100%; }}
        .no-print {{ display: none !important; }}
    }}
    </style>
    <div class="no-print" style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 12px 20px; border-radius: 8px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center;">
        <span style="font-size: 14px; font-weight: bold; color: #1E293B;">
            📄 Total de Recibos Prontos: <span style="color: #2563EB;">{len(lista_mots)}</span>
        </span>
        <button onclick="window.print()" style="background-color: #2563EB; color: #FFFFFF; border: none; padding: 8px 18px; border-radius: 6px; font-weight: bold; cursor: pointer; font-size: 13px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            🖨️ Imprimir Todos os Recibos ({len(lista_mots)})
        </button>
    </div>
    """]

  cards_html = []
  for m_nome in lista_mots:
    row = res_f[res_f["MOTORISTA"] == m_nome]
    if not row.empty:
      card_html = gerar_html_unico_recibo(
          row.iloc[0], m_nome, periodo_ini, periodo_fim, fator_c
      )
      cards_html.append(card_html)

  if not cards_html:
    return (
        "<div style='text-align: center; padding: 40px; color: #64748B;'>👉"
        " Nenhum recibo gerado (verifique se o motorista selecionado está"
        " Inativo).</div>"
    )

  return (
      "".join(recibos_html)
      + "<div class='recibo-container' style='display: flex; flex-direction:"
      " column; gap: 30px;'>"
      + "".join(cards_html)
      + "</div>"
  )


def gerar_opcoes_exclusao_descl(df_descl: pd.DataFrame):
  if df_descl.empty:
    return gr.Dropdown(
        choices=["Nenhum registro para excluir"],
        value="Nenhum registro para excluir",
        interactive=False,
    )
  opcoes = []
  for i, r in df_descl.reset_index(drop=True).iterrows():
    mot = r["MOTORISTA"]
    crit_curto = (
        str(r["CRITERIO"]).split("-")[0].strip()
        if "-" in str(r["CRITERIO"])
        else str(r["CRITERIO"])[:10]
    )
    opcoes.append(f"[{i}] {mot} - Critério {crit_curto}")
  return gr.Dropdown(choices=opcoes, value=opcoes[0], interactive=True)


# ================================================================
# EXECUÇÃO PRINCIPAL E MONTAGEM DA INTERFACE
# ================================================================
config = AppConfig()
config.verificar_arquivos()

loader = DataLoader(config)
engine = RewardEngine()

precos = loader.carregar_precos()
frota, mapa_frota = loader.carregar_frota()
cadastro = loader.carregar_cadastro_motoristas()
abastecimentos = loader.carregar_abastecimentos(mapa_frota)

eventos = engine.calcular_eventos_consumo(abastecimentos)

datas_validas = (
    abastecimentos["DATA_FILTRO"].dropna()
    if "DATA_FILTRO" in abastecimentos.columns
    else eventos["DATA_NUM"].dropna()
)
if not datas_validas.empty:
  min_date_default = datas_validas.min().strftime("%d/%m/%Y")
  max_date_default = datas_validas.max().strftime("%d/%m/%Y")
else:
  min_date_default = "01/01/2026"
  max_date_default = "31/12/2026"

df_ausencias_global = carregar_ausencias()
df_desclassificacoes_global = carregar_desclassificacoes()
mapa_cat_custom_global = carregar_categorias_customizadas()

(
    f_premio_init,
    f_gasto_init,
    f_km_init,
    f_litros_init,
    f_media_init,
    f_mots_init,
    res_view_init,
    rh_view_init,
    det_view_init,
    df_multi_init,
    mots_multi_drop_init,
    res_f_init,
) = aplicar_filtros(
    min_date_default,
    max_date_default,
    "TODOS",
    "",
    "TODAS",
    "TODAS",
    df_ausencias_global,
    df_desclassificacoes_global,
    mapa_cat_custom_global,
)

mots_lista = ["TODOS"] + sorted(list(res_f_init["MOTORISTA"].dropna().unique()))
cats_lista = ["TODAS"] + sorted(list(res_f_init["CATEGORIA"].dropna().unique()))
cats_precos_opcoes = sorted(list(precos["TIPO"].unique()))
filiais_lista = ["TODAS"] + sorted([
    str(b) for b in res_f_init["BASE"].dropna().unique() if str(b).strip() != ""
])
mots_opcao = sorted(list(res_f_init["MOTORISTA"].dropna().unique()))

with gr.Blocks(
    theme=gr.themes.Soft(), title="Dashboard do Prêmio de Motoristas"
) as app:

  state_resumo = gr.State(value=res_f_init)
  state_ausencias = gr.State(value=df_ausencias_global)
  state_desclassificacoes = gr.State(value=df_desclassificacoes_global)
  state_cat_custom = gr.State(value=mapa_cat_custom_global)

  gr.HTML("""
        <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 16px; padding: 20px 28px; display: flex; align-items: center; gap: 24px; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.03); margin-bottom: 20px;">
            <div style="display: flex; flex-direction: column; align-items: center; width: 95px; flex-shrink: 0;">
                <div style="width: 85px; height: 42px; position: relative; border-radius: 2px; overflow: hidden; display: flex; flex-direction: column; border: 1px solid #CBD5E1;">
                    <div style="height: 33.3%; background-color: #0099DA;"></div>
                    <div style="height: 33.3%; background-color: #FFD700;"></div>
                    <div style="height: 33.3%; background-color: #1E2B7A;"></div>
                    <svg viewBox="0 0 24 24" style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 24px; height: 24px;">
                        <path fill="#0099DA" stroke="#FFFFFF" stroke-width="1.5" d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"/>
                        <path fill="#FFFFFF" d="M10 8.5a2.5 2.5 0 0 0 2.5 2.5 0.7 0.7 0 0 0-1.4 0z"/>
                    </svg>
                </div>
                <span style="font-family: 'Arial Black', sans-serif; font-weight: 900; color: #1E2B7A; font-size: 13px; margin-top: 3px;">Ciapetro</span>
                <span style="font-family: sans-serif; font-size: 5px; color: #475569; text-align: center; text-transform: uppercase;">Distribuidora de Combustíveis</span>
            </div>

            <div style="display: flex; flex-direction: column; justify-content: center;">
                <h1 style="margin: 0; color: #0F172A; font-size: 26px; font-weight: 800;">
                    Dashboard do Prêmio de Motoristas
                </h1>
                <p style="margin: 4px 0 0 0; color: #64748B; font-size: 14px;">
                    Visão gerencial de consumo, desempenho e prêmio
                </p>
            </div>
        </div>
        """)

  with gr.Row():
    f_dt_ini = gr.Textbox(
        label="📅 Data Início",
        value=min_date_default,
        placeholder="DD/MM/AAAA ou AAAA-MM-DD",
    )
    f_dt_fim = gr.Textbox(
        label="📅 Data Fim",
        value=max_date_default,
        placeholder="DD/MM/AAAA ou AAAA-MM-DD",
    )
    f_mot = gr.Dropdown(
        choices=mots_lista,
        value="TODOS",
        label="👤 Filtrar Motorista",
        filterable=True,
    )

  with gr.Row():
    f_plc = gr.Textbox(
        label="🔍 Filtrar Placa", placeholder="Ex: ABC1234 ou ABC1D23"
    )
    f_cat = gr.Dropdown(
        choices=cats_lista, value="TODAS", label="🏷️ Categoria de Veículo"
    )
    f_fil = gr.Dropdown(
        choices=filiais_lista,
        value="TODAS",
        label="🏢 Filtrar Filial / Base",
        filterable=True,
    )

  with gr.Row():
    btn_aplicar = gr.Button("⚡ Aplicar Filtros", variant="primary")
    btn_limpar = gr.Button("🔄 Limpar Filtros", variant="secondary")

  gr.Markdown("---")

  with gr.Row():
    kpi_p = gr.Textbox(
        label="💰 Total em Prêmios", value=f_premio_init, interactive=False
    )
    kpi_gasto_comb = gr.Textbox(
        label="💳 Total Gasto Combustível",
        value=f_gasto_init,
        interactive=False,
    )
    kpi_k = gr.Textbox(
        label="RODADO TOTAL (KM)", value=f_km_init, interactive=False
    )
    kpi_l = gr.Textbox(
        label="⛽ COMBUSTÍVEL TOTAL", value=f_litros_init, interactive=False
    )
    kpi_avg = gr.Textbox(
        label="🎯 MÉDIA GERAL (KM/L)", value=f_media_init, interactive=False
    )
    kpi_m = gr.Textbox(
        label="👥 MOTORISTAS NA LISTA", value=f_mots_init, interactive=False
    )

  gr.Markdown("---")

  with gr.Tabs():
    with gr.Tab("📊 Resumo de Premiações por Motorista"):
      grid_resumo = gr.Dataframe(value=res_view_init, interactive=False)

    with gr.Tab("⚙️ Gestão de Cadastros"):
      gr.Markdown(
          "### 🛠️ **Cadastro de Motoristas e Frota (Placas / Filiais / Tipos)**"
      )
      gr.Markdown(
          "Adicione novos motoristas ou veículos diretamente no sistema. Os"
          " dados informados aqui são salvos e integrados automaticamente"
          " aos cálculos e filtros."
      )

      with gr.Row():
        with gr.Column():
          gr.Markdown("#### 👤 **Cadastrar Novo Motorista**")
          in_cad_mot_nome = gr.Textbox(
              label="Nome do Motorista", placeholder="Ex: JOAO SILVA"
          )
          in_cad_mot_tipo = gr.Dropdown(
              choices=cats_precos_opcoes + ["FOLGUISTA"],
              value=cats_precos_opcoes[0] if cats_precos_opcoes else "TRUCK",
              label="Tipo / Categoria Padrão",
          )
          in_cad_mot_base = gr.Textbox(
              label="Filial / Base", placeholder="Ex: CIANORTE ou UBERABA"
          )
          btn_cad_mot = gr.Button(
              "➕ Salvar Novo Motorista", variant="primary"
          )
          out_msg_mot = gr.Markdown("")

        with gr.Column():
          gr.Markdown("#### 🚛 **Cadastrar Nova Placa / Frota**")
          in_cad_placa = gr.Textbox(
              label="Placa / Cavalo", placeholder="Ex: XYZ9999"
          )
          in_cad_placa_tipo = gr.Dropdown(
              choices=cats_precos_opcoes,
              value=cats_precos_opcoes[0] if cats_precos_opcoes else "TRUCK",
              label="Tipo de Veículo",
          )
          btn_cad_frota = gr.Button("➕ Salvar Nova Placa", variant="primary")
          out_msg_frota = gr.Markdown("")

      gr.Markdown("---")
      gr.Markdown("### 📋 **Relação Atual de Cadastros no Sistema**")
      with gr.Row():
        grid_cad_motoristas = gr.Dataframe(
            value=cadastro[[
                "MOTORISTA_CADASTRO",
                "TIPO_CADASTRO",
                "BASE_CADASTRO",
                "STATUS",
                "DATA_INATIVACAO",
            ]].rename(columns={
                "MOTORISTA_CADASTRO": "MOTORISTA",
                "TIPO_CADASTRO": "TIPO",
                "BASE_CADASTRO": "BASE",
                "DATA_INATIVACAO": "DATA INATIVAÇÃO",
            }),
            interactive=False,
            label="Motoristas Cadastrados",
        )
        grid_cad_frota = gr.Dataframe(
            value=frota[[
                "PLACA_PADRONIZADA",
                "TIPO",
                "STATUS",
                "DATA_INATIVACAO",
            ]].rename(columns={
                "PLACA_PADRONIZADA": "PLACA",
                "TIPO": "TIPO VEICULO",
                "DATA_INATIVACAO": "DATA INATIVAÇÃO",
            }),
            interactive=False,
            label="Frota Cadastrada",
        )

      gr.Markdown("---")
      gr.Markdown("### ❌ **Inativar / Reativar Cadastros**")
      gr.Markdown(
          "Se um motorista for demitido ou um veículo vendido, utilize as opções"
          " abaixo. Os dados históricos permanecerão nos cálculos, mas eles"
          " constarão como INATIVOS no sistema, inclusive com sua data de saída."
      )

      with gr.Row():
        with gr.Column():
          opcoes_mots_cad = sorted(
              list(cadastro["MOTORISTA_CADASTRO"].unique())
          )
          inativar_mot_dropdown = gr.Dropdown(
              choices=opcoes_mots_cad,
              label="👤 Selecionar Motorista para Inativar/Reativar",
          )
          with gr.Row():
            btn_inativar_mot = gr.Button(
                "❌ Inativar Motorista", variant="stop"
            )
            btn_reativar_mot = gr.Button(
                "✅ Reativar Motorista", variant="secondary"
            )
          out_msg_inativar_mot = gr.Markdown("")

        with gr.Column():
          opcoes_placas_cad = sorted(
              list(frota["PLACA_PADRONIZADA"].unique())
          )
          inativar_placa_dropdown = gr.Dropdown(
              choices=opcoes_placas_cad,
              label="🚛 Selecionar Placa para Inativar/Reativar",
          )
          with gr.Row():
            btn_inativar_placa = gr.Button(
                "❌ Inativar Placa", variant="stop"
            )
            btn_reativar_placa = gr.Button(
                "✅ Reativar Placa", variant="secondary"
            )
          out_msg_inativar_placa = gr.Markdown("")

    with gr.Tab("🚛 Múltiplas Placas (Média Separada por Placa)"):
      gr.Markdown(
          "### 🚛 **Cálculo Separado por Placa (Motoristas que abasteceram em"
          " > 1 placa)**"
      )
      gr.Markdown(
          "Motoristas que abasteceram em múltiplos veículos têm suas médias e"
          " consumos discriminados individualmente por placa abaixo."
      )
      grid_multi_placas = gr.Dataframe(value=df_multi_init, interactive=False)

    with gr.Tab("🏷️ Definir Categoria de Pagamento (Multi-Placas)"):
      gr.Markdown("### 🏷️ **Marcação da Categoria para Pagamento do Prêmio**")
      gr.Markdown(
          "Para motoristas que abasteceram em mais de uma placa de categorias"
          " diferentes, escolha abaixo qual categoria será considerada para a"
          " régua de pagamento do prêmio."
      )

      # O seletor de placa é preenchido dinamicamente de acordo com o motorista.
      def placas_iniciais_do_motorista(nome_motorista):
        mot_norm = DataUtils.normalizar_texto(nome_motorista)
        if not mot_norm or mot_norm in ("NENHUM MOTORISTA", "TODOS"):
          return []
        placas = (
            eventos.loc[
                eventos["CONDUTOR_NORMALIZADO"].eq(mot_norm),
                "PLACA_PADRONIZADA",
            ]
            .dropna()
            .astype(str)
            .str.strip()
        )
        return sorted([p for p in placas.unique().tolist() if p])

      placas_multi_iniciais = placas_iniciais_do_motorista(
          mots_multi_drop_init.value
      )

      with gr.Row():
        sel_mot_multi = gr.Dropdown(
            choices=mots_multi_drop_init.choices,
            value=mots_multi_drop_init.value,
            label="👤 Selecionar Motorista com Múltiplas Placas",
            filterable=True,
        )
        sel_placa_multi = gr.Dropdown(
            choices=placas_multi_iniciais,
            value=placas_multi_iniciais[0] if placas_multi_iniciais else None,
            label="🚛 Selecionar Placa Específica",
            filterable=True,
        )
        sel_cat_marca = gr.Dropdown(
            choices=cats_precos_opcoes,
            value=cats_precos_opcoes[0] if cats_precos_opcoes else None,
            label="🏷️ Categoria a ser Considerada para Pagamento",
        )
        btn_salvar_cat_marca = gr.Button(
            "💾 Salvar Categoria Escolhida", variant="primary"
        )

      out_msg_cat = gr.Markdown("")

      gr.Markdown("#### 📋 **Mapeamentos Manuais de Categoria Ativos**")

      def gerar_df_custom(mapa):
        colunas = ["MOTORISTA", "PLACA", "CATEGORIA DEFINIDA"]
        if not mapa:
          return pd.DataFrame(columns=colunas)
        registros_custom = []
        for chave, categoria in mapa.items():
          chave = str(chave).strip().upper()
          if "|||" in chave:
            mot, placa = chave.split("|||", 1)
          else:
            mot, placa = chave, ""
          registros_custom.append([mot, placa, categoria])
        return pd.DataFrame(registros_custom, columns=colunas)

      grid_custom_cats = gr.Dataframe(
          value=gerar_df_custom(mapa_cat_custom_global), interactive=False
      )

    with gr.Tab("👔 Relatório RH - Lançamento de Pagamento"):
      gr.Markdown("### 👔 **Relatório RH - Pagamento de Prêmios**")
      gr.Markdown(
          "Relação simplificada de motoristas, filiais e valores devidos para"
          " lançamento na folha de pagamento pelo setor de RH. (*Motoristas"
          " Inativos não aparecem aqui*)"
      )
      grid_rh = gr.Dataframe(value=rh_view_init, interactive=False)

    with gr.Tab("📋 Detalhamento dos Abastecimentos"):
      grid_eventos = gr.Dataframe(value=det_view_init, interactive=False)

    with gr.Tab("📄 Recibo de Premiação"):
      gr.Markdown("### 🖨️ **Gerador de Recibo de Premiação para Assinatura**")

      mots_ativos_inic = (
          res_f_init[res_f_init["STATUS_MOTORISTA"] == "ATIVO"]["MOTORISTA"]
          .dropna()
          .unique()
          if "STATUS_MOTORISTA" in res_f_init.columns
          else res_f_init["MOTORISTA"].dropna().unique()
      )
      opcoes_recibo_inic = [
          "SELECIONE...",
          "TODOS OS MOTORISTAS (TODAS AS FILIAIS)",
      ] + sorted(list(mots_ativos_inic))

      with gr.Row():
        rec_fil = gr.Dropdown(
            choices=filiais_lista,
            value="TODAS",
            label="🏢 Filtrar Filial / Base",
            filterable=True,
        )
        rec_mot = gr.Dropdown(
            choices=opcoes_recibo_inic,
            value="TODOS OS MOTORISTAS (TODAS AS FILIAIS)",
            label="👤 Selecionar Motorista",
            filterable=True,
        )
        rec_ini = gr.Textbox(label="📅 Período Início", value=min_date_default)
        rec_fim = gr.Textbox(label="📅 Período Fim", value=max_date_default)

      with gr.Row():
        rec_fator = gr.Textbox(label="⚖️ Fator Carga", value="50%")
        btn_recibo = gr.Button(
            "📄 Gerar / Atualizar Recibo(s)", variant="primary"
        )

      recibo_output = gr.HTML(
          value=(
              "<div style='text-align: center; padding: 40px; color:"
              " #64748B;'>👉 Selecione um motorista ou uma filial acima para"
              " gerar os recibos.</div>"
          )
      )

    with gr.Tab("🏥 Lançamento de Atestados e Férias"):
      gr.Markdown(
          "### 🏥 **Lançamento de Ausências (Desconto de Atestados /"
          " Férias)**"
      )
      gr.Markdown(
          "Os dias lançados aqui serão calculados com base na data de início e fim, "
          "e automaticamente descontados da quantidade de **dias efetivos** e do "
          "**valor final do prêmio** no dashboard e nos recibos."
      )

      gr.Markdown("✅ **Cálculo automático ativo: Data de Início + Data Fim → Dias Ausente**")

      with gr.Row():
        aus_mot = gr.Dropdown(
            choices=mots_opcao, label="👤 Motorista", filterable=True
        )
        aus_tipo = gr.Radio(
            choices=["Atestado Médico", "Férias", "Outro Afastamento"],
            value="Atestado Médico",
            label="📌 Tipo de Ausência",
        )
        aus_data = gr.Textbox(
            label="📅 Data de Início",
            value="",
            placeholder="DD/MM/AAAA",
            interactive=True,
        )
        aus_data_fim = gr.Textbox(
            label="📅 Data Fim",
            value="",
            placeholder="DD/MM/AAAA",
            interactive=True,
        )
        aus_dias = gr.Number(
            label="🔢 Dias Ausente (calculado)",
            value=0,
            precision=0,
            interactive=False,
        )

      aus_obs = gr.Textbox(
          label="📝 Observação / Motivo",
          placeholder="Ex: CID 10, Licença médica, Férias regulamentares...",
      )

      with gr.Row():
        btn_add_ausencia = gr.Button("➕ Lançar Ausência", variant="primary")
        btn_limpar_ausencias = gr.Button(
            "🗑️ Limpar Todos os Lançamentos", variant="stop"
        )

      gr.Markdown("#### 📋 **Histórico de Ausências Lançadas**")
      grid_ausencias = gr.Dataframe(
          value=df_ausencias_global, interactive=False
      )

    with gr.Tab("🚫 Gestão de Desclassificações (Pilar 1)"):
      gr.Markdown(
          "### 🚫 **1º Pilar - Controles Administrativos e Operacionais**"
      )
      gr.Markdown(
          "Registrar infrações operacionais. Os critérios de **5 a 15**"
          " desclassificam o motorista diretamente. Já os critérios de **1 a"
          " 4** acumulam pontos (o acúmulo de **mais de 129 pontos** nestes"
          " critérios também gera a desclassificação automática com prêmio"
          " zerado em R$ 0,00)."
      )

      with gr.Row():
        descl_mot = gr.Dropdown(
            choices=mots_opcao, label="👤 Motorista", filterable=True
        )
        descl_crit = gr.Dropdown(
            choices=CRITERIOS_PILAR_1,
            value=CRITERIOS_PILAR_1[0],
            label="📌 Critério / Infração",
        )
        descl_pontos = gr.Number(
            label="🔢 Quantidade de Pontos / Eventos", value=1, precision=0
        )

      descl_obs = gr.Textbox(
          label="📝 Observação / Detalhes da Infração",
          placeholder=(
              "Ex: Ocorrência de freada brusca gravada pela telemetria, não uso"
              " do cinto..."
          ),
      )

      with gr.Row():
        btn_add_desclassificacao = gr.Button(
            "🚫 Registrar Infração / Desclassificação", variant="primary"
        )

      gr.Markdown("---")
      gr.Markdown(
          "#### 📋 **Histórico de Desclassificações e Infrações Registradas**"
      )
      grid_desclassificacoes = gr.Dataframe(
          value=df_desclassificacoes_global, interactive=False
      )

      opcoes_excl_inic = gerar_opcoes_exclusao_descl(
          df_desclassificacoes_global
      )

      with gr.Row():
        descl_excluir_sel = gr.Dropdown(
            choices=opcoes_excl_inic.choices,
            value=opcoes_excl_inic.value,
            label="🗑️ Selecionar Registro para Excluir",
            filterable=True,
            interactive=opcoes_excl_inic.interactive,
        )
        btn_excluir_item_descl = gr.Button(
            "🗑️ Excluir Registro Selecionado", variant="stop"
        )
        btn_limpar_desclassificacoes = gr.Button(
            "⚠️ Limpar TODOS os Lançamentos", variant="secondary"
        )

  # ================================================================
  # LÓGICA E EVENTOS DO GRADIO
  # ================================================================

  def gerenciar_status_motorista(nome, inativar=True):
    global cadastro
    if not nome:
      grid_cad_m = cadastro[[
          "MOTORISTA_CADASTRO",
          "TIPO_CADASTRO",
          "BASE_CADASTRO",
          "STATUS",
          "DATA_INATIVACAO",
      ]].rename(columns={
          "MOTORISTA_CADASTRO": "MOTORISTA",
          "TIPO_CADASTRO": "TIPO",
          "BASE_CADASTRO": "BASE",
          "DATA_INATIVACAO": "DATA INATIVAÇÃO",
      })
      return "⚠️ Selecione um motorista.", grid_cad_m

    alternar_inativo("MOTORISTA", nome, inativar)
    cadastro = loader.carregar_cadastro_motoristas()
    grid_cad_m = cadastro[[
        "MOTORISTA_CADASTRO",
        "TIPO_CADASTRO",
        "BASE_CADASTRO",
        "STATUS",
        "DATA_INATIVACAO",
    ]].rename(columns={
        "MOTORISTA_CADASTRO": "MOTORISTA",
        "TIPO_CADASTRO": "TIPO",
        "BASE_CADASTRO": "BASE",
        "DATA_INATIVACAO": "DATA INATIVAÇÃO",
    })
    acao = "inativado" if inativar else "reativado"
    return f"✅ Motorista **{nome}** {acao} com sucesso!", grid_cad_m

  def gerenciar_status_placa(placa, inativar=True):
    global frota, mapa_frota
    if not placa:
      grid_cad_f = frota[[
          "PLACA_PADRONIZADA",
          "TIPO",
          "STATUS",
          "DATA_INATIVACAO",
      ]].rename(columns={
          "PLACA_PADRONIZADA": "PLACA",
          "TIPO": "TIPO VEICULO",
          "DATA_INATIVACAO": "DATA INATIVAÇÃO",
      })
      return "⚠️ Selecione uma placa.", grid_cad_f

    alternar_inativo("PLACA", placa, inativar)
    frota, mapa_frota = loader.carregar_frota()
    grid_cad_f = frota[[
        "PLACA_PADRONIZADA",
        "TIPO",
        "STATUS",
        "DATA_INATIVACAO",
    ]].rename(columns={
        "PLACA_PADRONIZADA": "PLACA",
        "TIPO": "TIPO VEICULO",
        "DATA_INATIVACAO": "DATA INATIVAÇÃO",
    })
    acao = "inativada" if inativar else "reativada"
    return f"✅ Placa **{placa}** {acao} com sucesso!", grid_cad_f

  def adicionar_motorista_cad(nome, tipo, base):
    global cadastro
    if not nome or not nome.strip():
      grid_cad_m = cadastro[[
          "MOTORISTA_CADASTRO",
          "TIPO_CADASTRO",
          "BASE_CADASTRO",
          "STATUS",
          "DATA_INATIVACAO",
      ]].rename(columns={
          "MOTORISTA_CADASTRO": "MOTORISTA",
          "TIPO_CADASTRO": "TIPO",
          "BASE_CADASTRO": "BASE",
          "DATA_INATIVACAO": "DATA INATIVAÇÃO",
      })
      return "⚠️ Digite o nome do motorista.", grid_cad_m

    nome_norm = DataUtils.normalizar_texto(nome)
    tipo_norm = DataUtils.normalizar_texto(tipo)
    base_norm = DataUtils.normalizar_texto(base)

    df_custom = carregar_motoristas_customizados()
    novo_reg = pd.DataFrame(
        [{"MOTORISTAS": nome_norm, "TIPO": tipo_norm, "BASE": base_norm}]
    )
    df_custom = pd.concat([df_custom, novo_reg], ignore_index=True)
    salvar_motoristas_customizados(df_custom)

    cadastro = loader.carregar_cadastro_motoristas()
    grid_cad_m = cadastro[[
        "MOTORISTA_CADASTRO",
        "TIPO_CADASTRO",
        "BASE_CADASTRO",
        "STATUS",
        "DATA_INATIVACAO",
    ]].rename(columns={
        "MOTORISTA_CADASTRO": "MOTORISTA",
        "TIPO_CADASTRO": "TIPO",
        "BASE_CADASTRO": "BASE",
        "DATA_INATIVACAO": "DATA INATIVAÇÃO",
    })
    return (
        f"✅ Motorista **{nome_norm}** cadastrado com sucesso!",
        grid_cad_m,
    )

  def adicionar_placa_frota(placa, tipo):
    global frota, mapa_frota
    if not placa or not placa.strip():
      grid_cad_f = frota[[
          "PLACA_PADRONIZADA",
          "TIPO",
          "STATUS",
          "DATA_INATIVACAO",
      ]].rename(columns={
          "PLACA_PADRONIZADA": "PLACA",
          "TIPO": "TIPO VEICULO",
          "DATA_INATIVACAO": "DATA INATIVAÇÃO",
      })
      return "⚠️ Digite a placa do veículo.", grid_cad_f

    placa_norm = DataUtils.padronizar_placa(placa)
    tipo_norm = DataUtils.normalizar_texto(tipo)

    df_custom = carregar_frota_customizada()
    novo_reg = pd.DataFrame([{"CAVALO": placa_norm, "TIPO": tipo_norm}])
    df_custom = pd.concat([df_custom, novo_reg], ignore_index=True)
    salvar_frota_customizada(df_custom)

    frota, mapa_frota = loader.carregar_frota()
    grid_cad_f = frota[[
        "PLACA_PADRONIZADA",
        "TIPO",
        "STATUS",
        "DATA_INATIVACAO",
    ]].rename(columns={
        "PLACA_PADRONIZADA": "PLACA",
        "TIPO": "TIPO VEICULO",
        "DATA_INATIVACAO": "DATA INATIVAÇÃO",
    })
    return f"✅ Placa **{placa_norm}** cadastrada com sucesso!", grid_cad_f

  def salvar_categoria_customizada_handler(
      mot_chave,
      placa_escolhida,
      cat_escolhida,
      map_cat_st,
      dt_ini,
      dt_fim,
      mot,
      plc,
      cat,
      fil,
      aus_df,
      descl_df,
  ):
    mot_norm = DataUtils.normalizar_texto(mot_chave)
    placa_norm = DataUtils.padronizar_placa(placa_escolhida)

    if not mot_norm or mot_norm in ["NENHUM MOTORISTA", ""]:
      msg = "⚠️ Selecione um motorista válido."
      res_empty = aplicar_filtros(
          dt_ini, dt_fim, mot, plc, cat, fil, aus_df, descl_df, map_cat_st
      )
      return (msg, map_cat_st, gerar_df_custom(map_cat_st)) + res_empty

    if not placa_norm:
      msg = "⚠️ Selecione uma placa válida para esse motorista."
      res_empty = aplicar_filtros(
          dt_ini, dt_fim, mot, plc, cat, fil, aus_df, descl_df, map_cat_st
      )
      return (msg, map_cat_st, gerar_df_custom(map_cat_st)) + res_empty

    if not cat_escolhida:
      msg = "⚠️ Selecione a categoria que será considerada para pagamento."
      res_empty = aplicar_filtros(
          dt_ini, dt_fim, mot, plc, cat, fil, aus_df, descl_df, map_cat_st
      )
      return (msg, map_cat_st, gerar_df_custom(map_cat_st)) + res_empty

    map_cat_st = dict(map_cat_st or {})
    chave = normalizar_chave_categoria_customizada(mot_norm, placa_norm)
    map_cat_st[chave] = DataUtils.normalizar_texto(cat_escolhida)
    salvar_categorias_customizadas(map_cat_st)

    res = aplicar_filtros(
        dt_ini, dt_fim, mot, plc, cat, fil, aus_df, descl_df, map_cat_st
    )
    msg = (
        f"✅ Categoria **{cat_escolhida}** salva para o motorista "
        f"**{mot_norm}** na placa **{placa_norm}**."
    )
    grid_c = gerar_df_custom(map_cat_st)
    return (msg, map_cat_st, grid_c) + res

  def adicionar_ausencia_handler(
      mot,
      tipo_aus,
      dt_ini_aus,
      dt_fim_aus,
      dias_aus,
      obs_aus,
      aus_df,
      dt_ini,
      dt_fim,
      mot_f,
      plc_f,
      cat_f,
      fil_f,
      descl_df,
      map_cat_st,
  ):
    if not mot:
      res_empty = aplicar_filtros(
          dt_ini,
          dt_fim,
          mot_f,
          plc_f,
          cat_f,
          fil_f,
          aus_df,
          descl_df,
          map_cat_st,
      )
      return (aus_df,) + res_empty

    dias_calculados = calcular_dias_ausencia(dt_ini_aus, dt_fim_aus)
    if dias_calculados <= 0:
      res_empty = aplicar_filtros(
          dt_ini, dt_fim, mot_f, plc_f, cat_f, fil_f, aus_df, descl_df, map_cat_st
      )
      return (aus_df,) + res_empty

    novo_reg = pd.DataFrame([{
        "MOTORISTA": mot,
        "TIPO_AUSENCIA": tipo_aus,
        "DATA_INICIO": dt_ini_aus,
        "DATA_FIM": dt_fim_aus,
        "DIAS": dias_calculados,
        "OBSERVACAO": obs_aus,
    }])
    novo_aus_df = pd.concat([aus_df, novo_reg], ignore_index=True)
    salvar_ausencias(novo_aus_df)

    res = aplicar_filtros(
        dt_ini,
        dt_fim,
        mot_f,
        plc_f,
        cat_f,
        fil_f,
        novo_aus_df,
        descl_df,
        map_cat_st,
    )
    return (novo_aus_df,) + res

  def limpar_ausencias_handler(
      dt_ini, dt_fim, mot_f, plc_f, cat_f, fil_f, descl_df, map_cat_st
  ):
    df_vazio = pd.DataFrame(
        columns=[
            "MOTORISTA",
            "TIPO_AUSENCIA",
            "DATA_INICIO",
            "DATA_FIM",
            "DIAS",
            "OBSERVACAO",
        ]
    )
    salvar_ausencias(df_vazio)
    res = aplicar_filtros(
        dt_ini,
        dt_fim,
        mot_f,
        plc_f,
        cat_f,
        fil_f,
        df_vazio,
        descl_df,
        map_cat_st,
    )
    return (df_vazio,) + res

  def adicionar_desclassificacao_handler(
      mot,
      crit,
      pontos,
      obs,
      descl_df,
      dt_ini,
      dt_fim,
      mot_f,
      plc_f,
      cat_f,
      fil_f,
      aus_df,
      map_cat_st,
  ):
    if not mot:
      res_empty = aplicar_filtros(
          dt_ini,
          dt_fim,
          mot_f,
          plc_f,
          cat_f,
          fil_f,
          aus_df,
          descl_df,
          map_cat_st,
      )
      drop_e = gerar_opcoes_exclusao_descl(descl_df)
      return (descl_df, drop_e) + res_empty

    num_crit = int(crit.split("-")[0].strip()) if "-" in crit else 1
    tipo_impacto = "DESCLASSIFICADO" if num_crit >= 5 else "PONTOS"

    novo_reg = pd.DataFrame([{
        "MOTORISTA": mot,
        "CRITERIO": crit,
        "PONTOS": pontos,
        "TIPO_IMPACTO": tipo_impacto,
        "OBSERVACAO": obs,
    }])
    novo_descl_df = pd.concat([descl_df, novo_reg], ignore_index=True)
    salvar_desclassificacoes(novo_descl_df)

    res = aplicar_filtros(
        dt_ini,
        dt_fim,
        mot_f,
        plc_f,
        cat_f,
        fil_f,
        aus_df,
        novo_descl_df,
        map_cat_st,
    )
    drop_e = gerar_opcoes_exclusao_descl(novo_descl_df)
    return (novo_descl_df, drop_e) + res

  def excluir_desclassificacao_item_handler(
      item_sel,
      descl_df,
      dt_ini,
      dt_fim,
      mot_f,
      plc_f,
      cat_f,
      fil_f,
      aus_df,
      map_cat_st,
  ):
    if not item_sel or "Nenhum" in item_sel:
      res_empty = aplicar_filtros(
          dt_ini,
          dt_fim,
          mot_f,
          plc_f,
          cat_f,
          fil_f,
          aus_df,
          descl_df,
          map_cat_st,
      )
      drop_e = gerar_opcoes_exclusao_descl(descl_df)
      return (descl_df, drop_e) + res_empty

    try:
      idx = int(item_sel.split("]")[0].replace("[", ""))
      novo_descl_df = descl_df.drop(index=idx).reset_index(drop=True)
      salvar_desclassificacoes(novo_descl_df)
    except Exception:
      novo_descl_df = descl_df

    res = aplicar_filtros(
        dt_ini,
        dt_fim,
        mot_f,
        plc_f,
        cat_f,
        fil_f,
        aus_df,
        novo_descl_df,
        map_cat_st,
    )
    drop_e = gerar_opcoes_exclusao_descl(novo_descl_df)
    return (novo_descl_df, drop_e) + res

  def limpar_desclassificacoes_handler(
      dt_ini, dt_fim, mot_f, plc_f, cat_f, fil_f, aus_df, map_cat_st
  ):
    df_vazio = pd.DataFrame(
        columns=[
            "MOTORISTA",
            "CRITERIO",
            "PONTOS",
            "TIPO_IMPACTO",
            "OBSERVACAO",
        ]
    )
    salvar_desclassificacoes(df_vazio)
    res = aplicar_filtros(
        dt_ini,
        dt_fim,
        mot_f,
        plc_f,
        cat_f,
        fil_f,
        aus_df,
        df_vazio,
        map_cat_st,
    )
    drop_e = gerar_opcoes_exclusao_descl(df_vazio)
    return (df_vazio, drop_e) + res

  # Mapeamento de botões e ações
  btn_aplicar.click(
      fn=aplicar_filtros,
      inputs=[
          f_dt_ini,
          f_dt_fim,
          f_mot,
          f_plc,
          f_cat,
          f_fil,
          state_ausencias,
          state_desclassificacoes,
          state_cat_custom,
      ],
      outputs=[
          kpi_p,
          kpi_gasto_comb,
          kpi_k,
          kpi_l,
          kpi_avg,
          kpi_m,
          grid_resumo,
          grid_rh,
          grid_eventos,
          grid_multi_placas,
          sel_mot_multi,
          state_resumo,
      ],
  )

  btn_limpar.click(
      fn=lambda aus_df, descl_df, map_cat: (
          min_date_default,
          max_date_default,
          "TODOS",
          "",
          "TODAS",
          "TODAS",
      )
      + aplicar_filtros(
          min_date_default,
          max_date_default,
          "TODOS",
          "",
          "TODAS",
          "TODAS",
          aus_df,
          descl_df,
          map_cat,
      ),
      inputs=[state_ausencias, state_desclassificacoes, state_cat_custom],
      outputs=[
          f_dt_ini,
          f_dt_fim,
          f_mot,
          f_plc,
          f_cat,
          f_fil,
          kpi_p,
          kpi_gasto_comb,
          kpi_k,
          kpi_l,
          kpi_avg,
          kpi_m,
          grid_resumo,
          grid_rh,
          grid_eventos,
          grid_multi_placas,
          sel_mot_multi,
          state_resumo,
      ],
  )

  btn_cad_mot.click(
      fn=adicionar_motorista_cad,
      inputs=[in_cad_mot_nome, in_cad_mot_tipo, in_cad_mot_base],
      outputs=[out_msg_mot, grid_cad_motoristas],
  )

  btn_cad_frota.click(
      fn=adicionar_placa_frota,
      inputs=[in_cad_placa, in_cad_placa_tipo],
      outputs=[out_msg_frota, grid_cad_frota],
  )

  btn_inativar_mot.click(
      fn=lambda m: gerenciar_status_motorista(m, True),
      inputs=[inativar_mot_dropdown],
      outputs=[out_msg_inativar_mot, grid_cad_motoristas],
  )

  btn_reativar_mot.click(
      fn=lambda m: gerenciar_status_motorista(m, False),
      inputs=[inativar_mot_dropdown],
      outputs=[out_msg_inativar_mot, grid_cad_motoristas],
  )

  btn_inativar_placa.click(
      fn=lambda p: gerenciar_status_placa(p, True),
      inputs=[inativar_placa_dropdown],
      outputs=[out_msg_inativar_placa, grid_cad_frota],
  )

  btn_reativar_placa.click(
      fn=lambda p: gerenciar_status_placa(p, False),
      inputs=[inativar_placa_dropdown],
      outputs=[out_msg_inativar_placa, grid_cad_frota],
  )

  # Ao trocar o motorista, atualiza automaticamente a lista de placas disponíveis.
  def atualizar_placas_do_motorista(mot_chave):
    placas = placas_iniciais_do_motorista(mot_chave)
    return gr.Dropdown(
        choices=placas,
        value=placas[0] if placas else None,
        interactive=bool(placas),
    )

  sel_mot_multi.change(
      fn=atualizar_placas_do_motorista,
      inputs=[sel_mot_multi],
      outputs=[sel_placa_multi],
  )

  btn_salvar_cat_marca.click(
      fn=salvar_categoria_customizada_handler,
      inputs=[
          sel_mot_multi,
          sel_placa_multi,
          sel_cat_marca,
          state_cat_custom,
          f_dt_ini,
          f_dt_fim,
          f_mot,
          f_plc,
          f_cat,
          f_fil,
          state_ausencias,
          state_desclassificacoes,
      ],
      outputs=[
          out_msg_cat,
          state_cat_custom,
          grid_custom_cats,
          kpi_p,
          kpi_gasto_comb,
          kpi_k,
          kpi_l,
          kpi_avg,
          kpi_m,
          grid_resumo,
          grid_rh,
          grid_eventos,
          grid_multi_placas,
          sel_mot_multi,
          state_resumo,
      ],
  )

  btn_recibo.click(
      fn=gerar_recibos_lote,
      inputs=[rec_fil, rec_mot, rec_ini, rec_fim, rec_fator, state_resumo],
      outputs=[recibo_output],
  )

  aus_data.input(
      fn=calcular_dias_ausencia,
      inputs=[aus_data, aus_data_fim],
      outputs=[aus_dias],
  )
  aus_data_fim.input(
      fn=calcular_dias_ausencia,
      inputs=[aus_data, aus_data_fim],
      outputs=[aus_dias],
  )
  aus_data.change(
      fn=calcular_dias_ausencia,
      inputs=[aus_data, aus_data_fim],
      outputs=[aus_dias],
  )
  aus_data_fim.change(
      fn=calcular_dias_ausencia,
      inputs=[aus_data, aus_data_fim],
      outputs=[aus_dias],
  )

  btn_add_ausencia.click(
      fn=adicionar_ausencia_handler,
      inputs=[
          aus_mot,
          aus_tipo,
          aus_data,
          aus_data_fim,
          aus_dias,
          aus_obs,
          state_ausencias,
          f_dt_ini,
          f_dt_fim,
          f_mot,
          f_plc,
          f_cat,
          f_fil,
          state_desclassificacoes,
          state_cat_custom,
      ],
      outputs=[
          state_ausencias,
          kpi_p,
          kpi_gasto_comb,
          kpi_k,
          kpi_l,
          kpi_avg,
          kpi_m,
          grid_resumo,
          grid_rh,
          grid_eventos,
          grid_multi_placas,
          sel_mot_multi,
          state_resumo,
      ],
  ).then(
      fn=lambda aus_df: aus_df, inputs=[state_ausencias], outputs=[grid_ausencias]
  )

  btn_limpar_ausencias.click(
      fn=limpar_ausencias_handler,
      inputs=[
          f_dt_ini,
          f_dt_fim,
          f_mot,
          f_plc,
          f_cat,
          f_fil,
          state_desclassificacoes,
          state_cat_custom,
      ],
      outputs=[
          state_ausencias,
          kpi_p,
          kpi_gasto_comb,
          kpi_k,
          kpi_l,
          kpi_avg,
          kpi_m,
          grid_resumo,
          grid_rh,
          grid_eventos,
          grid_multi_placas,
          sel_mot_multi,
          state_resumo,
      ],
  ).then(
      fn=lambda aus_df: aus_df, inputs=[state_ausencias], outputs=[grid_ausencias]
  )

  btn_add_desclassificacao.click(
      fn=adicionar_desclassificacao_handler,
      inputs=[
          descl_mot,
          descl_crit,
          descl_pontos,
          descl_obs,
          state_desclassificacoes,
          f_dt_ini,
          f_dt_fim,
          f_mot,
          f_plc,
          f_cat,
          f_fil,
          state_ausencias,
          state_cat_custom,
      ],
      outputs=[
          state_desclassificacoes,
          descl_excluir_sel,
          kpi_p,
          kpi_gasto_comb,
          kpi_k,
          kpi_l,
          kpi_avg,
          kpi_m,
          grid_resumo,
          grid_rh,
          grid_eventos,
          grid_multi_placas,
          sel_mot_multi,
          state_resumo,
      ],
  ).then(
      fn=lambda descl_df: descl_df,
      inputs=[state_desclassificacoes],
      outputs=[grid_desclassificacoes],
  )

  btn_excluir_item_descl.click(
      fn=excluir_desclassificacao_item_handler,
      inputs=[
          descl_excluir_sel,
          state_desclassificacoes,
          f_dt_ini,
          f_dt_fim,
          f_mot,
          f_plc,
          f_cat,
          f_fil,
          state_ausencias,
          state_cat_custom,
      ],
      outputs=[
          state_desclassificacoes,
          descl_excluir_sel,
          kpi_p,
          kpi_gasto_comb,
          kpi_k,
          kpi_l,
          kpi_avg,
          kpi_m,
          grid_resumo,
          grid_rh,
          grid_eventos,
          grid_multi_placas,
          sel_mot_multi,
          state_resumo,
      ],
  ).then(
      fn=lambda descl_df: descl_df,
      inputs=[state_desclassificacoes],
      outputs=[grid_desclassificacoes],
  )

  btn_limpar_desclassificacoes.click(
      fn=limpar_desclassificacoes_handler,
      inputs=[
          f_dt_ini,
          f_dt_fim,
          f_mot,
          f_plc,
          f_cat,
          f_fil,
          state_ausencias,
          state_cat_custom,
      ],
      outputs=[
          state_desclassificacoes,
          descl_excluir_sel,
          kpi_p,
          kpi_gasto_comb,
          kpi_k,
          kpi_l,
          kpi_avg,
          kpi_m,
          grid_resumo,
          grid_rh,
          grid_eventos,
          grid_multi_placas,
          sel_mot_multi,
          state_resumo,
      ],
  ).then(
      fn=lambda descl_df: descl_df,
      inputs=[state_desclassificacoes],
      outputs=[grid_desclassificacoes],
  )

if __name__ == "__main__":
  port = int(os.getenv("PORT", 7860))
  app.launch(server_name="0.0.0.0", server_port=port)
