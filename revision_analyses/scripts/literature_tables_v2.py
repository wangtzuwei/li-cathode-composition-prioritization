"""Version 2 of the literature CSVs (analysis 08b; see revision_analyses/README.md).

Writes two new files in literature/; the v1 files and the original GPT files are not touched:
  Cr_F_LiRich_primary_verified_samples_v2_20261024.csv = v1 (17 rows, unchanged) + Kim 2013 (7) + Pang 2017 (7)
  Task_B_verified_synthesis_evidence_v2_20261024.csv  = v1 (8 rows, unchanged)  + Giovine 2024 (4)

The new rows were read from the PDFs in literature/pdfs/ with pdftotext (2026-10-23).  Every value
is stated directly in the text, a table or the SI; nothing was read off a figure (this laptop has no PDF page
renderer, so figure-only data are written as 未報告 with the figure named).  Pages are journal pages with the
PDF page in brackets.
"""
from __future__ import annotations

import pandas as pd

import followup_checks as f6

LIT = f6.LITERATURE.parent
SAMPLES_V1 = LIT / "Cr_F_LiRich_primary_verified_samples_v1_20261023.csv"
SAMPLES_V2 = LIT / "Cr_F_LiRich_primary_verified_samples_v2_20261024.csv"
TASKB_V1 = LIT / "Task_B_verified_synthesis_evidence_v1_20261023.csv"
TASKB_V2 = LIT / "Task_B_verified_synthesis_evidence_v2_20261024.csv"
NR = "未報告"

# ---------------------------------------------------------------------------------------------------------
# Kim 2013, J Electroceram 30:159-165 (pdfs/10.1007_s10832-012-9778-4.pdf, 7 PDF pages, journal p159 = PDF p1)
# ---------------------------------------------------------------------------------------------------------
KIM = {
    "DOI": "10.1007/s10832-012-9778-4", "年份": "2013", "作者_簡稱": "Kim et al.",
    "論文標題": "Effects of transition metal doping and surface treatment to improve the electrochemical "
                "performance of Li2MnO3",
    "材料類別_與Li2MnO3直接性": "純 Li2MnO3（C2/m，Table 1 晶格參數）；機械化學法加入 1 mol% 過渡金屬；Cr 另做 2、5 mol%",
    "合成方法": "LiOH·H2O、MnCO3·nH2O、M(NO3)b·xH2O（M = Al、Cr、Fe、Zn）；行星式球磨濕磨（丙酮；FRITSCH "
                "Pulverisette 5，150 rpm，1 h；球粉重量比 20:1）；80 °C 烘箱乾燥",
    "鍛燒溫度時間氣氛": "400 °C 32 h 空氣（box furnace）",
    "充放電電壓範圍_V_vsLi": "2.0–4.8",
    "首圈電流密度或C_rate": "10 mA/g（CC/CV 充電至 4.8 V，CC 放電至 2.0 V）",
    "後續電流密度或C_rate": "Fig. 8：前 10 圈 10 mA/g，之後 10 圈 20 mA/g（只有圖）；EIS 前另組電池以 100 mA/g 跑 3 圈",
    "論文定義1C": NR, "測試溫度": "room temperature（未提供數值）",
    "電解液組成": "1 M LiPF6；EC:DMC:DEC = 1:1:1（體積比）",
    "活化後或第N圈放電容量_mAh_g": NR, "活化後或第N圈的循環次數": NR, "該N圈測試速率": NR,
    "數據來源類型": "Table 2 與正文直接列值；沒有圖讀值",
    "圖讀取估計誤差_mAh_g": "不適用（未從圖讀取）",
    "DOI確認連結": "https://doi.org/10.1007/s10832-012-9778-4",
    "全文取得形式": "出版社 PDF（7 PDF 頁；literature/pdfs/10.1007_s10832-012-9778-4.pdf），pdftotext 讀全文",
}
KIM_SOURCE_T2 = ("Experimental p160–161（PDF p2–3）；Table 2 p163（PDF p5）；正文 p163（PDF p5）列出各樣品首圈放電容量"
                 "和 Cr 樣品的首圈充電容量")
KIM_NOTE = ("Table 2 另有首圈效率和 SEM 平均粒徑兩欄，pdftotext 擷取時這兩欄的列序錯位，未轉錄；充放電容量兩欄的列序"
            "完整，且和正文 p163 的數值相同。Fig. 6–8 未讀值。")


def kim(sample: str, nominal: str, undoped: bool, element: str, x: str, site: str, charge: str, discharge: str,
        other: str, source: str = KIM_SOURCE_T2, note: str = KIM_NOTE) -> dict[str, str]:
    return {**KIM, "論文中名義化學式_保留原文參數寫法": nominal, "樣品名": sample, "是否未摻雜對照": "是" if undoped else "否",
            "摻雜元素": element, "摻雜量_原文x": x, "摻雜位點與組成變動": site,
            "首圈充電容量_mAh_g": charge, "首圈放電容量_mAh_g": discharge, "其他已報告容量與條件": other,
            "容量數字的精確出處_Table_Figure_頁碼": source, "備註": note}


TM_SITE = ("前驅物莫耳比 Li : Mn : {m} = 2 : 0.99 : 0.01（p160）；作者寫成 Li[LixMnyMz]O2，並指出過渡金屬同時有摻雜和"
           "表面修飾效果（p162）")
KIM_ROWS = [
    kim("Li2MnO3（pristine）", "Li2MnO3", True, "無", "0", "不適用", "308.2", "205.0", NR),
    kim("Al-Li2MnO3", "Li[LixMnyMz]O2，M = Al；Li : Mn : Al = 2 : 0.99 : 0.01", False, "Al", "0.01",
        TM_SITE.format(m="Al"), "333.7", "202.4", NR),
    kim("Cr-Li2MnO3（1 mol%）", "Li[LixMnyMz]O2，M = Cr；Li : Mn : Cr = 2 : 0.99 : 0.01", False, "Cr", "0.01",
        TM_SITE.format(m="Cr") + "；HRTEM、EDS、XPS（Cr 2p 約 587.9／578.8 eV）顯示表面改變，作者推論部分 Cr 以氧化物"
        "（可能是 Cr2O3）存在於表面（p162–163）", "418.7", "234.9", NR),
    kim("Fe-Li2MnO3", "Li[LixMnyMz]O2，M = Fe；Li : Mn : Fe = 2 : 0.99 : 0.01", False, "Fe", "0.01",
        TM_SITE.format(m="Fe"), "379.9", "211.5", NR),
    kim("Zn-Li2MnO3", "Li[LixMnyMz]O2，M = Zn；Li : Mn : Zn = 2 : 0.99 : 0.01", False, "Zn", "0.01",
        TM_SITE.format(m="Zn"), "354.2", "204.5", NR),
    *[kim(f"Cr-Li2MnO3（{pct} mol%）", f"Li2MnxCr1-xO3（Cr {pct} mol%）", False, "Cr", x,
          "原文只寫「Cr 2 mol% and 5 mol% Li2MnxCr1-xO3 samples were prepared」（p160）", NR, NR,
          f"原文 p164（PDF p6）：「the charge and discharge capacities of Li2MnO3 decreased to {value} mAhg-1 for "
          f"{pct} mol% Cr」。只有一個數值，沒有說明是充電或放電容量；對應 Fig. 7（未讀值）",
          "Experimental p160（PDF p2）；正文 p164（PDF p6）；Fig. 7",
          "首圈充電與首圈放電欄寫未報告，原因見「其他已報告容量與條件」。Fig. 7 未讀值。")
      for pct, x, value in [("2", "0.02", "223.2"), ("5", "0.05", "191.1")]],
]

# ---------------------------------------------------------------------------------------------------------
# Pang 2017, Chem. Mater. 29, 10299-10311 (pdfs/10.1021_acs.chemmater.7b02930.pdf + _SI.pdf; p10299 = PDF p1)
# ---------------------------------------------------------------------------------------------------------
PANG = {
    "DOI": "10.1021/acs.chemmater.7b02930", "年份": "2017", "作者_簡稱": "Pang et al.",
    "論文標題": "Effects of Fluorine and Chromium Doping on the Performance of Lithium-Rich Li1+xMO2 "
                "(M = Ni, Mn, Co) Positive Electrodes",
    "材料類別_與Li2MnO3直接性": "Li-rich layered；NPD 為 R-3m LiMO2 + C2/m Li2MnO3 雙相（重量比 4:1）；非純 Li2MnO3",
    "合成方法": "碳酸鹽共沉澱（MnSO4·H2O、NiSO4·7H2O、CoSO4·7H2O；Na2CO3 沉澱劑、NH4OH 螯合劑；60 °C，pH 7.5–8.5，"
                "800 rpm）；120 °C 乾燥；與 LiOH 混合",
    "論文定義1C": NR,
    "測試溫度": "未報告（半電池；in operando 18650 電池的外部溫度維持 22.5 °C）",
    "電解液組成": "1 M LiPF6；EC:DMC = 1:1（體積比）",
    "圖讀取估計誤差_mAh_g": "不適用（未從圖讀取）",
    "DOI確認連結": "https://doi.org/10.1021/acs.chemmater.7b02930",
    "全文取得形式": "出版社 PDF 與 SI（正文 13 PDF 頁、SI 8 PDF 頁；literature/pdfs/），pdftotext 讀全文",
}
PANG_MAIN = {
    "充放電電壓範圍_V_vsLi": "2.0–4.8（前 2 圈）；之後 2.0–4.6",
    "首圈電流密度或C_rate": "0.1C（充）／0.1D（放），前 2 圈",
    "後續電流密度或C_rate": "倍率測試放電 0.1／0.2／0.5／1.0 D（充電速率未報告）；全電池（vs Li4Ti5O12，0.5–3.3 V）前 2 圈 "
                           "0.1C／0.1D 化成，之後 0.2C／0.5D 至 200 圈",
    "數據來源類型": "正文與 SI Table S3 直接列值；沒有圖讀值",
    "容量數字的精確出處_Table_Figure_頁碼": "Experimental p10300–10301（PDF p2–3）；首圈與第 2 圈 p10304（PDF p6）；倍率 "
                                         "p10305（PDF p7）；全電池 p10305（PDF p7）與 SI Table S3（SI PDF p8）",
}
PANG_PRELIM = {
    "充放電電壓範圍_V_vsLi": NR, "首圈電流密度或C_rate": "1C（前期測試，p10300）", "後續電流密度或C_rate": NR,
    "首圈充電容量_mAh_g": NR, "首圈放電容量_mAh_g": NR,
    "活化後或第N圈放電容量_mAh_g": NR, "活化後或第N圈的循環次數": NR, "該N圈測試速率": NR,
    "數據來源類型": "SI Figure S1 圖說直接列值；沒有圖讀值",
    "容量數字的精確出處_Table_Figure_頁碼": "Experimental p10300（PDF p2）；SI Figure S1 圖說（SI PDF p1）",
}
DOPING_STEP = "LNCM 與 {src} 研磨混合後 600 °C 4 h 空氣（p10300）"
LNCM_FIRE = "900 °C 16 h 空氣（LNCM）"
NOT_SAID = "未摻雜 LNCM 是否也經過 600 °C 4 h 處理：未報告"
FULL_CELL = "全電池（vs LTO，0.5–3.3 V，0.2C／0.5D）放電容量（SI Table S3）：{c}；200 圈保持率 {r}（正文 p10305）"


def full_cell(values: list[str], retention: str) -> str:
    return FULL_CELL.format(c="；".join(f"全電池第{n}圈 {v}" for n, v in zip([10, 50, 100, 150, 200], values)),
                            r=retention)


def pang(sample: str, nominal: str, undoped: bool, element: str, x: str, site: str, fire: str, values: dict,
         note: str, prelim: bool = False) -> dict[str, str]:
    return {**PANG, **(PANG_PRELIM if prelim else PANG_MAIN), "論文中名義化學式_保留原文參數寫法": nominal,
            "樣品名": sample, "是否未摻雜對照": "是" if undoped else "否", "摻雜元素": element, "摻雜量_原文x": x,
            "摻雜位點與組成變動": site, "鍛燒溫度時間氣氛": fire, **values, "備註": note}


PANG_TABLE_NOTE = ("正文 Table 1（ICP-AES）在 pdftotext 擷取時欄位錯位，未轉錄。SI Table S3 的標題寫「18650-type full "
                   "batteries」，正文 p10305 寫「full-configuration coin cells」，兩處原文都照錄。Fig. 5、6 未讀值。")
PANG_ROWS = [
    pang("LNCM", "Li1.2Ni0.23Co0.12Mn0.5O2", True, "無", "0",
         "NPD：0.15(8)Li2MnO3·0.85(8)Li1.21(4)Ni0.27(4)Co0.13(3)Mn0.39(3)O2（p10302）", LNCM_FIRE,
         {"首圈充電容量_mAh_g": "341", "首圈放電容量_mAh_g": "262", "活化後或第N圈放電容量_mAh_g": "第2圈 263",
          "活化後或第N圈的循環次數": "2", "該N圈測試速率": "0.1C／0.1D",
          "其他已報告容量與條件": "首圈庫倫效率 77%；第 2 圈充電 266；倍率（2.0–4.6 V）放電 0.1／0.2／0.5／1 D："
                                 "242.8／229.0／214.0／199.9（圈數未報告）；"
                                 + full_cell(["198.5185", "184.7151", "178.7322", "170.3704", "163.0912"], "79%")},
         PANG_TABLE_NOTE),
    pang("Cr-LNCM", "Li1.2Ni0.23Co0.12Mn0.5O2 with 2% Cr", False, "Cr", "0.02",
         "原文寫「2% Cr」，未說明百分比的基準；作者推論 Cr 在 R-3m 3a（過渡金屬）位置；NPD：0.15(9)Li2MnO3·"
         "0.85(9)Li1.25(5)Ni0.26(5)Co0.11(4)Mn0.38(4)Cr0.01(4)O2（p10302–10303）；SI Table S2：Cr 0.85 wt%",
         f"{LNCM_FIRE}；{DOPING_STEP.format(src='Cr(NO3)3·H2O')}",
         {"首圈充電容量_mAh_g": NR, "首圈放電容量_mAh_g": "258", "活化後或第N圈放電容量_mAh_g": NR,
          "活化後或第N圈的循環次數": NR, "該N圈測試速率": NR,
          "其他已報告容量與條件": "首圈庫倫效率「similar to LNCM」（未列數值）；倍率（2.0–4.6 V）放電 0.1／0.2／0.5／1 D："
                                 "230.5／211.0／192.9／183（圈數未報告）；"
                                 + full_cell(["197.5419", "193.6034", "190.9218", "185.0978", "181.6201"], "90%")
                                 + "；前期測試（SI Fig. S1，1C）首圈不可逆容量 46.3"},
         f"{NOT_SAID}。{PANG_TABLE_NOTE}"),
    pang("F-LNCM", "Li1.2Ni0.23Co0.12Mn0.5O2 with 5% F", False, "F", "0.05",
         "作者推論 F 在 R-3m 6c（O）位置；NPD：0.15(16)Li2MnO3·0.85(16)Li1.20(5)Ni0.24(5)Co0.13(4)Mn0.43(4)"
         "O1.90(6)F0.10(6)（p10302）",
         f"{LNCM_FIRE}；{DOPING_STEP.format(src='NH4F')}",
         {"首圈充電容量_mAh_g": NR, "首圈放電容量_mAh_g": "274", "活化後或第N圈放電容量_mAh_g": NR,
          "活化後或第N圈的循環次數": NR, "該N圈測試速率": NR,
          "其他已報告容量與條件": "首圈庫倫效率 86%；倍率（2.0–4.6 V）放電 0.1／0.2／0.5／1.0 D："
                                 "246.4／234.6／223.8／208.9（圈數未報告）；"
                                 + full_cell(["198.3382", "174.4653", "160.3468", "152.0087", "142.9624"], "70%")
                                 + "；前期測試（SI Fig. S1）1C 容量 195.3（未說明圈數、充或放電）"},
         f"{NOT_SAID}。{PANG_TABLE_NOTE}"),
    *[pang(f"Cr {pct}%（前期測試）", f"Li1.2Ni0.23Co0.12Mn0.5O2 with {pct}% Cr", False, "Cr", x,
           "原文寫「Tests of 1, 2, and 3% Cr」（p10300），未說明百分比的基準", NR,
           {"其他已報告容量與條件": f"SI Fig. S1：首圈不可逆容量 {v} mAh/g"},
           "只在 SI Figure S1 出現的前期測試樣品；合成條件與其他測試條件未報告。Fig. S1 未讀值。", prelim=True)
      for pct, x, v in [("1", "0.01", "79.2"), ("3", "0.03", "80.14")]],
    *[pang(f"F {pct}%（前期測試）", f"Li1.2Ni0.23Co0.12Mn0.5O2 with {pct}% F", False, "F", x,
           "原文寫「Tests of 5, 7.5, and 10% F」（p10300）", NR,
           {"其他已報告容量與條件": f"SI Fig. S1：1C 容量 {v} mAh/g（未說明圈數、充或放電）"},
           "只在 SI Figure S1 出現的前期測試樣品；合成條件與其他測試條件未報告。Fig. S1 未讀值。", prelim=True)
      for pct, x, v in [("7.5", "0.075", "186.0"), ("10", "0.10", "153.1")]],
]

# ---------------------------------------------------------------------------------------------------------
# Giovine 2024, Chem. Mater. 36, 3643-3654 (pdfs/10.1021_acs.chemmater.3c03138.pdf + _SI.pdf; p3643 = PDF p1)
# ---------------------------------------------------------------------------------------------------------
GIOVINE = {
    "證據類別": "4 F 定量與合成路線（DRX）", "直接適用程度": "DRX（Li–Mn2+–Ti4+–O–F）；非層狀、非 Li2MnO3、無 Cr",
    "第一作者": "Giovine Raynald", "年份": "2024", "DOI": "10.1021/acs.chemmater.3c03138",
    "結構類型": "cation-disordered rock salt（DRX）",
    "原始全文網址": "本機 literature/pdfs/10.1021_acs.chemmater.3c03138.pdf 與 _SI.pdf（CC-BY 4.0）",
    "DOI網址": "https://doi.org/10.1021/acs.chemmater.3c03138",
    "其他重要控制/侷限": "ICP 以目標 Mn 含量為參考（假設沒有 Mn 流失）；O 沒有定量；DRX 化學式假設沒有陽離子、陰離子空缺"
                       "（Note S5）。19F NMR 看不到直接鍵結 Mn 的 F，所以 NMR 給的 DRX 中 F 是下限，上限由隨機分布機率換算"
                       "（p3646–3647）。",
    "F定量出處": "方法 p3645、p3646–3647（PDF p3–5）；ICP／F-ISE：SI Table S4（SI PDF p10）；DRX 化學式：正文 Table 1 "
                 "p3651（PDF p9）與正文",
    "相純度出處": "正文 p3647–3650（PDF p5–8）；SI Table S3、S6、S8（SI PDF p10、12、13）",
    "問題與風險出處": "正文 p3646–3651（PDF p4–9）",
}
SOLID = {
    "合成路線": "固相：球磨混合 → 壓錠 → 燒結", "前驅物": "LiF、Li2CO3、MnO、TiO2（anatase）", "氟來源": "LiF（99.9%）",
    "Li額外添加比例(相對目標)": "Li2CO3 過量 10 mol%",
    "前處理與混合": "Retsch PM200 行星式球磨 300 rpm 6 h；壓錠", "煆燒/熱處理溫度°C": "800",
    "煆燒/熱處理時間": "12 h（升溫 5 °C/min，自然冷卻至室溫）", "氣氛": "Ar 持續流通；燒完迅速移入 Ar 手套箱",
    "組成出處": "Experimental p3644（PDF p2）", "方法及前驅物出處": "Experimental p3644（PDF p2）",
    "Li添加出處": "Experimental p3644（PDF p2）", "熱處理出處": "Experimental p3644（PDF p2）",
}
SI_NOTE = ("SI 已讀。SI Table S3、S5、S6、S7 在 pdftotext 擷取時欄位錯位，相關數值採用正文的直接陳述；Table S4、S8 版面完整。"
           "Fig. 2–4、S3–S7 未讀值。")
TASKB_ROWS = [
    {**GIOVINE, **SOLID, "研究材料原文名義組成": "Li1.25Mn0.25Ti0.50O1.75F0.25（LMTF25）", "系列/樣品": "LMTF25（未水洗）",
     "F是否流失/有無證據": "有：F-ISE 測得 F0.17（目標 0.25），流失 0.08 F（總 F 的 33%）；ICP 測得 Li 流失 0.07（總 Li 的 5%）。"
                          "作者以 Li、F 等量流失推論 LiF 在 800 °C、Ar 流下揮發（p3647）",
     "F化學定量與實際晶格F": "ICP-OES（Li、Mn、Ti，以 Mn 為參考）＋ F-ISE（總 F）＋ 7Li／19F ss-NMR（DRX 與反磁性雜相）＋ 碳酸鹽"
                            "滴定（TiMS）。樣品 Li:Mn:Ti:F = 1.31:0.25:0.50:0.17（SI Table S4）。DRX 相 Li1.22Mn0.26Ti0.52O2−xFx，"
                            "x = 0.12–0.15（Table 1、p3648）。F 在 LiF 中占總 F 的 14–28%（下限／上限，p3648）",
     "相純度檢測": "SXRD：兩個 DRX 相（a = 4.171、4.178 Å，89:11）＋ 結晶 LiF 0.5 wt%；ss-NMR：Li 在 DRX 90%，LiF 2–4%、"
                  "Li2CO3 0%、Li2O／LiOH 6–8%；碳酸鹽滴定 Li2CO3 0.6 ± 0.5 mmol/mol DRX（p3647–3648）",
     "雜相與作者困難": "只看 XRD 會得到 DRX 占 99.5 wt%（98.8% 的 Li），ss-NMR 顯示 DRX 只含 90% 的 Li；X-ray 和中子無法區分 O 與 F"
                      "（p3643、p3647）；10 mol% Li 過量造成 ICP Li1.31 高於目標 1.25（p3647）",
     "備註與未讀附件": "電化學（Li 半電池，1.8–4.7 V）：首圈放電 210 mAh/g，平均放電電壓 3.12 V，664 Wh/kg（p3650）。電流："
                      "Experimental 寫 C/20（p3645），Fig. 4 圖說與正文寫 20 mA/g（p3649–3650）。" + SI_NOTE},
    {**GIOVINE, **SOLID,
     "研究材料原文名義組成": "Li1.25Mn0.20Ti0.55O1.85F0.15（LMTF15；Experimental 與正文）；摘要寫 Li1.25Mn0.225Ti0.525O1.85F0.15",
     "系列/樣品": "LMTF15",
     "F是否流失/有無證據": "有，較少：F-ISE 測得 F0.13（目標 0.15），流失為總 F 的 13%；Li 流失 5%（p3648、p3651）",
     "F化學定量與實際晶格F": "同 LMTF25 的方法。樣品 Li:Mn:Ti:F = 1.30:0.20:0.53:0.13（SI Table S4）。DRX 相 Li1.24Mn0.21Ti0.55O2−xFx，"
                            "x = 0.12–0.13（Table 1、p3648）。F 在 LiF 中占總 F 的 3–5%（p3648）",
     "相純度檢測": "SXRD：兩個 DRX 相（a = 4.154、4.164 Å，52:48），沒有結晶雜相；ss-NMR：Li 在 DRX 92%，LiF <1%、Li2CO3 0%、"
                  "Li2O／LiOH 7%；Li2CO3 5.6 ± 0.5 mmol/mol DRX（p3648）",
     "雜相與作者困難": "依 LMTF25 結果把目標 F 降到 0.15；作者稱此組成（約 6% F 取代）接近固相法、Li1.25 時的 F 溶解上限（p3648）；"
                      "10 mol% Li 過量仍偏高（ICP Li1.30）",
     "備註與未讀附件": "沒有電化學數據。" + SI_NOTE},
    {**GIOVINE, **SOLID, "研究材料原文名義組成": "Li1.25Mn0.25Ti0.50O1.75F0.25（LMTF25w）", "系列/樣品": "LMTF25w（LMTF25 水洗後）",
     "前處理與混合": "同 LMTF25；燒結後以 5 mL 除氣去離子水（freeze/pump/thaw 3 次）在 N2 下沖洗約 300 mg 粉末，N2 吹乾 24 h，"
                    "手套箱過渡艙真空乾燥過夜，回收率 80%（p3645–3646；SI Note S4）",
     "F是否流失/有無證據": "水洗使樣品 Li、F 下降：Li1.31、F0.17 → Li1.28、F0.12（ICP／F-ISE，p3649）",
     "F化學定量與實際晶格F": "樣品 Li:Mn:Ti:F = 1.28:0.25:0.50:0.12（SI Table S4）。DRX 相 Li1.25Mn0.25Ti0.50O1.89F0.11（p3650；"
                            "Table 1：F 0.11）。F 在 LiF 中占總 F 的 5–11%（p3649）",
     "相純度檢測": "實驗室 XRD 為單相；SXRD 有少量結晶 LiF；兩個 DRX 相比例 89:11 → 79:21；ss-NMR：Li 在 DRX 97%，LiF 0–1%、"
                  "Li2CO3 0%、Li2O／LiOH 2–3%；Li2CO3 0.3 ± 0.5 mmol/mol DRX（p3649–3650）",
     "雜相與作者困難": "作者稱 DRX Li 含量看似上升，但在 ss-NMR 擬合的 ±4% 誤差內（p3650）",
     "備註與未讀附件": "電化學（Li 半電池，20 mA/g，1.8–4.7 V）：首圈放電 232 mAh/g（未水洗 210）；25 圈後放電容量比未水洗高約 "
                      "11%；平均放電電壓 3.21 V；744 Wh/kg（p3650）。" + SI_NOTE},
    {**GIOVINE, "研究材料原文名義組成": "Li1.33Mn0.33Ti0.33O1.33F0.66（LMTF66）", "系列/樣品": "LMTF66",
     "合成路線": "機械化學（高能球磨），依 Lee et al.（文中 ref 10）的程序", "前驅物": "LiF、Li2O、MnO、TiO2（anatase）",
     "氟來源": "LiF（99.9%）", "Li額外添加比例(相對目標)": "Li2O 過量 10 mol%",
     "前處理與混合": "Ar 手套箱中 1 g 粉末裝入 50 mL 不鏽鋼罐（10 mm 球 5 顆、5 mm 球 10 顆），Ar 下密封；Retsch PM200 450 rpm "
                    "40 h（每 30 min 停 5 min、轉向反轉）；完成後在 Ar 手套箱中取粉",
     "煆燒/熱處理溫度°C": "不適用（沒有燒結）", "煆燒/熱處理時間": "球磨 40 h", "氣氛": "Ar（密封罐）",
     "F是否流失/有無證據": "F 流失極小，在 F-ISE 誤差內（F0.64，目標 0.66；SI Table S4）；Li 流失 0.08（5%）（p3651）。作者歸因"
                          "於密閉容器減少 F 揮發（p3651）",
     "F化學定量與實際晶格F": "同 LMTF25 的方法。樣品 Li:Mn:Ti:F = 1.38:0.33:0.32:0.64（SI Table S4）。DRX 相 Li1.27Mn0.37Ti0.36O2−xFx，"
                            "x = 0.56–0.62（Table 1、p3651）。F 在 LiF 中占總 F 的 4–12%（p3651）",
     "相純度檢測": "實驗室 XRD 和 SXRD：單一 rock salt 相，a = 4.208 Å，沒有結晶雜相；繞射峰明顯寬化，粒徑約 100–200 nm；"
                  "ss-NMR：Li 在 DRX 82%，LiF 2–6%、Li2CO3 0%、Li2O／LiOH 12–16%；Li2CO3 4.4 ± 0.5 mmol/mol DRX（p3650–3651）",
     "雜相與作者困難": "球磨可能使雜相非晶化，降低繞射工具的效用（p3650）；Li2O／LiOH 雜相比例在 4 個樣品中最高（p3651）",
     "組成出處": "Experimental p3644（PDF p2）", "方法及前驅物出處": "Experimental p3644（PDF p2）",
     "Li添加出處": "Experimental p3644（PDF p2）", "熱處理出處": "Experimental p3644（PDF p2）",
     "備註與未讀附件": "沒有本文的電化學數據（文中引用 Lee et al. 報告的 260 mAh/g，1.6–4.8 V）。作者結論：固相法 F 取代 <10%"
                      "（F<0.2），高能球磨可達約 30%（F0.6）（摘要、p3644、p3651）。" + SI_NOTE},
]


def extend(src, dst, rows: list[dict[str, str]]) -> pd.DataFrame:
    v1 = pd.read_csv(src, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    new = pd.DataFrame(rows)
    missing, extra = set(v1.columns) - set(new.columns), set(new.columns) - set(v1.columns)
    if missing or extra:
        raise RuntimeError(f"{dst.name}: missing {sorted(missing)}, extra {sorted(extra)}")
    out = pd.concat([v1, new[v1.columns]], ignore_index=True)
    out.to_csv(dst, index=False, encoding="utf-8-sig")
    print(f"{dst.name}: {len(v1)} v1 rows + {len(new)} new rows = {len(out)}")
    return out


def main() -> None:
    extend(SAMPLES_V1, SAMPLES_V2, KIM_ROWS + PANG_ROWS)
    extend(TASKB_V1, TASKB_V2, TASKB_ROWS)


if __name__ == "__main__":
    main()
