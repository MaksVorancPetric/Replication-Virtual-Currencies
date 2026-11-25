"""
01_clean.py — Build the analysis data from the raw files.

Python port of the authors' Code1_DataImportAndClean.do (OSF package).
Inputs  (never modified): data/raw/RawData_submission_Anonymized.xlsx
                          data/raw/Characteristics.xlsx
Outputs: data/clean/entries.csv all survey entries incl. non-completers (932)
         data/clean/wide.csv    one row per participant (753)
         data/clean/long.csv    one row per participant x lottery (753 x 16)

Each step follows the corresponding block in the .do file. Places where
we deliberately reproduce (or flag) a choice of the authors are marked
with "NOTE".
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
CLEAN.mkdir(parents=True, exist_ok=True)

# Lotteries in the order used in the paper (Table 1).
# Risky: "<p green>_<p red>"; ambiguous: A1..A5.
LOTTERIES = [
    "0_100", "10_90", "20_80", "30_70", "40_60", "50_50",
    "60_40", "70_30", "80_20", "90_10", "100_0",
    "Amb_Green_C", "Amb_Green_R", "Amb_Red_R", "Amb_Red_C", "Amb_Unknown",
]
LOTTERY_LABELS = [f"R{i}" for i in range(11)] + ["A1", "A2", "A3", "A4", "A5"]
# NOTE: paper labels A1 = green common, A2 = green rare, A3 = red common,
# A4 = red rare. The .do file orders Red_R before Red_C; we keep the
# paper's labels so figures match Table 1.
PAPER_LABEL = {
    "Amb_Green_C": "A1", "Amb_Green_R": "A2",
    "Amb_Red_C": "A3", "Amb_Red_R": "A4", "Amb_Unknown": "A5",
}
TREAT_LABELS = {1: "TC", 2: "T1", 3: "T100", 4: "T93", 5: "T107", 6: "T0.1"}


def clean_experiment() -> pd.DataFrame:
    df = pd.read_excel(RAW / "RawData_submission_Anonymized.xlsx")
    n_raw = len(df)

    df["treatment"] = pd.to_numeric(df["treatment"])
    df["win"] = pd.to_numeric(df["win"])  # tokens per GBP 1 in that treatment

    # Dependent variable: WTP normalised to GBP (divide by exchange rate)
    wtp = pd.DataFrame({f"C_{x}": df[f"Choice_{x}"] / df["win"] for x in LOTTERIES})
    df = pd.concat([df, wtp], axis=1).copy()

    # Completed = reached the last page (exchange-rate question)
    df["completed"] = df["T_Exchange_Rate_PageSubmit"].notna().astype(int)

    # Attention checks: input must equal 2x (risky) / 3x (ambiguous) the
    # maximum winnable amount. Compare with a tolerance (floating point).
    df["A_risky"] = df["Q_Att_risky"] / df["win"]
    df["A_amb"] = df["Q_Att_Amb"] / df["win"]
    df["attention"] = (np.isclose(df["A_risky"], 2) & np.isclose(df["A_amb"], 3)).astype(int)

    # Comprehension: number of attempts needed (0 = never passed)
    def passed(k):
        return ((df[f"Comprehensive_{k}_Q1"] == 1) & (df[f"Comprehensive_{k}_Q2"] == 2)
                & (df[f"Comprehensive_{k}_Q3"] == 1) & (df[f"Comprehensive_{k}_Q4"] == 2))
    df["N_Comprehension"] = np.select([passed(1), passed(2), passed(3)], [1, 2, 3], default=0)

    # Screened out = took all three comprehension rounds and did not finish
    df["screen_out"] = (df["T_Try1_PageSubmit"].notna() & df["T_Comprehensive_2_PageSubmit"].notna()
                        & df["T_Comprehensive_3_PageSubmit"].notna()
                        & (df["completed"] == 0)).astype(int)

    # Risk preference (Falk et al. item, 0-10) and 3-group version
    df["Risk_Preference"] = np.select(
        [df["Risk_Taking_Falk_1"] < 5, df["Risk_Taking_Falk_1"] == 5, df["Risk_Taking_Falk_1"] > 5],
        [0, 1, 2], default=np.nan)

    df["duration_min"] = df["Durationinseconds"] / 60
    # NOTE (authors' coding): Table 2 reports duration in a "minutes.seconds"
    # format (6.30 = 6 min 30 s), not decimal minutes. Reproduced here; the
    # mean of this variable is therefore not a meaningful duration.
    df["duration_mss"] = np.floor(df["duration_min"]) + (df["duration_min"] % 1) * 60 / 100

    # Loot box knowledge: answers 2 or 3 = knows what a loot box is
    # (missing answers are coded 0 by .isin; the check in 05 shows this is
    # identical to the authors' variable)
    df["Knowledge_Loot_box"] = df["knoledge_Loot_box"].isin([2, 3]).astype(int)

    # Order of blocks (risky first) from the Qualtrics flow variables
    rf = {"FL_25_DO": "FL_26|FL_27", "FL_606_DO": "FL_607|FL_622", "FL_216_DO": "FL_217|FL_232",
          "FL_411_DO": "FL_412|FL_427", "FL_816_DO": "FL_817|FL_832"}
    df["risky_first"] = np.any([df[c] == v for c, v in rf.items()], axis=0).astype(int)

    df["start"] = pd.to_datetime(df["StartDate"])
    df["wave"] = df["start"].dt.strftime("%Y-%m")

    keep = (["Q_ID", "treatment", "win", "completed", "screen_out", "attention", "N_Comprehension",
             "Risk_Taking_Falk_1", "Risk_Preference", "Knowledge_Loot_box", "duration_min", "duration_mss",
             "risky_first", "start", "wave"] + [f"C_{x}" for x in LOTTERIES])
    df = df[keep]

    print(f"[clean] raw survey rows: {n_raw}; completed: {df['completed'].sum()}; "
          f"screened out: {df['screen_out'].sum()}")
    # NOTE: some Q_IDs appear more than once (re-entries). The authors keep
    # the completed row only (drop if completed != 1); we do the same.
    dup = df["Q_ID"].duplicated(keep=False)
    print(f"[clean] rows with a repeated Q_ID: {dup.sum()} ({df.loc[dup, 'Q_ID'].nunique()} IDs)")
    return df


def clean_characteristics() -> pd.DataFrame:
    ch = pd.read_excel(RAW / "Characteristics.xlsx")
    ch = ch[ch["Time taken"].notna() & (ch["Completion code"] != "NOCODE")].copy()
    ch["Age"] = pd.to_numeric(ch["Age"])
    # NOTE (coding of Female): the authors' released data and code code "Prefer not to say" as 2
    # (Female_authors: Male 0, Female 1, Prefer not to say 2; identical to Female in their LongData.dta),
    # so the variable is not a pure dummy. The published Table 3 is reproduced cell by cell only if
    # "Prefer not to say" is coded 3 (Female_paper). We use Female_paper in the regressions; with the
    # released coding columns (2) and (4) differ by at most 0.003 (see 06_extension.py, table E7).
    ch["Female_authors"] = ch["Sex"].map({"Male": 0, "Female": 1, "Prefer not to say": 2})
    ch["Female_paper"] = ch["Sex"].map({"Male": 0, "Female": 1, "Prefer not to say": 3})
    ch["Female"] = ch["Sex"].map({"Male": 0, "Female": 1})
    ch["Student"] = ch["Student status"].map({"No": 0, "Yes": 1})
    ch = ch.rename(columns={"Participant id": "Q_ID"})
    return ch[["Q_ID", "Age", "Female", "Female_authors", "Female_paper", "Student"]]


def main():
    exp = clean_experiment()
    # All survey entries (incl. non-completers and re-entries), used by 06_extension.py
    exp.drop(columns=[c for c in exp.columns if c.startswith("C_")]).to_csv(CLEAN / "entries.csv", index=False)
    exp = exp[exp["completed"] == 1].copy()
    assert exp["Q_ID"].is_unique, "duplicate IDs among completed participants"

    ch = clean_characteristics()
    wide = exp.merge(ch, on="Q_ID", how="left", validate="1:1")
    print(f"[clean] final sample: {len(wide)}; missing demographics: {wide['Age'].isna().sum()}")

    wide["treat"] = wide["treatment"].map(TREAT_LABELS)
    # treatment_5: T93 and T107 pooled (as in the paper)
    wide["treat5"] = wide["treat"].replace({"T93": "T93&T107", "T107": "T93&T107"})
    wide.to_csv(CLEAN / "wide.csv", index=False)

    # Long format: one row per participant x lottery
    long = wide.melt(
        id_vars=[c for c in wide.columns if not c.startswith("C_")],
        value_vars=[f"C_{x}" for x in LOTTERIES], var_name="lottery_raw", value_name="wtp")
    long["lottery_raw"] = long["lottery_raw"].str[2:]
    long["lottery_id"] = long["lottery_raw"].map({x: i + 1 for i, x in enumerate(LOTTERIES)})
    risky_lab = {x: f"R{i}" for i, x in enumerate(LOTTERIES[:11])}
    long["lottery"] = long["lottery_raw"].map({**risky_lab, **PAPER_LABEL})
    long["risky"] = (long["lottery_id"] <= 11).astype(int)
    # Win probability for risky lotteries (0, 0.1, ..., 1)
    p = pd.to_numeric(long["lottery_raw"].str.split("_").str[0], errors="coerce") / 100
    long["p_win"] = p.where(long["risky"] == 1)
    long = long.sort_values(["treatment", "Q_ID", "lottery_id"])
    long.to_csv(CLEAN / "long.csv", index=False)
    print(f"[clean] long data: {len(long)} rows")


if __name__ == "__main__":
    main()
