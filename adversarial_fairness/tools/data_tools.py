"""
LangChain tools for dataset loading and sensitive attribute identification.

load_dataset      — generic CSV loader; LLM decides drops, binarization, target
identify_sensitive — LLM reads column names/types and returns sensitive attrs
                     + binarization rules for each
"""

import json
import re
import pandas as pd
import numpy as np
import torch
import ollama as _ollama
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from langchain.tools import tool
from langchain_ollama import OllamaLLM

from state import state

# LLM shared instance (tools import this too)
OLLAMA_MODEL = "llama3.1"   # must be a model pulled in Ollama (run: ollama list)

llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0.1)


def _llm_invoke(prompt: str) -> str:
    resp = _ollama.generate(
        model=OLLAMA_MODEL,
        prompt=prompt,
        options={"num_gpu": 0, "temperature": 0.1},
    )
    return resp.response


#
# Fingerprint — computed after load_dataset, used for lambda warm-start
#

def compute_fingerprint(state) -> dict:
    """
    Structural fingerprint of the loaded dataset.
    Stored in long-term memory and used to find similar past runs
    for lambda warm-starting.
    """
    y = state.y_train.cpu().numpy()
    s = state.sensitive_train.cpu().numpy()
    n = len(y)

    pos_rate       = float(y.mean())
    class_imbalance = round(min(pos_rate, 1.0 - pos_rate), 4)

    group_ratios = []
    for i in range(s.shape[1]):
        col = s[:, i]
        group_ratios.append(min(float((col == 0).mean()), float((col == 1).mean())))
    group_size_ratio = round(min(group_ratios), 4) if group_ratios else 0.5

    if n < 10_000:
        bucket = "small"
    elif n < 100_000:
        bucket = "medium"
    else:
        bucket = "large"

    return {
        "n_sensitive_attrs":   len(state.sensitive_attrs),
        "class_imbalance":     class_imbalance,
        "group_size_ratio":    group_size_ratio,
        "dataset_size_bucket": bucket,
    }


#
# Helpers
#

def _extract_json(text: str) -> dict:
    """Extract first JSON object from LLM prose response."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    raise ValueError(f"No valid JSON found in LLM response:\n{text}")


def _infer_dataset_key(dataset_path: str = "", dataset_name: str = "", columns=None) -> str:
    """Infer known dataset key from path/name/columns to avoid schema mismatches."""
    cols = list(columns) if columns is not None else []
    haystack = " ".join([
        str(dataset_path or "").lower(),
        str(dataset_name or "").lower(),
        " ".join([str(c).lower() for c in cols]),
    ])

    # Match specific dataset markers FIRST. The generic "income" -> adult rule is
    # last because several datasets contain "income" in their path/columns
    # (e.g. census_income_kdd, ACS PINCP) and must not be mistaken for Adult.
    if "compas" in haystack or "two_year_recid" in haystack:
        return "compas"
    if "german" in haystack or "credit" in haystack:
        return "german"
    if "utkface" in haystack or ("ethnicity" in haystack and "gender" in haystack):
        return "utkface"
    if "kdd" in haystack or "census-income" in haystack or "census_income" in haystack:
        return "kdd"
    if "acs" in haystack or "psam_p" in haystack or "pums" in haystack:
        return "acs"
    if "bank" in haystack or "bank_marketing" in haystack or "bank-additional" in haystack:
        return "bank"
    if "migration" in haystack or "legal_entry" in haystack or "hims" in haystack or "coastal_origin" in haystack:
        return "migration"
    if "adult" in haystack or "income" in haystack:
        return "adult"
    return ""


def _binarize_column(series: pd.Series, rule: dict) -> pd.Series:
    """
    Binarize a column according to an LLM-provided rule.
    rule = {"positive_value": "Male"}  → Male=1, else=0
    rule = {"threshold": 25}           → >25 = 1, else 0  (numeric only)
    Falls back to sort-and-pick if rule doesn't match column dtype.
    """
    is_numeric = pd.api.types.is_numeric_dtype(series)

    if "positive_value" in rule:
        pv = rule["positive_value"]
        # Cast positive_value to match series dtype
        if is_numeric:
            try:
                pv = type(series.dropna().iloc[0])(pv)
            except Exception:
                pass
        return (series == pv).astype(int)

    elif "threshold" in rule and is_numeric:
        return (series > rule["threshold"]).astype(int)

    else:
        # String column with threshold rule, or unknown rule — sort and pick
        try:
            vals = sorted(series.dropna().unique())
        except TypeError:
            vals = list(series.dropna().unique())
        pos = vals[1] if len(vals) > 1 else vals[0]
        return (series == pos).astype(int)


#
# Tool 1 — identify_sensitive
#

@tool
def identify_sensitive(dataset_path: str, dataset_name: str = "") -> str:
    """
    Reads dataset column names and sample values, then asks the LLM which
    columns are sensitive attributes (race, gender, age, etc.) and how to
    binarize them. Also asks the LLM to identify the target column and
    columns to drop (IDs, duplicates, leakage).

    Returns a JSON string with keys:
      - sensitive_attrs: list of column names
      - binarization_rules: {col: {rule}} per sensitive attr
      - target_col: name of the prediction target
      - columns_to_drop: list of cols to exclude from features
      - justification: LLM reasoning
    """
    # Column schemas for headerless datasets
    _headerless_schemas = {
        "adult": [
            "age", "workclass", "fnlwgt", "education", "education-num",
            "marital-status", "occupation", "relationship", "race", "sex",
            "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
        ],
        "kdd": [
            "age", "class_of_worker", "detailed_industry_recode", "detailed_occupation_recode",
            "education", "wage_per_hour", "enroll_in_edu_inst_last_wk", "marital_stat",
            "major_industry_code", "major_occupation_code", "race", "hispanic_origin", "sex",
            "member_of_labor_union", "reason_for_unemployment", "full_or_part_time_employment_stat",
            "capital_gains", "capital_losses", "dividends_from_stocks", "tax_filer_stat",
            "region_of_prev_residence", "state_of_prev_residence", "household_family_stat",
            "household_summary_in_household", "instance_weight",
            "migration_code_change_in_msa", "migration_code_change_in_reg",
            "migration_code_move_within_reg", "live_in_this_house_1yr_ago",
            "migration_prev_res_in_sunbelt", "num_persons_worked_for_employer",
            "family_members_under_18", "country_of_birth_father", "country_of_birth_mother",
            "country_of_birth_self", "citizenship", "own_business_or_self_employed",
            "fill_inc_questionnaire_for_veterans_admin", "veterans_benefits",
            "weeks_worked_in_year", "year", "income",
        ],
    }

    # Read a small sample for schema inference
    _kw = dict(nrows=5, skipinitialspace=True, na_values=["?", "NA", "N/A", ""])
    inferred_key_early = _infer_dataset_key(dataset_path, dataset_name, [])
    header_names = _headerless_schemas.get(inferred_key_early)
    try:
        if header_names:
            df = pd.read_csv(dataset_path, header=None, names=header_names, **_kw)
        else:
            df = pd.read_csv(dataset_path, **_kw)
        if len(df.columns) == 1:
            raise ValueError("Single column — wrong separator")
    except Exception:
        try:
            df = pd.read_csv(dataset_path, sep=";", **_kw)
        except Exception:
            df = pd.read_csv(dataset_path, sep=r"\s*,\s*", engine="python", **_kw)

    # Skip raw survey-code columns (e.g. V100, V102_M) — never sensitive attributes
    # and they bloat the prompt beyond what the LLM can handle.
    visible_cols = [c for c in df.columns if not re.match(r'^V\d', str(c))]
    n_skipped_v = len(df.columns) - len(visible_cols)
    if n_skipped_v:
        print(f"[info] Skipped {n_skipped_v} raw survey-code columns from LLM schema (V-prefixed)")

    inferred_key = _infer_dataset_key(dataset_path, dataset_name, df.columns)
    if inferred_key and dataset_name and inferred_key not in dataset_name.lower():
        print(f"[info] dataset name/path mismatch detected. Using inferred schema hint '{inferred_key}'.")

    # Per-dataset hardcoded fallbacks (used if the LLM fails or picks wrong cols).
    # Defined here (before schema trimming) so we can also use them to keep the
    # known target/sensitive columns visible in very wide files.
    fallbacks = {
        "adult":   {"sensitive_attrs": ["sex", "race"], "binarization_rules": {"sex": {"positive_value": "Male"}, "race": {"positive_value": "White"}}, "target_col": "income", "columns_to_drop": ["fnlwgt", "education-num"], "justification": "fallback"},
        "german":  {"sensitive_attrs": ["Sex", "Age"], "binarization_rules": {"Sex": {"positive_value": "male"}, "Age": {"threshold": 25}}, "target_col": "Class", "columns_to_drop": ["Unnamed: 0"], "justification": "fallback"},
        "compas":  {"sensitive_attrs": ["race", "sex"], "binarization_rules": {"race": {"positive_value": "Caucasian"}, "sex": {"positive_value": "Male"}}, "target_col": "two_year_recid", "columns_to_drop": ["id", "name", "first", "last", "dob", "c_case_number", "r_case_number", "vr_case_number", "compas_screening_date", "c_jail_in", "c_jail_out", "r_offense_date", "r_jail_in", "r_jail_out", "vr_offense_date", "screening_date", "v_screening_date", "in_custody", "out_custody", "c_offense_date", "c_arrest_date", "is_recid", "is_violent_recid", "decile_score", "decile_score.1", "score_text", "v_decile_score", "v_score_text", "priors_count.1"], "justification": "fallback"},
        "utkface": {"sensitive_attrs": ["ethnicity", "gender"], "binarization_rules": {"ethnicity": {"positive_value": 0}, "gender": {"positive_value": 0}}, "target_col": "age", "columns_to_drop": ["img_name", "pixels"], "justification": "fallback"},
        "kdd":     {"sensitive_attrs": ["race", "sex"], "binarization_rules": {"race": {"positive_value": "White"}, "sex": {"positive_value": "Male"}}, "target_col": "income", "columns_to_drop": ["instance_weight"], "justification": "fallback"},
        "bank":    {"sensitive_attrs": ["age"], "binarization_rules": {"age": {"threshold": 40}}, "target_col": "y", "columns_to_drop": [], "justification": "fallback"},
        "acs":     {"sensitive_attrs": ["RAC1P", "SEX"], "binarization_rules": {"RAC1P": {"positive_value": 1}, "SEX": {"positive_value": 1}}, "target_col": "PINCP", "columns_to_drop": ["RT", "SERIALNO", "SPORDER", "PUMA", "ADJINC", "PWGTP"] + [f"PWGTP{i}" for i in range(1, 81)], "justification": "fallback"},
    }

    # Cap schema at 40 columns to keep the prompt small. For most datasets the
    # demographics/target sit up front, so the first 40 are enough. BUT some wide
    # files put them late — ACS PUMS (286 cols) has SEX@68, PINCP@103, RAC1P@111
    # and KDD has income@41 — so we ALWAYS additionally keep the recognized
    # dataset's target + sensitive columns. Without this the LLM never sees them
    # and cannot possibly pick them (and the guardrails can't catch it because the
    # wrong-but-existing target it picks instead passes the "exists" check).
    MAX_SCHEMA_COLS = 40
    if len(visible_cols) > MAX_SCHEMA_COLS:
        head     = visible_cols[:MAX_SCHEMA_COLS]
        fb       = fallbacks.get(inferred_key, {})
        must_keep = [fb.get("target_col")] + list(fb.get("sensitive_attrs", []))
        extra    = [c for c in visible_cols[MAX_SCHEMA_COLS:] if c in must_keep]
        n_trimmed = len(visible_cols) - len(head) - len(extra)
        visible_cols = head + extra
        msg = f"[info] Schema trimmed to first {MAX_SCHEMA_COLS} columns ({n_trimmed} later columns hidden)"
        if extra:
            msg += f"; kept key columns out of order: {extra}"
        print(msg)

    # Compact format: 1 sample value per column keeps the prompt short
    schema_str = ""
    for col in visible_cols:
        sample = df[col].dropna().iloc[0] if df[col].dropna().shape[0] > 0 else "N/A"
        schema_str += f"  - {col}: {sample}\n"

    prompt_dataset_name = inferred_key or dataset_name or "unknown"

    # How many sensitive attributes to request, per dataset (user spec):
    #   migration -> 3, bank -> 1 (age only), everything else (incl. uploads) -> 2.
    _N_SENSITIVE = {"migration": 3, "bank": 1}
    n_sensitive = _N_SENSITIVE.get(inferred_key, 2)

    # Count-specific guidance for step 2 of the prompt.
    if n_sensitive == 1:
        sens_guidance = (
            "   - Choose THE SINGLE most important sensitive attribute "
            "(typically age; otherwise gender/sex).\n"
        )
    elif n_sensitive == 3:
        sens_guidance = (
            "   - Choose 3 attributes that span DIFFERENT demographic dimensions — "
            "ideally one from each category:\n"
            "       (a) Gender / sex\n"
            "       (b) Geographic origin (region, province, coast of birth)\n"
            "       (c) Socioeconomic background (education level, class, or occupation)\n"
            "   - Do NOT pick two attributes that measure the same underlying concept "
            "(e.g. not two geographic-origin columns).\n"
        )
    else:  # 2
        sens_guidance = (
            "   - Choose the 2 most important sensitive attributes, from DIFFERENT "
            "dimensions (e.g. gender/sex AND race/ethnicity, or gender AND age).\n"
            "   - Do NOT pick two attributes that measure the same underlying concept.\n"
        )

    # JSON example with exactly n_sensitive placeholders.
    _ex_attrs = ", ".join(f'"col{i+1}"' for i in range(n_sensitive))
    _ex_rules = ",\n".join(
        f'    "col{i+1}": {{"positive_value": "value"}}' for i in range(n_sensitive)
    )

    prompt = f"""You are a fairness-aware ML expert analyzing a dataset for bias correction.

Dataset name: {prompt_dataset_name}
Columns and sample values:
{schema_str}

Your task:
1. Identify the TARGET column — the legal, administrative, or socioeconomic STATUS outcome we want to predict.
   - Good targets: income level, employment status, legal authorization/entry status, credit approval, recidivism, job quality index.
   - NOT a good target: geographic destination, travel route, country visited, or any column that is itself a sensitive demographic attribute.
   - The target should be something a classifier predicts to make a decision ABOUT the person.

2. Identify EXACTLY {n_sensitive} SENSITIVE/PROTECTED ATTRIBUTE(S) — pre-existing personal demographic characteristics the person was born with or had BEFORE the event being studied, and that should NOT influence the prediction.
   - Good examples: gender/sex, region/province/coast of BIRTH or ORIGIN (where they came from), education level category at the time of migration, ethnicity, religion, age.
   - NOT sensitive:
     * "current" columns (current country, current city, current job) — these are POST-event outcomes or choices, not pre-existing traits.
     * destination country or country of residence — this is where someone ended up, not where they came from.
     * computed index or score columns (quality indices, benefit sums, composite scores).
     * binary flag derivations of an attribute already in your list (e.g., if you pick an education-level column, do NOT also pick a "has_higher_educ" binary flag).
{sens_guidance}   - If both an original multi-category column AND a binary derived version of the same concept are present, prefer the original categorical column (e.g., prefer an education-level category over a binary "has higher education" flag).
   - Return EXACTLY {n_sensitive} attribute(s) in "sensitive_attrs" — no more, no fewer.

3. For each sensitive attr, define a binarization rule: either {{"positive_value": "<value>"}} for categorical or {{"threshold": <number>}} for numeric.

4. Identify columns to drop: IDs, row numbers, duplicates, direct leakage of the target.

Respond with ONLY this JSON (no prose):
{{
  "sensitive_attrs": [{_ex_attrs}],
  "binarization_rules": {{
{_ex_rules}
  }},
  "target_col": "column_name",
  "columns_to_drop": ["id_col", "name_col"],
  "justification": "brief reason"
}}"""

    fallback_key = inferred_key or _infer_dataset_key("", dataset_name, [])

    try:
        response = _llm_invoke(prompt)
        result = _extract_json(response)

        normalized_cols = {str(c).strip() for c in df.columns}

        # Guardrail 1: target column must exist in the dataset
        target_missing = str(result.get("target_col", "")).strip() not in normalized_cols

        # Guardrail 2: sensitive attrs must not be weight/ID columns that are in
        # the known drop list. The LLM sometimes picks PWGTP1/PWGTP2 for ACS.
        known_drop = set(fallbacks.get(fallback_key, {}).get("columns_to_drop", []))
        bad_attrs = [
            a for a in result.get("sensitive_attrs", [])
            if a in known_drop or str(a).startswith("PWGTP")
        ]

        # Guardrail 3: target must not be a known-drop or weight column.
        # ACS: LLM sometimes picks PWGTP (person weight) instead of PINCP.
        proposed_target = str(result.get("target_col", "")).strip()
        bad_target = proposed_target in known_drop or proposed_target.startswith("PWGTP")

        if (target_missing or bad_target or bad_attrs) and fallback_key in fallbacks:
            print("\n" + "!" * 60)
            print(f"  LLM FAILED — dataset: {fallback_key.upper()}")
            if target_missing:
                print(f"     Wrong target column : '{result.get('target_col')}' not in dataset")
            if bad_target:
                print(f"     Bad target column   : '{proposed_target}' is a weight/drop column")
            if bad_attrs:
                print(f"     Wrong sensitive attrs: {bad_attrs}")
                print(f"     (picked weight/ID columns instead of demographic ones)")
            print(f"  -> Using hardcoded fallback: {fallbacks[fallback_key]['sensitive_attrs']}")
            print("!" * 60 + "\n")
            result = fallbacks[fallback_key]
        else:
            print(f"[llm] identify_sensitive OK -> {result.get('sensitive_attrs')} | target={result.get('target_col')}")
    except Exception as e:
        matched = fallbacks.get(fallback_key)
        if matched:
            print("\n" + "!" * 60)
            print(f"  [!] LLM EXCEPTION — dataset: {fallback_key.upper()}")
            print(f"     Error: {e}")
            print(f"  -> Using hardcoded fallback: {matched['sensitive_attrs']}")
            print("!" * 60 + "\n")
            result = matched
        else:
            raise RuntimeError(
                f"LLM failed for '{dataset_name}' and no fallback exists.\n"
                f"  Cause: {type(e).__name__}: {e}"
            )

    # Enforce the required number of sensitive attributes for this dataset.
    # (migration=3, bank=1, others=2). Trim if the LLM returned too many.
    sa = result.get("sensitive_attrs", [])
    if len(sa) > n_sensitive:
        kept = sa[:n_sensitive]
        print(f"[guardrail] LLM returned {len(sa)} sensitive attrs; keeping first "
              f"{n_sensitive}: {kept}")
        result["sensitive_attrs"] = kept
        rules = result.get("binarization_rules", {})
        result["binarization_rules"] = {a: rules[a] for a in kept if a in rules}
    elif len(sa) < n_sensitive:
        print(f"[guardrail] WARNING: expected {n_sensitive} sensitive attrs but LLM "
              f"returned {len(sa)}: {sa} — proceeding with what was returned.")

    # Guardrail: the LLM sometimes fuses a column name and its positive value into
    # one string, e.g. "race: White" or "sex=Female", instead of returning the bare
    # column name "race" with a separate binarization rule. Split these back out so
    # the column lookup in load_dataset succeeds. The trailing value, if present and
    # the attr has no rule yet, becomes the positive_value of its binarization rule.
    normalized_cols = {str(c).strip() for c in df.columns}

    def _split_col_value(token: str):
        """'race: White' -> ('race', 'White'); 'race' -> ('race', None)."""
        t = str(token).strip()
        if t in normalized_cols:
            return t, None
        for sep in (":", "="):
            if sep in t:
                col, val = t.split(sep, 1)
                col, val = col.strip(), val.strip()
                if col in normalized_cols:
                    return col, (val or None)
        return t, None   # leave unchanged if we can't resolve it

    rules = dict(result.get("binarization_rules", {}) or {})
    clean_attrs = []
    for attr in result.get("sensitive_attrs", []):
        col, val = _split_col_value(attr)
        clean_attrs.append(col)
        if col != attr:
            rule = rules.pop(attr, None) or rules.get(col)
            if rule is None and val is not None:
                rule = {"positive_value": val}
            if rule is not None:
                rules[col] = rule
            print(f"[guardrail] normalized sensitive attr '{attr}' -> column '{col}'"
                  + (f" (positive_value='{val}')" if val is not None else ""))
    result["sensitive_attrs"] = clean_attrs
    result["binarization_rules"] = rules

    # Same fix for the target column ("income: >50K" -> "income").
    tgt_col, _ = _split_col_value(result.get("target_col", ""))
    if tgt_col != str(result.get("target_col", "")).strip():
        print(f"[guardrail] normalized target '{result.get('target_col')}' -> '{tgt_col}'")
        result["target_col"] = tgt_col

    # Store in global state
    state.sensitive_attrs = result["sensitive_attrs"]
    state.binarization_rules = result["binarization_rules"]
    state.target_col = result["target_col"]
    state.columns_to_drop = result.get("columns_to_drop", [])
    state.dataset_name = dataset_name
    state.dataset_path = dataset_path

    return json.dumps(result, indent=2)


# Tool 2 — load_dataset

@tool
def load_dataset(dataset_path: str, dataset_name: str = "") -> str:
    """
    Generic dataset loader. Requires identify_sensitive to have been called first
    (to know target, sensitive attrs, binarization rules, columns to drop).

    Handles:
    - Multiple separators (comma, semicolon, whitespace)
    - Missing value markers (?, NA, N/A, empty)
    - Numeric and categorical columns
    - Binarization of sensitive attributes per LLM rules
    - One-hot encoding of remaining categoricals
    - Standard scaling of features
    - 80/20 train/test split

    Returns a summary string of the loaded dataset.
    """
    if not state.target_col:
        return "ERROR: Call identify_sensitive first to set target_col and sensitive_attrs."

    # Load raw CSV
    load_kwargs = dict(skipinitialspace=True, na_values=["?", "NA", "N/A", ""])

    try:
        df = pd.read_csv(dataset_path, **load_kwargs)
        # Detect accidental single-column load caused by wrong separator
        if len(df.columns) == 1:
            raise ValueError("Single column — wrong separator")
    except Exception:
        try:
            df = pd.read_csv(dataset_path, sep=";", **load_kwargs)
        except Exception:
            df = pd.read_csv(dataset_path, sep=r"\s*,\s*", engine="python", header=None, **load_kwargs)
        # Try to assign column names from .names file if available
        # (KDD case — we'll let the LLM handle naming via identify_sensitive)

    # Some raw datasets ship without headers. If the target column is missing,
    # reload with a known schema for the selected dataset.
    if state.target_col not in df.columns:
        name_lower = (dataset_name or state.dataset_name or "").lower()
        inferred_key = _infer_dataset_key(dataset_path, name_lower, df.columns)
        schema_map = {
            "adult": [
                "age", "workclass", "fnlwgt", "education", "education-num",
                "marital-status", "occupation", "relationship", "race", "sex",
                "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
            ],
            "kdd": [
                "age", "class_of_worker", "detailed_industry_recode", "detailed_occupation_recode",
                "education", "wage_per_hour", "enroll_in_edu_inst_last_wk", "marital_stat",
                "major_industry_code", "major_occupation_code", "race", "hispanic_origin", "sex",
                "member_of_labor_union", "reason_for_unemployment", "full_or_part_time_employment_stat",
                "capital_gains", "capital_losses", "dividends_from_stocks", "tax_filer_stat",
                "region_of_prev_residence", "state_of_prev_residence", "household_family_stat",
                "household_summary_in_household", "instance_weight",
                "migration_code_change_in_msa", "migration_code_change_in_reg",
                "migration_code_move_within_reg", "live_in_this_house_1yr_ago",
                "migration_prev_res_in_sunbelt", "num_persons_worked_for_employer",
                "family_members_under_18", "country_of_birth_father", "country_of_birth_mother",
                "country_of_birth_self", "citizenship", "own_business_or_self_employed",
                "fill_inc_questionnaire_for_veterans_admin", "veterans_benefits",
                "weeks_worked_in_year", "year", "income",
            ],
        }
        # Prefer path/column inference over user-supplied dataset name when they conflict.
        if inferred_key:
            name_lower = inferred_key
        for key, columns in schema_map.items():
            if key in name_lower:
                df = pd.read_csv(dataset_path, header=None, names=columns, **load_kwargs)
                break

    df.columns = [str(c).strip() for c in df.columns]

    # Migration: drop all raw survey code columns (V-prefixed)
    if state.dataset_name == "migration":
        v_cols = [c for c in df.columns if re.match(r'^V\d', c)]
        df.drop(columns=v_cols, inplace=True, errors='ignore')
        print(f"[info] Migration: dropped {len(v_cols)} raw survey code columns (V-prefixed)")

    # UTKFace: expand space-separated pixel string into pixel_0…pixel_N cols
    if "pixels" in df.columns:
        print("[info] Detected 'pixels' column — expanding into individual pixel features...")
        # Parse each row directly to float32 via numpy — avoids a large object DataFrame
        pixel_arrays = df["pixels"].astype(str).apply(
            lambda s: np.fromstring(s, dtype=np.float32, sep=" ")
        ).values
        pixel_matrix = np.vstack(pixel_arrays)  # (N, n_pixels) float32
        pixel_cols = [f"pixel_{i}" for i in range(pixel_matrix.shape[1])]
        pixel_df = pd.DataFrame(pixel_matrix, columns=pixel_cols, index=df.index)
        df = pd.concat([df.drop(columns=["pixels"]), pixel_df], axis=1)
        # Remove "pixels" from drop list since we already handled it
        state.columns_to_drop = [c for c in state.columns_to_drop if c != "pixels"]

    # Drop irrelevant columns
    drop_cols = [c for c in state.columns_to_drop if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    # Drop rows with missing target
    df.dropna(subset=[state.target_col], inplace=True)

    # Binarize target
    target_series = df[state.target_col].str.strip() if df[state.target_col].dtype == object else df[state.target_col]
    unique_targets = sorted(target_series.dropna().unique())
    if len(unique_targets) == 2:
        # Binary: higher/second value = 1 (e.g., ">50K", "2", True)
        pos_target = unique_targets[1]
        y = (target_series == pos_target).astype(int).values
    else:
        # Multiclass: try numeric median, else label-encode then split at median
        numeric_target = pd.to_numeric(target_series, errors="coerce")
        if numeric_target.notna().sum() > len(target_series) * 0.9:
            med = numeric_target.median()
            y = (numeric_target > med).astype(int).values
        else:
            # Categorical with >2 classes: encode and split at median code
            codes = pd.Categorical(target_series).codes
            med = np.median(codes)
            y = (codes > med).astype(int)
            print(f"[warn] Target '{state.target_col}' has {len(unique_targets)} string classes — "
                  f"binarized by label-code median ({med:.0f})")

    df.drop(columns=[state.target_col], inplace=True)

    # Extract and binarize sensitive attributes
    sensitive_cols = []
    for attr in state.sensitive_attrs:
        if attr not in df.columns:
            print(f"[warn] sensitive attr '{attr}' not found in columns: {df.columns.tolist()}")
            continue
        rule = state.binarization_rules.get(attr, {})
        s = df[attr]
        if df[attr].dtype == object:
            s = s.str.strip()
        s_bin = _binarize_column(s, rule)
        sensitive_cols.append(s_bin.values)
        df.drop(columns=[attr], inplace=True)

    if not sensitive_cols:
        return (
            "ERROR: none of the sensitive attributes "
            f"{state.sensitive_attrs} were found among the dataset columns "
            f"{df.columns.tolist()}. Check identify_sensitive output / fallback."
        )

    sensitive_matrix = np.stack(sensitive_cols, axis=1)  # (N, n_sensitive)

    # Drop high-cardinality string columns (IDs, free text, charge descriptions)
    # Use two thresholds:
    #   > MAX_UNIQUE_ABS  unique values → always drop (e.g. case numbers, descriptions)
    #   > MAX_UNIQUE_FRAC of rows       → drop (e.g. dates unique per row)
    MAX_UNIQUE_ABS  = 50
    MAX_UNIQUE_FRAC = 0.1
    for col in df.select_dtypes(include="object").columns:
        n_unique = df[col].nunique()
        if n_unique > MAX_UNIQUE_ABS or n_unique > MAX_UNIQUE_FRAC * len(df):
            df.drop(columns=[col], inplace=True)
            print(f"[info] Dropped high-cardinality column: {col} ({n_unique} unique)")

    # Fill remaining missing values
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip().fillna("Unknown")
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())

    # Subsample very large datasets to cap memory and training time
    MAX_SAMPLES = 100_000
    if len(df) > MAX_SAMPLES:
        rng = np.random.default_rng(42)
        keep = rng.choice(len(df), size=MAX_SAMPLES, replace=False)
        keep.sort()
        df = df.iloc[keep].reset_index(drop=True)
        y  = y[keep]
        sensitive_matrix = sensitive_matrix[keep]
        print(f"[info] Subsampled to {MAX_SAMPLES} rows (was {len(keep) + (len(y) - MAX_SAMPLES)})")

    # One-hot encode categoricals
    df = pd.get_dummies(df, drop_first=True)

    # Convert directly to float32 — avoids an intermediate float64 allocation
    X = df.to_numpy(dtype=np.float32, na_value=0.0)
    state.feature_names = df.columns.tolist()

    # Train/test split
    idx = np.arange(len(X))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)

    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    s_tr, s_te = sensitive_matrix[train_idx], sensitive_matrix[test_idx]

    # Drop near-constant columns before scaling (std ~= 0 causes NaN after scale)
    col_stds = X_tr.std(axis=0)
    valid_cols = col_stds > 1e-6
    if not valid_cols.all():
        n_dropped = (~valid_cols).sum()
        print(f"[info] Dropped {n_dropped} near-constant columns (std ~= 0)")
        X_tr = X_tr[:, valid_cols]
        X_te = X_te[:, valid_cols]
        state.feature_names = [n for n, v in zip(state.feature_names, valid_cols) if v]

    # Normalize features
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)

    # Replace any residual NaN/Inf with 0 (safety net)
    X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
    X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute class imbalance weight (n_negative / n_positive) for BCELoss
    n_pos = float(y_tr.sum())
    n_neg = float(len(y_tr) - n_pos)
    state.pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0
    if state.pos_weight > 2.0:
        print(f"[data] Class imbalance detected: pos_weight={state.pos_weight:.2f} "
              f"({n_pos:.0f} pos / {n_neg:.0f} neg) — will use weighted loss")

    # Convert to tensors
    dev = state.device
    state.X_train = torch.tensor(X_tr, dtype=torch.float32).to(dev)
    state.X_test  = torch.tensor(X_te, dtype=torch.float32).to(dev)
    state.y_train = torch.tensor(y_tr, dtype=torch.float32).to(dev)
    state.y_test  = torch.tensor(y_te, dtype=torch.float32).to(dev)
    state.sensitive_train = torch.tensor(s_tr, dtype=torch.float32).to(dev)
    state.sensitive_test  = torch.tensor(s_te, dtype=torch.float32).to(dev)

    summary = (
        f"Dataset loaded: {len(X)} samples, {X.shape[1]} features\n"
        f"Train: {len(X_tr)} | Test: {len(X_te)}\n"
        f"Target: '{state.target_col}' | Positive rate: {y.mean():.2%}\n"
        f"Sensitive attrs: {state.sensitive_attrs}\n"
        f"Device: {state.device}"
    )
    print(f"[data] {summary}")
    return summary
