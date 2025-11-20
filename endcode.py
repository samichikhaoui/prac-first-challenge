"""
1) Basic log analysis
   - compute simple statistics (cases, events, variants, etc.)
   - create three descriptive figures:
       * activity_frequency.png
       * case_length_distribution.png
       * case_duration_days.png

2) Process discovery on a filtered log
   - reduce the log to the top N most frequent variants
   - discover two Petri nets:
       * Inductive Miner
       * Heuristics Miner
   - compute quality metrics:
       * fitness (token-based)
       * precision (ETConformance)
       * generalization
       * two custom simplicity metrics
   - save Petri net visualizations:
       * petri_inductive.png
       * petri_heuristics.png
3) BPMN conformance
   - load a hand-made BPMN model (Neuer Prozess3.bpmn)
   - convert it to a Petri net
   - filter the log to:
       * only 'complete' lifecycle events
       * only activities that appear in the BPMN
   - compute fitness, precision, generalization and simplicity
     for the BPMN-based Petri net.
"""

###############################################################################
# Imports and global settings
###############################################################################

import os

import pm4py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pm4py.statistics.variants.log import get as variants_get
from pm4py.objects.log.obj import EventLog

# Quality metrics for discovered Petri nets
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.algo.evaluation.precision import algorithm as precision_algorithm
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator

# BPMN import and conversion
from pm4py.objects.bpmn.importer import importer as bpmn_importer
from pm4py.objects.conversion.bpmn import converter as bpmn_converter

# Conformance + simplicity for BPMN-based Petri net
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.algo.evaluation.precision import algorithm as precision_eval
from pm4py.algo.evaluation.generalization import algorithm as gen_eval
from pm4py.algo.evaluation.simplicity import algorithm as simp_eval


# --- base paths (adapt to repo structure if needed) ---

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_PATH = os.path.join(BASE_DIR, "data", "BPI Challenge 2017.xes.gz")
BPMN_PATH = os.path.join(BASE_DIR, "Neuer Prozess3.bpmn")

FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# --- column names used in the event log ---

CASE_COL = "case:concept:name"
ACT_COL = "concept:name"
TIMESTAMP_COL = "time:timestamp"
LIFECYCLE_COL = "lifecycle:transition"


# 1) BASIC ANALYSIS
# In this first part I just try to “get to know” the log.
# Before talking about miners, metrics or BPMN, I want to see:
# - how big the log is,
# - how many activities and variants it has,
# - how long cases stay in the system,
# - and how much rework there is.
# The idea is: if I don’t understand the basic distributions,
# any fancy model later will be hard to interpret

def basic_analysis(log):
    print("=== BASIC ANALYSIS ===")

    # Convert the event log into a pandas DataFrame
    df = pm4py.convert_to_dataframe(log)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])

    # Separate case-level and event-level attributes
    case_attrs = [c for c in df.columns if c.startswith("case:")]
    event_attrs = [c for c in df.columns if not c.startswith("case:")]

    # ---------------- Basic metrics ----------------
    n_events = len(df)
    n_cases = df[CASE_COL].nunique()
    n_activities = df[ACT_COL].nunique()

    # Variants = distinct sequences of activities (trace variants)
    variants = variants_get.get_variants(log)
    n_variants = len(variants)

    # Number of labels at case and event level
    n_case_labels = len(case_attrs)
    n_event_labels = len(event_attrs)

    # Case length = number of events per case
    case_lengths = df.groupby(CASE_COL)[ACT_COL].count()
    mean_case_length = case_lengths.mean()
    std_case_length = case_lengths.std()

    # Case duration = time between first and last event per case
    case_start = df.groupby(CASE_COL)[TIMESTAMP_COL].min()
    case_end = df.groupby(CASE_COL)[TIMESTAMP_COL].max()
    durations = case_end - case_start
    dur_sec = durations.dt.total_seconds()

    mean_dur_sec = dur_sec.mean()
    std_dur_sec = dur_sec.std()
    mean_dur_days = mean_dur_sec / 86400
    std_dur_days = std_dur_sec / 86400

    # Categorical event attributes (for possible further analysis)
    cat_event_attrs = [
        c for c in event_attrs
        if df[c].dtype == "object"
    ]
    n_cat_event_attrs = len(cat_event_attrs)

    # ---------------- Extra “bonus” metrics ----------------

    # Most frequent activity in the log
    most_freq_act = df[ACT_COL].value_counts().idxmax()

    # Most frequent resource (assuming the column exists)
    most_freq_res = df["org:resource"].value_counts().idxmax()

    # Percentage of cases that contain rework (an activity that repeats)
    rework_cases = df.groupby(CASE_COL)[ACT_COL].apply(
        lambda x: x.duplicated().any()
    )
    pct_rework = 100 * rework_cases.mean()

    # Print everything
    print("Number of events:", n_events)
    print("Number of cases:", n_cases)
    print("Number of unique activities:", n_activities)
    print("Number of variants:", n_variants)
    print("Number of case attributes:", n_case_labels)
    print("Number of event attributes:", n_event_labels)
    print(
        "Case length (mean ± std): "
        f"{mean_case_length:.2f} ± {std_case_length:.2f}"
    )
    print("Average case duration (days):", f"{mean_dur_days:.2f}")
    print("Std. dev. of case duration (days):", f"{std_dur_days:.2f}")
    print("Number of categorical event attributes:", n_cat_event_attrs)
    print("Most frequent activity:", most_freq_act)
    print("Most used resource:", most_freq_res)
    print("Percentage of cases with rework:", f"{pct_rework:.2f}%")
    print()

    # ---------------- Visualizations ----------------
    sns.set(style="whitegrid")

    # 1) Activity frequency
    plt.figure(figsize=(8, 6))
    sns.countplot(
        y=ACT_COL,
        data=df,
        order=df[ACT_COL].value_counts().index
    )
    plt.title("Activity Frequency")
    plt.xlabel("Number of events")
    plt.ylabel("Activity")
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "activity_frequency.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")

    # 2) Distribution of events per case
    plt.figure(figsize=(8, 5))
    plt.hist(case_lengths, bins=50)
    plt.title("Distribution of Events per Case")
    plt.xlabel("Number of events")
    plt.ylabel("Number of cases")
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "case_length_distribution.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")

    # 3) Distribution of case durations (in days)
    plt.figure(figsize=(8, 5))
    plt.hist(durations.dt.days, bins=50)
    plt.title("Distribution of Case Durations (in Days)")
    plt.xlabel("Duration (days)")
    plt.ylabel("Number of cases")
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "case_duration_days.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")
    print()



# 2) PROCESS DISCOVERY (Inductive + Heuristics Miner)
# This second part is where I let the algorithms “draw” the process for me.
# Instead of trying to sketch the model by hand directly . Chat Gpt helped me
# To avoid a crazy spaghetti model, I:
# - focus on the top N variants (e.g. 300) that repeat most often,
# - discover one model with the Inductive Miner (clean, block-structured),
# - and another with the Heuristics Miner (more raw, shows messy reality).
# Then I compute fitness, precision, generalization, and two simplicity
# metrics.


def reduce_log_to_top_variants(log_full, n_variants=300):
    print(f"Computing variants and keeping top {n_variants}...")

    variants_count = variants_get.get_variants(log_full)
    sorted_variants = sorted(
        variants_count.items(),
        key=lambda x: -len(x[1])  # sort by number of traces (descending)
    )

    top_variants = sorted_variants[:n_variants]

    # Collect all traces for those top variants
    filtered_traces = []
    for var_name, traces in top_variants:
        filtered_traces.extend(traces)

    # Build a new EventLog from the selected traces
    reduced_log = EventLog(filtered_traces)

    print(
        f"Filtered log size: {len(reduced_log)} traces "
        f"(vs {len(log_full)} in full log)"
    )
    print()
    return reduced_log


def compute_all_metrics(log_for_metrics, net, im, fm, name="model"):
    """
    Compute the quality metrics:
    - Fitness (token-based)
    - Precision (ETConformance)
    - Generalization
    - Two custom simplicity metrics (structure and arcs).
    """
    print(f"Computing metrics for {name} ...")

    # Fitness (token-based replay)
    fitness_res = replay_fitness.apply(
        log_for_metrics, net, im, fm,
        variant=replay_fitness.Variants.TOKEN_BASED
    )
    fitness = fitness_res["log_fitness"]

    # Precision (ETConformance variant from pm4py)
    precision = precision_algorithm.apply(
        log_for_metrics, net, im, fm,
        variant=precision_algorithm.Variants.ETCONFORMANCE_TOKEN
    )

    # Generalization
    generalization = generalization_evaluator.apply(
        log_for_metrics, net, im, fm
    )

    # Simplicity – two custom metrics
    n_places = len(net.places)
    n_transitions = len([t for t in net.transitions if t.label is not None])
    n_arcs = len(net.arcs)

    # Structural simplicity: fewer places + transitions = simpler model
    simple_struct = 1.0 / (1.0 + n_places + n_transitions)

    # Arc-based simplicity: fewer arcs per node = simpler
    n_nodes = max(1, n_places + n_transitions)
    avg_arcs_per_node = n_arcs / n_nodes
    simple_arcs = 1.0 / (1.0 + avg_arcs_per_node)

    print(f"----- {name} -----")
    print(f"Fitness (log)      : {fitness:.3f}")
    print(f"Precision (ETConf) : {precision:.3f}")
    print(f"Generalization     : {generalization:.3f}")
    print(f"Simplicity_struct  : {simple_struct:.3f}")
    print(f"Simplicity_arcs    : {simple_arcs:.3f}")
    print(
        f"#places={n_places}, "
        f"#transitions={n_transitions}, "
        f"#arcs={n_arcs}"
    )
    print()

    return {
        "fitness": fitness,
        "precision": precision,
        "generalization": generalization,
        "simplicity_struct": simple_struct,
        "simplicity_arcs": simple_arcs,
        "n_places": n_places,
        "n_transitions": n_transitions,
        "n_arcs": n_arcs,
    }


def discover_process_models(reduced_log):
    """
    Discover two Petri nets on the reduced log:
    - one with the Inductive Miner
    - one with the Heuristics Miner

    Save the visualizations and compute quality metrics for both.
    """
    print("=== PROCESS DISCOVERY ON FILTERED LOG ===")

    case_id_key = CASE_COL
    activity_key = ACT_COL
    timestamp_key = TIMESTAMP_COL

    # --- Inductive Miner ---
    print("Discovering Petri net with Inductive Miner...")
    net_im, im_im, fm_im = pm4py.discover_petri_net_inductive(
        reduced_log,
        case_id_key=case_id_key,
        activity_key=activity_key,
        timestamp_key=timestamp_key
    )
    path_im = os.path.join(FIGURES_DIR, "petri_inductive.png")
    pm4py.save_vis_petri_net(net_im, im_im, fm_im, path_im)
    print(f"Saved {path_im}")
    print()

    # --- Heuristics Miner ---
    print("Discovering Petri net with Heuristics Miner...")
    net_hm, im_hm, fm_hm = pm4py.discover_petri_net_heuristics(
        reduced_log,
        case_id_key=case_id_key,
        activity_key=activity_key,
        timestamp_key=timestamp_key
    )
    path_hm = os.path.join(FIGURES_DIR, "petri_heuristics.png")
    pm4py.save_vis_petri_net(net_hm, im_hm, fm_hm, path_hm)
    print(f"Saved {path_hm}")
    print()

    # --- Metrics on the same reduced log ---
    print("Computing quality metrics on filtered log...")
    metrics_im = compute_all_metrics(
        reduced_log, net_im, im_im, fm_im, name="Inductive Miner"
    )
    metrics_hm = compute_all_metrics(
        reduced_log, net_hm, im_hm, fm_hm, name="Heuristics Miner"
    )
    print("Done with process discovery.\n")

    return (net_im, im_im, fm_im, metrics_im), (net_hm, im_hm, fm_hm, metrics_hm)


# 3) BPMN CONFORMANCE (hand-crafted model)
# This last part is where I bring everything together:
# after looking at the data and letting the miners propose models,
# I drew my own BPMN model by hand.
#
# How did I design it?
# - I looked at the most frequent variants.
# - I noticed the same patterns coming back again and again
#   (e.g. A_Create Application → A_Submitted → W_Complete application,
#        O_Create Offer → O_Created → O_Sent (mail and online), etc.).
# - I used these repeating chunks as the “spine” of the BPMN:
#   they define the main path and the main branches.
# The code here does not build the BPMN automatically (that was done
# in signavio), but it:
# - loads the BPMN file,
# - converts it to a Petri net,
# - filters the log so that it matches the activities in the BPMN,
# - and then measures how well this hand-crafted model fits the data
#   (fitness, precision, generalization, simplicity).


def calculate_bpmn_conformance(log_full):
    """
    Load the BPMN model, convert it to a Petri net, and compute
    conformance metrics with respect to a filtered version of the log.

    Filtering steps:
    - keep only 'complete' lifecycle events
    - keep only activities that actually appear in the BPMN model
    """
    print("=== BPMN CONFORMANCE ANALYSIS ===")

    # --- 1) Load BPMN model and convert to Petri net ---
    print(f"Loading BPMN model from {BPMN_PATH} ...")
    bpmn = bpmn_importer.apply(BPMN_PATH)
    net, im, fm = bpmn_converter.apply(bpmn)
    print("BPMN successfully converted to Petri net.")
    print()

    # --- 2) Convert the full log to DataFrame ---
    df = pm4py.convert_to_dataframe(log_full)
    print("Traces (cases) in the original log:", df[CASE_COL].nunique())
    print("Total events in original log:", len(df))

    # --- 3) Keep only 'complete' lifecycle events ---
    df = df[df[LIFECYCLE_COL] == "complete"].copy()
    print("Events after filtering to 'complete' only:", len(df))

    # --- 4) Keep only activities present in the BPMN model ---
    # This set should match the activity labels used in the BPMN diagram
    bpmn_acts = {
        "A_Create Application", "A_Submitted", "W_Handle leads",
        "W_Complete application", "W_Validate application",
        "A_Accepted", "A_Denied", "A_Pending", "A_Cancelled",
        "O_Create Offer", "O_Created", "O_Sent (mail and online)",
        "O_Accepted", "O_Refused", "A_Concept", "A_Complete",
    }
    df = df[df[ACT_COL].isin(bpmn_acts)].copy()
    print("Events after keeping only BPMN activities:", len(df))

    # --- 5) Convert the filtered DataFrame back to an event log ---
    log_filtered = pm4py.convert_to_event_log(df)
    print("Traces (cases) after filtering:", len(log_filtered))

    if len(log_filtered) == 0:
        raise RuntimeError("Filtered log is empty -> check the filters and activity names!")

    # --- 6) Calculate conformance metrics ---

    def _calculate_conformance_metrics(log_filtered, net, im, fm):
        """
        Compute fitness, precision, generalization and simplicity
        for the BPMN-based Petri net.
        """
        # Token replay: one fitness score per trace
        replayed = token_replay.apply(log_filtered, net, im, fm)
        fitness = sum(t["trace_fitness"] for t in replayed) / len(replayed)

        # Precision and generalization
        precision = precision_eval.apply(log_filtered, net, im, fm)
        generalization = gen_eval.apply(log_filtered, net, im, fm)

        # Simplicity from pm4py
        simplicity = simp_eval.apply(net)

        return fitness, precision, generalization, simplicity

    fitness, precision, generalization, simplicity = _calculate_conformance_metrics(
        log_filtered, net, im, fm
    )

    print("\n=== RESULTS (BPMN model) ===")
    print("Fitness       :", round(fitness, 3))
    print("Precision     :", round(precision, 3))
    print("Generalization:", round(generalization, 3))
    print("Simplicity    :", round(simplicity, 3))
    print()

    return {
        "fitness": fitness,
        "precision": precision,
        "generalization": generalization,
        "simplicity": simplicity,
    }

# MAIN ENTRY POINT


def main():
    # 1) Load the full event log once and reuse it everywhere
    print("Loading full event log...")
    log_full = pm4py.read_xes(LOG_PATH)
    print("Done. Number of traces in full log:", len(log_full))
    print()

    # 2) Basic analysis + descriptive figures
    basic_analysis(log_full)

    # 3) Reduce log to top variants and discover process models
    reduced_log = reduce_log_to_top_variants(log_full, n_variants=300)
    (net_im, im_im, fm_im, metrics_im), (net_hm, im_hm, fm_hm, metrics_hm) = \
        discover_process_models(reduced_log)

    # 4) Conformance analysis of the BPMN model
    bpmn_metrics = calculate_bpmn_conformance(log_full)

    # 5) Very short high-level summary printed at the end
    print("=== SUMMARY (short) ===")
    print("Inductive Miner fitness :", f"{metrics_im['fitness']:.3f}")
    print("Heuristics Miner fitness:", f"{metrics_hm['fitness']:.3f}")
    print("BPMN model fitness      :", f"{bpmn_metrics['fitness']:.3f}")
    print("Check figures/ and console for full details.\n")


if __name__ == "__main__":
    main()
