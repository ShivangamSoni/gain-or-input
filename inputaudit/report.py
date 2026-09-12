"""Markdown rendering of data-tier results."""


def _f(x, nd=3):
    return "--" if x is None else f"{x:.{nd}f}"


def data_report(r: dict) -> str:
    m = r["meta"]
    L = [f"# Input-availability audit -- data tier\n",
         f"{m['n']} rows, {len(m['classes'])} classes, {m['n_test']} test rows; text fields: "
         f"{', '.join(m['text_fields'])}; source column: {m['source'] or '--'}\n",
         "| check | verdict | finding |", "|---|---|---|"]
    d1 = r["D1"]
    L.append(f"| D1 provenance | {d1['verdict']} | " + "; ".join(
        f"{f}: {v['why']}" for f, v in d1["fields"].items()) + " |")
    d2 = r["D2"]
    fl = [f"`{n}` -> {x['top_class']} (precision {x['precision']:.2f}, {x['support']} rows)"
          for n, x in d2["relations"].items() if x["verdict"] == "FLAG"]
    L.append(f"| D2 field relations | {d2['verdict']} | {'; '.join(fl) or 'no label-revealing relation'} |")
    for f, x in r["D3"].items():
        L.append(f"| D3 surface screen ({f}) | {x['verdict']} | shape-only macro-F1 {_f(x['shape_f1'])}, "
                 f"lift {x['lift']:+.3f}, share {_f(x['share'], 2)} {x['level']} |")
    if r.get("D4"):
        for f, x in r["D4"].items():
            if "source_auc_from_shape" in x:
                L.append(f"| D4 source leak ({f}) | {x['verdict']} | source from shape AUC "
                         f"{x['source_auc_from_shape']:.3f}; source-label V {x['source_label_cramers_v']:.2f} |")
    if r.get("D6"):
        x = r["D6"]
        leak = ("" if x["leaky_share"] is None else
                f"; lookup answers {x['leaky_share']:.3f} (chance {x['leaky_null']:.3f}, "
                f"excess {x['leaky_excess']:+.3f})")
        L.append(f"| D6 near-duplicate leakage | {x['verdict']} | {x['dup_share']:.3f} of test items "
                 f"duplicate a train/val item (exact text {x['exact_text_share']:.3f}){leak} |")
    if r.get("D5"):
        x = r["D5"]
        mx = x["max_single_modality"]
        L.append(f"| D5 modality check | {x['verdict']} | best single modality {mx['modality']} "
                 f"({mx['encoder']}) {mx['macro_f1']:.3f}; image: {x['claims']['image']}; "
                 f"text: {x['claims']['text']} |")
    L.append("\n## D1 per-source profile\n")
    L.append("| field | source | n | empty | mean chars | upper-case ratio |")
    L.append("|---|---|---|---|---|---|")
    for f, g in d1["profile"].items():
        for s, p in g.items():
            L.append(f"| {f} | {s} | {p['n']} | {p['empty']:.3f} | {p['mean_chars']:.1f} | "
                     f"{p['upper_ratio']:.3f} |")
    L.append("\n## D2 relations\n")
    L.append("| relation | rows | top class | precision | recall | verdict |")
    L.append("|---|---|---|---|---|---|")
    for n, x in d2["relations"].items():
        L.append(f"| {n} | {x['support']} | {x['top_class']} | {x['precision']:.3f} | "
                 f"{x['recall']:.3f} | {x['verdict']} |")
    if r.get("D5"):
        L.append("\n## D5 probes (test macro-F1)\n")
        L.append("| encoder | image | text | both | value of image | value of text |")
        L.append("|---|---|---|---|---|---|")
        for n, e in r["D5"]["encoders"].items():
            mm = e["macro_f1"]
            L.append(f"| {n} | {mm['image']:.3f} | {mm['text']:.3f} | {mm['both']:.3f} | "
                     f"{e['value_of_image']:+.3f} | {e['value_of_text']:+.3f} |")
    return "\n".join(L) + "\n"
