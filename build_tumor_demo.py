#!/usr/bin/env python3
"""
build_tumor_demo.py -- construct the Heiland et al. 2024 tumor model config
directly from XML, following the same discipline as no_prolif.py: edit the
file, then verify by reading it back. Never open the result in Studio before
a run; Studio rewrites cycle parameters on save.

Usage:
    py -3.12 build_tumor_demo.py --template config/template.xml \
                                 --spec tumor_demo_spec.json \
                                 --out config/tumor_demo.xml

Then verify independently:
    findstr /C:"phase_transition_rate" config\\tumor_demo.xml
    findstr /C:"cell_rules.csv" config\\tumor_demo.xml
"""

import argparse, json, sys, copy
from pathlib import Path
import xml.etree.ElementTree as ET


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def setval(parent, path, value):
    """Set text of parent/path, creating the node if absent. Returns the node."""
    node = parent.find(path)
    if node is None:
        cur = parent
        for tag in path.split("/"):
            nxt = cur.find(tag)
            if nxt is None:
                nxt = ET.SubElement(cur, tag)
            cur = nxt
        node = cur
    node.text = str(value)
    return node


# ---------------------------------------------------------------- domain/time

def apply_domain(root, d):
    dom = root.find("domain") or die("no <domain> in template")
    for k in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "dx", "dy", "dz"):
        setval(dom, k, d[k])
    setval(dom, "use_2D", "true" if d["use_2D"] else "false")


def apply_overall(root, o):
    ov = root.find("overall") or die("no <overall> in template")
    setval(ov, "max_time", o["max_time"])
    setval(ov, "dt_diffusion", o["dt_diffusion"])
    setval(ov, "dt_mechanics", o["dt_mechanics"])
    setval(ov, "dt_phenotype", o["dt_phenotype"])

    par = root.find("parallel")
    if par is not None:
        setval(par, "omp_num_threads", o["omp_num_threads"])

    save = root.find("save")
    if save is not None:
        setval(save, "folder", o["output_folder"])
        setval(save, "full_data/interval", o["full_interval"])
        setval(save, "SVG/interval", o["svg_interval"])


# ---------------------------------------------------------------- substrates

def apply_substrates(root, subs, opts):
    me = root.find("microenvironment_setup") or die("no <microenvironment_setup>")

    # Remove template placeholder substrates; we define ours explicitly.
    for v in me.findall("variable"):
        me.remove(v)

    for idx, (name, s) in enumerate(subs.items()):
        if name.startswith("_"):
            continue
        var = ET.Element("variable", {"name": name, "units": "dimensionless", "ID": str(idx)})
        phys = ET.SubElement(var, "physical_parameter_set")
        dc = ET.SubElement(phys, "diffusion_coefficient", {"units": "micron^2/min"})
        dc.text = str(s["diffusion_coefficient"])
        dr = ET.SubElement(phys, "decay_rate", {"units": "1/min"})
        dr.text = str(s["decay_rate"])
        ic = ET.SubElement(var, "initial_condition", {"units": "mmHg"})
        ic.text = str(s["initial_condition"])
        dbc = ET.SubElement(var, "Dirichlet_boundary_condition",
                            {"units": "mmHg",
                             "enabled": "true" if s["dirichlet_enabled"] else "false"})
        dbc.text = str(s["dirichlet_value"])

        if s.get("boundaries"):
            opt = ET.SubElement(var, "Dirichlet_options")
            for bname, b in s["boundaries"].items():
                e = ET.SubElement(opt, f"boundary_value",
                                  {"ID": bname,
                                   "enabled": "true" if b["enabled"] else "false"})
                e.text = str(b["value"])
        me.append(var)

    o = me.find("options")
    if o is None:
        o = ET.SubElement(me, "options")
    setval(o, "calculate_gradients", str(opts["calculate_gradients"]).lower())
    setval(o, "track_internalized_substrates_in_each_agent",
           str(opts["track_internalized_substrates_in_each_agent"]).lower())


# ---------------------------------------------------------------- cell types

def find_def(root, name):
    for cd in root.find("cell_definitions").findall("cell_definition"):
        if cd.get("name") == name:
            return cd
    return None


def apply_cycle(cd, cyc):
    """Two modes, and they are mutually exclusive.

    'durations' (phase 0 duration 1440) is the paper's base model, used before
    the oxygen rule exists -- stages 0 and 1.

    'transition_rates' with rate 0 is what the oxygen rule requires from stage 2
    onward: the rule supplies cycle entry. Leaving phase_durations in place here
    would silently fight the rule, which is the mirror image of the no_prolif.py
    trap. Whichever mode is chosen, the other is stripped.
    """
    node = cd.find("phenotype/cycle")
    if node is None:
        die("cell_definition has no phenotype/cycle")


    node.set("code", "5")
    node.set("name", "live")
    mode = cyc.get("mode", "transition_rates")

    for tag in ("phase_durations", "phase_transition_rates"):
        for old in node.findall(tag):
            node.remove(old)

    if mode == "durations":
        pdn = ET.SubElement(node, "phase_durations", {"units": "min"})
        d = ET.SubElement(pdn, "duration", {"index": "0", "fixed_duration": "false"})
        d.text = str(cyc["phase_0_duration"])
    elif mode == "transition_rates":
        ptr = ET.SubElement(node, "phase_transition_rates", {"units": "1/min"})
        r = ET.SubElement(ptr, "rate",
                          {"start_index": "0", "end_index": "0", "fixed_duration": "false"})
        r.text = str(cyc["phase_0_transition_rate"])
    else:
        die(f"unknown cycle mode '{mode}' (expected durations or transition_rates)")


def apply_death(cd, death):
    for dm in cd.findall("phenotype/death/model"):
        code = dm.get("code")
        if code == "100" and "apoptosis_rate" in death:
            setval(dm, "death_rate", death["apoptosis_rate"])
        elif code == "101" and "necrosis_rate" in death:
            setval(dm, "death_rate", death["necrosis_rate"])


def apply_secretion(cd, sec):
    sroot = cd.find("phenotype/secretion")
    if sroot is None:
        sroot = ET.SubElement(cd.find("phenotype"), "secretion")
    for sub, vals in sec.items():
        if sub.startswith("_"):
            continue
        node = None
        for s in sroot.findall("substrate"):
            if s.get("name") == sub:
                node = s
                break
        if node is None:
            node = ET.SubElement(sroot, "substrate", {"name": sub})
        for k, xml_tag in (("secretion_rate", "secretion_rate"),
                           ("secretion_target", "secretion_target"),
                           ("uptake_rate", "uptake_rate"),
                           ("net_export_rate", "net_export_rate")):
            if k in vals:
                setval(node, xml_tag, vals[k])


def apply_mechanics(cd, mech):
    m = cd.find("phenotype/mechanics")
    for k, v in mech.items():
        if not k.startswith("_"):
            setval(m, k, v)


def apply_motility(cd, mot):
    m = cd.find("phenotype/motility")
    if "speed" in mot: setval(m, "speed", mot["speed"])
    if "persistence_time" in mot: setval(m, "persistence_time", mot["persistence_time"])
    if "migration_bias" in mot: setval(m, "migration_bias", mot["migration_bias"])
    opt = m.find("options")
    if opt is None:
        opt = ET.SubElement(m, "options")
    if "enabled" in mot:
        setval(opt, "enabled", str(mot["enabled"]).lower())
    if "use_2D" in mot:
        setval(opt, "use_2D", str(mot["use_2D"]).lower())
    if "chemotaxis" in mot:
        c = mot["chemotaxis"]
        ch = opt.find("chemotaxis")
        if ch is None:
            ch = ET.SubElement(opt, "chemotaxis")
        setval(ch, "enabled", str(c["enabled"]).lower())
        setval(ch, "substrate", c["substrate"])
        setval(ch, "direction", c["direction"])


def apply_interactions(cd, inter):
    ci = cd.find("phenotype/cell_interactions")
    if ci is None:
        ci = ET.SubElement(cd.find("phenotype"), "cell_interactions")
    if "dead_phagocytosis_rate" in inter:
        setval(ci, "dead_phagocytosis_rate", inter["dead_phagocytosis_rate"])
    if "damage_rate" in inter:
        setval(ci, "damage_rate", inter["damage_rate"])
    if "attack_rates" in inter:
        ar = ci.find("attack_rates")
        if ar is None:
            ar = ET.SubElement(ci, "attack_rates")
        for target, rate in inter["attack_rates"].items():
            node = None
            for r in ar.findall("attack_rate"):
                if r.get("name") == target:
                    node = r
                    break
            if node is None:
                node = ET.SubElement(ar, "attack_rate", {"name": target})
            node.text = str(rate)


CROSS_REFS = {
    # child tag -> (container tag, default value for a newly added entry)
    "cell_adhesion_affinity": ("cell_adhesion_affinities", "1.0"),
    "phagocytosis_rate":      ("live_phagocytosis_rates",  "0.0"),
    "attack_rate":            ("attack_rates",             "0.0"),
    "fusion_rate":            ("fusion_rates",             "0.0"),
    "transformation_rate":    ("transformation_rates",     "0.0"),
}




def sanitize_cross_references(root):
    """Every cell type carries a rate toward every OTHER cell type, keyed by name.

    Two things break those keys:
      1. Renaming the template's 'default' definition leaves its self-references
         pointing at a name that no longer exists. PhysiCell looks the name up,
         finds nothing, and dereferences a null pointer -- an access violation
         with no error message.
      2. Copying a cell type to make a new one (macrophage -> M1 macrophage)
         adds a type that no existing entry mentions. The 'transform to M1
         macrophage' rule then has no rate slot to write into.

    So after all cell types exist, rewrite these blocks to hold exactly one
    entry per defined type, preserving any value already set by name.
    """
    defs = root.find("cell_definitions")
    names = [cd.get("name") for cd in defs.findall("cell_definition")]

    for cd in defs.findall("cell_definition"):
        for child_tag, (container_tag, default) in CROSS_REFS.items():
            for container in cd.iter(container_tag):
                # keep whatever was already set for a still-valid name
                kept, units = {}, None
                for e in list(container.findall(child_tag)):
                    if units is None:
                        units = e.get("units")
                    if e.get("name") in names:
                        kept[e.get("name")] = e.text
                    container.remove(e)
                for n in names:
                    attrs = {"name": n}
                    if units:
                        attrs["units"] = units
                    e = ET.SubElement(container, child_tag, attrs)
                    e.text = kept.get(n, default)


def apply_cell_types(root, spec):
    cds = root.find("cell_definitions") or die("no <cell_definitions>")
    next_id = 0

    for name, c in spec.items():
        if name.startswith("_"):
            continue

        if "_from_template" in c:
            base = find_def(root, c["_from_template"])
            if base is None:
                die(f"template has no cell_definition named '{c['_from_template']}' "
                    f"-- check the template file you passed in")
            base.set("name", name)
            cd = base
        elif "_copy_of" in c:
            src = find_def(root, c["_copy_of"])
            if src is None:
                die(f"cannot copy '{c['_copy_of']}': not defined yet "
                    f"(order matters in the spec)")
            cd = copy.deepcopy(src)
            cd.set("name", name)
            cds.append(cd)
        else:
            die(f"cell type '{name}' has neither _from_template nor _copy_of")

        if "cycle" in c:        apply_cycle(cd, c["cycle"])
        if "death" in c:        apply_death(cd, c["death"])
        if "secretion" in c:    apply_secretion(cd, c["secretion"])
        if "mechanics" in c:    apply_mechanics(cd, c["mechanics"])
        if "motility" in c:     apply_motility(cd, c["motility"])
        if "interactions" in c: apply_interactions(cd, c["interactions"])

    for i, cd in enumerate(cds.findall("cell_definition")):
        cd.set("ID", str(i))

    def sanitize_substrate_references(root):
        """Point every substrate reference at a substrate that actually exists."""
        names = [s.get("name") for s in root.findall("microenvironment_setup/variable")]
        if not names:
            return
        default = names[0]
        for cd in root.findall("cell_definitions/cell_definition"):
            for sec in cd.findall("phenotype/secretion/substrate"):
                if sec.get("name") not in names:
                    sec.set("name", default)
            chem = cd.find("phenotype/motility/options/chemotaxis/substrate")
            if chem is not None and chem.text not in names:
                chem.text = default
            for cs in cd.findall(
                    "phenotype/motility/options/advanced_chemotaxis/chemotactic_sensitivities/chemotactic_sensitivity"):
                if cs.get("substrate") not in names:
                    cs.set("substrate", default)
        plot = root.find("save/SVG/plot_substrate/substrate")
        if plot is not None and plot.text not in names:
            plot.text = default

    sanitize_cross_references(root)
    sanitize_substrate_references(root)


# ---------------------------------------------------------------- ics + rules

def apply_ics(root, ics):
    ic = root.find("initial_conditions")
    if ic is None:
        ic = ET.SubElement(root, "initial_conditions")
    cp = ic.find("cell_positions")
    if cp is None:
        cp = ET.SubElement(ic, "cell_positions", {"type": "csv", "enabled": "true"})
    cp.set("enabled", "true" if ics["enabled"] else "false")
    setval(cp, "folder", ics["folder"])
    setval(cp, "filename", ics["filename"])


def apply_rules(root, rules):
    cr = root.find("cell_rules")
    if cr is None:
        cr = ET.SubElement(root, "cell_rules")
    rs = cr.find("rulesets")
    if rs is None:
        rs = ET.SubElement(cr, "rulesets")
    for r in rs.findall("ruleset"):
        rs.remove(r)
    r = ET.SubElement(rs, "ruleset",
                      {"protocol": "CBHG", "version": "3.0", "format": "CSV",
                       "enabled": "true" if rules["enabled"] else "false"})
    setval(r, "folder", rules["folder"])
    setval(r, "filename", rules["filename"])


def apply_user_params(root, up):
    upr = root.find("user_parameters")
    if upr is None:
        return
    for k, v in up.items():
        if k.startswith("_"):
            continue
        node = upr.find(k)
        if node is None:
            node = ET.SubElement(upr, k, {"type": "int", "units": "none"})
        node.text = str(v)


# ---------------------------------------------------------------- verify

def verify(path, spec):
    """Read the written file back and assert the things that have silently
    broken before. Failing loudly here is the whole point."""
    tree = ET.parse(path)
    root = tree.getroot()
    problems, notes = [], []

    defined = {cd.get("name") for cd in root.find("cell_definitions").findall("cell_definition")}
    wanted = {k for k in spec["cell_types"] if not k.startswith("_")}
    missing = wanted - defined
    if missing:
        problems.append(f"cell types in spec but not in XML: {sorted(missing)}")
    notes.append(f"cell definitions written: {sorted(defined)}")

    subs = {v.get("name") for v in root.find("microenvironment_setup").findall("variable")}
    notes.append(f"substrates written: {sorted(subs)}")

    # every type-to-type rate must name a cell type that exists
    for cd in root.find("cell_definitions").findall("cell_definition"):
        for child_tag in CROSS_REFS:
            for e in cd.iter(child_tag):
                if e.get("name") not in defined:
                    problems.append(
                        f"{cd.get('name')} has <{child_tag} name='{e.get('name')}'> "
                        f"but no such cell type is defined (PhysiCell segfaults on this)")

    # every chemotaxis / secretion substrate must actually exist
    for cd in root.find("cell_definitions").findall("cell_definition"):
        ch = cd.find("phenotype/motility/options/chemotaxis")
        if ch is not None and ch.findtext("enabled") == "true":
            s = ch.findtext("substrate")
            if s not in subs:
                problems.append(f"{cd.get('name')} chemotaxes toward undefined substrate '{s}'")

    want_rules = spec.get("rules", {}).get("enabled", True)

    # The two cycle modes must never coexist, and duration mode is only valid
    # before the oxygen->cycle-entry rule exists (paper stages 0 and 1).
    oxygen_cycle_rule = False
    if want_rules:
        rules_dir = root.findtext("cell_rules/rulesets/ruleset/folder") or "."
        rules_csv = root.findtext("cell_rules/rulesets/ruleset/filename")
        if rules_csv:
            for cand in (Path(rules_dir) / rules_csv,
                         Path(path).parent / rules_csv):
                if cand.exists():
                    oxygen_cycle_rule = any(
                        "oxygen" in ln and "cycle entry" in ln
                        for ln in cand.read_text().splitlines())
                    break

    for cd in root.find("cell_definitions").findall("cell_definition"):
        cyc = cd.find("phenotype/cycle")
        if cyc is None:
            continue
        has_dur = cyc.find("phase_durations") is not None
        has_rate = cyc.find("phase_transition_rates") is not None
        if has_dur and has_rate:
            problems.append(f"{cd.get('name')} has BOTH phase_durations and "
                            f"phase_transition_rates -- they will fight")
        if has_dur and oxygen_cycle_rule:
            problems.append(f"{cd.get('name')} uses phase_durations while an "
                            f"oxygen->cycle entry rule is active; the rule "
                            f"will be silently overridden")

    # Baseline stages legitimately have no rules, so "enabled" is only required
    # when the spec actually asks for a ruleset. The inverse matters too: rules
    # silently active in a stage that should have none would invalidate it.
    rs = root.find("cell_rules/rulesets/ruleset")
    if want_rules:
        if rs is None or rs.get("enabled") != "true":
            problems.append("rules ruleset missing or disabled")
        else:
            notes.append(f"rules: {rs.findtext('folder')}/{rs.findtext('filename')} "
                         f"(protocol {rs.get('protocol')} v{rs.get('version')})")
    else:
        if rs is not None and rs.get("enabled") == "true":
            problems.append("rules are enabled but this stage expects none")
        else:
            notes.append("rules: none (baseline stage, as intended)")

    cp = root.find("initial_conditions/cell_positions")
    if cp is None or cp.get("enabled") != "true":
        problems.append("initial conditions not enabled -- cells.csv will be ignored")

    return problems, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--spec", default="tumor_demo_spec.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    spec = json.load(open(a.spec))
    tree = ET.parse(a.template)
    root = tree.getroot()

    apply_domain(root, spec["domain"])
    apply_overall(root, spec["overall"])
    apply_substrates(root, spec["substrates"], spec["options"])
    apply_cell_types(root, spec["cell_types"])
    apply_ics(root, spec["initial_conditions"])
    apply_rules(root, spec["rules"])
    apply_user_params(root, spec["user_parameters"])

    ET.indent(tree, space="  ")
    tree.write(a.out, encoding="UTF-8", xml_declaration=True)

    problems, notes = verify(a.out, spec)
    print(f"wrote {a.out}")
    for n in notes:
        print("  " + n)
    if problems:
        print("\nVERIFICATION FAILED:")
        for p in problems:
            print("  ! " + p)
        sys.exit(1)
    print("\nverification passed")


if __name__ == "__main__":
    main()
