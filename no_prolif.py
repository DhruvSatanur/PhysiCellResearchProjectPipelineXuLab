import xml.etree.ElementTree as ET, shutil
f = "config/PhysiCell_settings.xml"
shutil.copy(f, f + ".backup")
tree = ET.parse(f); root = tree.getroot()
n = 0
for cd in root.iter("cell_definition"):
    ph = cd.find("phenotype")
    cyc = ph.find("cycle") if ph is not None else None
    if cyc is None:
        continue
    # this file uses phase DURATIONS -> set them huge so cells never divide
    for dur in cyc.iter("duration"):
        dur.text = "1e12"; n += 1
    # also zero any transition rates, in case some types use those instead
    for rate in cyc.iter("rate"):
        rate.text = "0"; n += 1
tree.write(f)
print(f"Set {n} cycle duration(s)/rate(s) to stop division -> {f}")