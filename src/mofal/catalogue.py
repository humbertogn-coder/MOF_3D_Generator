"""Design of the v3 candidate catalogue (chemically audited building blocks).

Every building block below was audited against the PORMAKE database
(``scripts/00_audit_pormake_bbs.py``):

* nodes are native PORMAKE carboxylate clusters, charge balanced for the stated
  metal oxidation state (no stripped hydrogens, no non-carboxylate nodes), and the
  metal label is taken from the building block itself;
* edges are carboxylate-free organic spacers (the carboxylate C/O atoms belong to
  the node building block), so the assembled linker is exactly the named
  dicarboxylate (e.g. N16 + E14 in pcu gives Zn4O(BDC)3 = C24H12O13Zn4, MOF-5).

Metal and topology effects are compared *within* a node family, so that the
metal axis is not confounded with a change of cluster type.
"""
from __future__ import annotations

# ---------------------------------------------------------------- nodes
# family -> list of (node_id, metal, ox_state, description)
NODE_FAMILIES = {
    # 6-c trinuclear M3(CO2)6 clusters
    "M3(CO2)6": {
        "topologies": ["pcu"],
        "nodes": [
            ("N548", "Mn", 2, "Mn3(CO2)6"),
            ("N334", "Fe", 2, "Fe3(CO2)6"),
            ("N491", "Co", 2, "Co3(CO2)6"),
            ("N382", "Zn", 2, "Zn3(CO2)6"),
            ("N406", "Mg", 2, "Mg3(CO2)6 (closed-shell s-block reference)"),
        ],
    },
    # 6-c Zn4O(CO2)6 (IRMOF reference series)
    "Zn4O(CO2)6": {
        "topologies": ["pcu"],
        "nodes": [("N16", "Zn", 2, "Zn4O(CO2)6, IRMOF-1 node")],
    },
    # 4-c square-planar M2(CO2)4 paddlewheels
    "M2(CO2)4-paddlewheel": {
        "topologies": ["nbo", "cds"],
        "nodes": [
            ("N288", "Mn", 2, "Mn2(CO2)4 paddlewheel"),
            ("N355", "Fe", 2, "Fe2(CO2)4 paddlewheel"),
            ("N317", "Co", 2, "Co2(CO2)4 paddlewheel"),
            ("N409", "Cu", 2, "Cu2(CO2)4 paddlewheel"),
            ("N153", "Zn", 2, "Zn2(CO2)4 paddlewheel"),
        ],
    },
    # 4-c tetrahedral M2(CO2)4 clusters
    "M2(CO2)4-tetrahedral": {
        "topologies": ["dia"],
        "nodes": [
            ("N398", "Co", 2, "Co2(CO2)4, tetrahedral connectivity"),
            ("N649", "Ni", 2, "Ni2(CO2)4, tetrahedral connectivity"),
            ("N256", "Zn", 2, "Zn2(CO2)4, tetrahedral connectivity"),
        ],
    },
    # 8-c M4(CO2)8 clusters
    "M4(CO2)8": {
        "topologies": ["bcu"],
        "nodes": [
            ("N350", "Fe", 2, "Fe4(CO2)8"),
            ("N35", "Co", 2, "Co4(CO2)8"),
        ],
    },
}

# ---------------------------------------------------------------- edges
# linker label -> (edge_id, family, n_rings, substituent, assembled dicarboxylate)
EDGES = {
    "short-H":      ("E14",  "BDC",  1, "H",      "1,4-benzenedicarboxylate (BDC)"),
    "short-NH2":    ("E72",  "BDC",  1, "NH2",    "2-amino-BDC"),
    "short-F4":     ("E228", "BDC",  1, "F4",     "tetrafluoro-BDC"),
    "short-Nring1": ("E166", "BDC",  1, "Nring1", "pyridine-2,5-dicarboxylate"),
    "long-H":       ("E34",  "BPDC", 2, "H",      "4,4'-biphenyldicarboxylate (BPDC)"),
    "long-F8":      ("E145", "BPDC", 2, "F8",     "octafluoro-BPDC"),
    "long-Nring1":  ("E107", "BPDC", 2, "Nring1", "phenylpyridine dicarboxylate"),
    "long-Nring2":  ("E35",  "BPDC", 2, "Nring2", "bipyridine dicarboxylate"),
}

OPEN_SHELL_3D = {"Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu"}
D_ELECTRONS_2PLUS = {"Mn": 5, "Fe": 6, "Co": 7, "Ni": 8, "Cu": 9, "Zn": 10, "Mg": 0}


def enumerate_candidates():
    """Yield one dict per (family, node, topology, linker) combination."""
    for fam, spec in NODE_FAMILIES.items():
        for topo in spec["topologies"]:
            for node_id, metal, ox, ndesc in spec["nodes"]:
                for lab, (edge_id, lfam, nring, subst, ldesc) in EDGES.items():
                    yield {
                        "name": f"{topo}__{metal}-{node_id}__{lab}",
                        "node_family": fam,
                        "topology": topo,
                        "node_id": node_id,
                        "metal": metal,
                        "metal_ox_state": ox,
                        "d_electrons": D_ELECTRONS_2PLUS.get(metal) if ox == 2 else None,
                        "open_shell": metal in OPEN_SHELL_3D,
                        "node_desc": ndesc,
                        "linker_label": lab,
                        "edge_id": edge_id,
                        "linker_family": lfam,
                        "n_rings": nring,
                        "substituent": subst,
                        "linker_desc": ldesc,
                    }
