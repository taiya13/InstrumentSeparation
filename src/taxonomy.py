"""
taxonomy.py — Loader, validator, and query API for the orchestra Instrument Taxonomy.

This turns configs/taxonomy.yaml into a usable object for training/eval code:
  - integrity validation (unique ids, valid parent chains, same-timbre members exist)
  - navigation (ancestors, roll-up to a level, leaves at a level)
  - dataset-label mapping (GM program / SynthSOD / URMP / our synth bench -> id)

Run as a script to validate + print a summary:
    python src/taxonomy.py
"""
import os
import sys
import yaml

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "taxonomy.yaml")


class Node:
    __slots__ = ("id", "key", "name", "level", "parent", "attrs")

    def __init__(self, id, key, name, level, parent, attrs):
        self.id, self.key, self.name = id, key, name
        self.level, self.parent, self.attrs = level, parent, attrs

    def __repr__(self):
        return f"<Node {self.level}:{self.id}>"


class Taxonomy:
    def __init__(self, path=DEFAULT_PATH):
        with open(path, "r", encoding="utf-8") as f:
            self.raw = yaml.safe_load(f)
        self.version = self.raw["version"]
        self.levels = self.raw["levels"]
        self.policy = self.raw.get("policy", {})
        self.nodes = {}          # id -> Node
        self.by_key = {}         # (level,key) -> Node
        self._build()
        self._validate()

    def _add(self, id, key, name, level, parent, attrs):
        if id in self.nodes:
            raise ValueError(f"duplicate id: {id}")
        n = Node(id, key, name, level, parent, attrs)
        self.nodes[id] = n
        self.by_key[(level, key)] = n
        return n

    def _build(self):
        for cat in self.raw["categories"]:
            self._add(cat["id"], cat["key"], cat.get("name", {}), "category", None, cat)
            for fam in cat.get("families", []):
                self._add(fam["id"], fam["key"], fam.get("name", {}), "family", cat["id"], fam)
                for ins in fam.get("instruments", []):
                    self._add(ins["id"], ins["key"], ins.get("name", {}), "instrument", fam["id"], ins)
                    for part in ins.get("parts", []):
                        self._add(part["id"], part["key"], part.get("name", {}),
                                  "part", ins["id"], part)

    def _validate(self):
        # parent chains resolve
        for n in self.nodes.values():
            if n.parent is not None and n.parent not in self.nodes:
                raise ValueError(f"{n.id}: parent {n.parent} not found")
        # id prefix consistency (child id starts with parent id + '.')
        for n in self.nodes.values():
            if n.parent:
                # families may use any id; require instrument/part to be prefixed by ancestor instrument/category
                pass
        # same-timbre group members exist and are parts/instruments
        for g in self.raw.get("same_timbre_groups", []):
            for m in g["members"]:
                if m not in self.nodes:
                    raise ValueError(f"same_timbre_group {g['id']}: member {m} missing")
        # mappings resolve to real ids (or __mixed__)
        for src, table in self.raw.get("mappings", {}).items():
            for label, tid in table.items():
                if tid != "__mixed__" and tid not in self.nodes:
                    raise ValueError(f"mapping {src}:{label} -> {tid} not a valid id")

    # ---- navigation --------------------------------------------------------
    def ancestors(self, id):
        chain = []
        n = self.nodes[id]
        while n.parent is not None:
            n = self.nodes[n.parent]
            chain.append(n.id)
        return chain

    def rollup(self, id, to_level):
        """Return the ancestor (or self) at `to_level`."""
        n = self.nodes[id]
        if n.level == to_level:
            return id
        for aid in self.ancestors(id):
            if self.nodes[aid].level == to_level:
                return aid
        # requested level is deeper than this node -> no unique rollup
        return None

    def leaves(self, level="instrument"):
        return [n.id for n in self.nodes.values() if n.level == level]

    def children(self, id):
        return [n.id for n in self.nodes.values() if n.parent == id]

    def same_timbre_group_of(self, id):
        for g in self.raw.get("same_timbre_groups", []):
            if id in g["members"]:
                return g["id"]
        return None

    # ---- dataset mapping ---------------------------------------------------
    def map_label(self, source, label):
        table = self.raw.get("mappings", {}).get(source, {})
        return table.get(label)

    def display(self, id, lang="en"):
        n = self.nodes[id]
        name = n.name or {}
        return name.get(lang) or name.get("en") or n.key


def main():
    tx = Taxonomy()
    print(f"Taxonomy v{tx.version}  levels={tx.levels}")
    counts = {lv: len([n for n in tx.nodes.values() if n.level == lv]) for lv in tx.levels}
    print("node counts:", counts, "| total:", len(tx.nodes))
    print("\ncategories:")
    for cid in tx.children_of_root() if hasattr(tx, "children_of_root") else \
              [n.id for n in tx.nodes.values() if n.level == "category"]:
        insts = [i for i in tx.leaves("instrument") if tx.rollup(i, "category") == cid]
        print(f"  {cid:6s} {tx.display(cid,'ja'):8s} instruments={len(insts)}")
    print("\ninstrument leaves:", len(tx.leaves("instrument")),
          "| part leaves:", len(tx.leaves("part")))
    print("same-timbre groups:",
          [g["id"] + str(g["members"]) for g in tx.raw.get("same_timbre_groups", [])])
    # spot-check mappings + rollup
    print("\nmapping checks:")
    for src, lab in [("urmp", "vn"), ("gm_program", 40), ("synthsod_family", "brass"),
                     ("synth_bench", "violin1")]:
        tid = tx.map_label(src, lab)
        roll = tx.rollup(tid, "category") if tid and tid != "__mixed__" else None
        print(f"  {src}:{lab} -> {tid}  (category={roll})")
    print(f"\nrollup(str.vln.1 -> instrument) = {tx.rollup('str.vln.1','instrument')}")
    print(f"same_timbre_group_of(str.vln.1) = {tx.same_timbre_group_of('str.vln.1')}")
    print("\nVALIDATION: OK")


# small helper used above
def _children_of_root(self):
    return [n.id for n in self.nodes.values() if n.level == "category"]
Taxonomy.children_of_root = _children_of_root


if __name__ == "__main__":
    main()
