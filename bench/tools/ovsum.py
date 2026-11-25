# Compact dump of an OpenVINO IR .xml: one line per layer with inputs, key attributes and output shape/precision.
import sys, xml.etree.ElementTree as ET, collections
def summarize(path, skip_const_data=True):
    r = ET.parse(path).getroot()
    layers = {l.get("id"): l for l in r.find("layers")}
    edges = collections.defaultdict(dict)  # to_layer -> {to_port: (from_layer, from_port)}
    for e in r.find("edges"):
        edges[e.get("to-layer")][int(e.get("to-port"))] = (e.get("from-layer"), e.get("from-port"))
    lines, cnt = [], collections.Counter()
    for lid, l in layers.items():
        t = l.get("type"); cnt[t] += 1
        d = dict(l.find("data").attrib) if l.find("data") is not None else {}
        for k in ("offset", "size", "version"): d.pop(k, None)
        outs = []
        for p in (l.find("output") or []):
            outs.append(p.get("precision") + "[" + ",".join(x.text for x in p) + "]")
        ins = ",".join(f"{edges[lid][i][0]}" + (f".{edges[lid][i][1]}" if layers[edges[lid][i][0]].find('output') is not None and len(layers[edges[lid][i][0]].find('output'))>1 else "") for i in sorted(edges[lid])) if lid in edges else ""
        lines.append(f"{lid:>4} {t:<14} <- [{ins}] {' '.join(f'{k}={v}' for k, v in d.items())} => {' '.join(outs)}  ({l.get('name')})")
    return lines, cnt
if __name__ == "__main__":
    lines, cnt = summarize(sys.argv[1])
    print("\n".join(lines)); print("\nop counts:", dict(cnt.most_common()))
