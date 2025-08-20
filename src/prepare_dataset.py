import jsonlines, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    ap.add_argument("--outfile", required=True)
    args = ap.parse_args()

    with jsonlines.open(args.infile) as rdr, jsonlines.open(args.outfile,"w") as wr:
        for ex in rdr:
            # Ensure instruction/input/output keys exist
            out = {
                "task": ex.get("task","unknown"),
                "instruction": ex.get("instruction",""),
                "input": ex.get("input",""),
                "output": ex.get("output","")
            }
            wr.write(out)

if __name__=="__main__":
    main()
