import os, csv, json

class CSVLogger:
    def __init__(self, out_csv):
        self.out_csv = out_csv
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        if not os.path.exists(out_csv):
            with open(out_csv, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['round','client','labeled_count','ve_median','bal_acc_rest','f1_macro_rest','auc_ovr_rest','pseudo_acc_meta','orig_train_n','aug_train_n','committee_local','committee_global'])

    def log(self, round_id, client, labeled_count, ve_median, bal_acc_rest, f1_macro_rest, auc_ovr_rest, pseudo_acc_meta, orig_n, aug_n, com_local, com_global):
        with open(self.out_csv, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow([round_id, client, labeled_count, ve_median, bal_acc_rest, f1_macro_rest, auc_ovr_rest, pseudo_acc_meta, orig_n, aug_n, com_local, com_global])

class JSONLLogger:
    def __init__(self, out_jsonl):
        self.out_jsonl = out_jsonl
        os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    def log(self, obj):
        with open(self.out_jsonl, 'a') as f:
            f.write(json.dumps(obj) + "\n")
