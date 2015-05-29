# Script to evaluate a result set of images
# Takes as argument the training folder and results_folder
# Outputs to results path tp, tn, fp, fn folders

import os
import shutil
import sys

if len(sys.argv) != 3:
    print("usage: python evaluate.py <training_folder> <results_folder>")
    print("<training_folder> contains 'has_signs', 'no_signs'")
    print("<results_folder> contains 'labelled_has_signs'")
    sys.exit(0)

training_folder = sys.argv[1]
results_folder = sys.argv[2]

# paths to folders containing gold standard training images
has_signs_folder = os.path.join(training_folder, 'has_signs')
no_signs_folder = os.path.join(training_folder, 'no_signs')

labelled_as_has_signs_folder = os.path.join(
    results_folder, 'labelled_as_has_signs'
)

has_signs_set = set(os.listdir(has_signs_folder))
no_signs_set = set(os.listdir(no_signs_folder))
all_signs_set = has_signs_set.union(no_signs_set)

labelled_as_has_signs_set = set(os.listdir(labelled_as_has_signs_folder))
labelled_as_no_signs_set = all_signs_set.difference(labelled_as_has_signs_set)

true_positives = has_signs_set.intersection(labelled_as_has_signs_set)
false_positives = no_signs_set.intersection(labelled_as_has_signs_set)
true_negatives = labelled_as_no_signs_set.intersection(no_signs_set)
false_negatives = labelled_as_no_signs_set.intersection(has_signs_set)

print("tp:", len(true_positives))
print("tn:", len(true_negatives))
print("fp:", len(false_positives))
print("fn:", len(false_negatives))

tp_folder = os.path.join(results_folder, 'tp')
fp_folder = os.path.join(results_folder, 'fp')
fn_folder = os.path.join(results_folder, 'fn')

for filename in true_positives:
    tp_in_path = os.path.join(labelled_as_has_signs_folder, filename)
    tp_out_path = os.path.join(tp_folder, filename)
    if os.path.exists(tp_in_path):
        shutil.copyfile(tp_in_path, tp_out_path)
    else:
        print("failed to find true positive image:", tp_in_path)

for filename in false_positives:
    fp_in_path = os.path.join(labelled_as_has_signs_folder, filename)
    fp_out_path = os.path.join(fp_folder, filename)
    if os.path.exists(fp_in_path):
        shutil.copyfile(fp_in_path, fp_out_path)
    else:
        print("failed to find false positive image:", fp_in_path)

# NOTE these don't have lebels... because they are negatives
# Source image from the training set - has signs
for filename in false_negatives:
    fn_in_path = os.path.join(has_signs_folder, filename)
    fn_out_path = os.path.join(fn_folder, filename)
    if os.path.exists(fn_in_path):
        shutil.copyfile(fn_in_path, fn_out_path)
    else:
        print("failed to find false negative image:", fn_in_path)
