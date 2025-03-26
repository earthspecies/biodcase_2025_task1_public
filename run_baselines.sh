datadir=$1
for data in zebra_finch aru;
do
for split in val;
do
# Nosync baseline
python baseline_nosync.py --output-dir=nosync_${data}_${split} --inference-dir=${datadir}/${data}/${split}/audio;
python evaluate.py --predictions-fp=nosync_${data}_${split}/predictions.csv --ground-truth-fp=${datadir}/${data}/${split}/annotations.csv;
# Crosscorrelation baseline
python baseline_crosscor.py --output-dir=crosscor_${data}_${split} --inference-dir=${datadir}/${data}/${split}/audio;
python evaluate.py --predictions-fp=crosscor_${data}_${split}/predictions.csv --ground-truth-fp=${datadir}/${data}/${split}/annotations.csv;
done
# Deeplearning baseline training
python baseline_deeplearning_training.py --output-dir=deeplearning_baseline_${data}_model --train-dir=${datadir}/${data}/train --val-dir=${datadir}/${data}/val --lr=0.0001 --n-epochs=25;
# Deeplearning baseline inference
for split in val;
do
python baseline_deeplearning_inference.py --output-dir=deeplearning_baseline_${data}_${split} --inference-dir=${datadir}/${data}/${split}/audio --pretrained-fp=deeplearning_baseline_${data}_model/model.pt;
python evaluate.py --predictions-fp=deeplearning_baseline_${data}_${split}/predictions.csv --ground-truth-fp=${datadir}/${data}/${split}/annotations.csv;
done
done
