# Patient Complaint Multi-label Classification (PyTorch)

Project: **基于 Transformer 及自监督学习的患者主诉文本多标签症状识别模型**

目录结构
```
patient_complaint_project/
├── data_cleaned/                 # put train_data_cleaned.json and test_data_cleaned.json here
├── ckpts/                        # model checkpoints will be saved here
├── analysis/                     # analysis outputs (plots/csv)
├── main.py                       # entrypoint
├── tokenizer.py                  # vocab & tokenizer (jieba)
├── dataset.py                    # Dataset classes & augmentation
├── model_transformer.py          # Transformer encoder + classifier
├── model_simcse.py               # SimCSE wrapper
├── train_simcse.py               # SimCSE training script
├── train_supervised.py           # supervised training script
├── evaluate.py                   # evaluation metrics
├── analysis_data.py              # data analysis & plots
├── utils.py                      # helper functions
├── requirements.txt
└── README.md
```

## Quick start

1. Put your `train_data_cleaned.json` and `test_data_cleaned.json` in `data_cleaned/`.

2. Install dependencies (recommended in a virtualenv):
```bash
pip install -r requirements.txt
```

3. Build vocabulary:
```bash
python main.py --stage build_vocab --data_dir ./data_cleaned --out_dir ./ckpts
```

4. Data analysis:
```bash
python main.py --stage analyze --data_dir ./data_cleaned --out_dir ./analysis
```

5. (Optional) Train SimCSE self-supervised encoder:
```bash
python main.py --stage simcse --data_dir ./data_cleaned --out_dir ./ckpts --epochs 3 --batch_size 32
```

6. Train supervised multi-label classifier (can load simcse checkpoint):
```bash
python main.py --stage train --data_dir ./data_cleaned --out_dir ./ckpts --simcse_ckpt ./ckpts/simcse.pth --epochs 6 --batch_size 16
```

7. Evaluate saved model:
```bash
python main.py --stage eval --data_dir ./data_cleaned --out_dir ./ckpts --model_ckpt ./ckpts/best_supervised.pth
```

## Notes

- This project **does not** require Hugging Face. All models are built with pure PyTorch and initialized randomly.
- If you want to use a pre-trained BERT/RoBERTa later, it is straightforward to replace the encoder with a pre-trained model for faster convergence and better performance.