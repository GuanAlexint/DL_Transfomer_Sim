import argparse, os
from tokenizer import Tokenizer
from utils import load_data_paths
from analysis_data import analyze_dataset
from train_simcse import train_simcse
from train_supervised import train_supervised

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['analyze','build_vocab','simcse','train','eval'], required=True)
    parser.add_argument('--data_dir', type=str, default='./data_cleaned')
    parser.add_argument('--out_dir', type=str, default='./ckpts')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--simcse_ckpt', type=str, default='')
    parser.add_argument('--model_ckpt', type=str, default='')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train, test = load_data_paths(args.data_dir)

    if args.stage == 'analyze':
        analyze_dataset(f"{args.data_dir}/train_data_cleaned.json", f"{args.data_dir}/test_data_cleaned.json", out_dir='./analysis')
        return

    # build vocab
    all_texts = [ (x.get('visit_sn','') or '') for x in (train + test) ]
    tok = Tokenizer()
    tok.build_vocab_from_texts(all_texts)
    vocab_path = os.path.join(args.out_dir, 'vocab.txt')
    tok.save_vocab(vocab_path)
    print('Saved vocab to', vocab_path)

    if args.stage == 'simcse':
        train_simcse(train, tok, os.path.join(args.out_dir, 'simcse.pth'), epochs=args.epochs, batch_size=args.batch_size)
    elif args.stage == 'train':
        train_supervised(train, test, tok, args.out_dir, simcse_ckpt=args.simcse_ckpt, epochs=args.epochs, batch_size=args.batch_size)
    elif args.stage == 'eval':
        # simple eval via train_supervised module (it saves best_supervised.pth)
        print('Run eval by loading model via train_supervised or implement separate eval.')
    print('Done.')

if __name__ == '__main__':
    main()