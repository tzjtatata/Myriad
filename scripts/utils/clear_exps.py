import os
import shutil
import argparse


def check_extents(file_names, postfix):
    for f in file_names:
        if f.endswith(f'.{postfix}'):
            return True
    return False


def check_jsonl(file_names):
    return check_extents(file_names, 'jsonl')


def check_pth(file_names):
    return check_extents(file_names, 'pth')


def find_pths(file_names):
    ret = []
    for f in file_names:
        if f.endswith('.pth'):
            try:
                ret.append(int(f[:-4].split('_')[-1]))
            except:
                print(f)
    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--empty", action='store_true')
    parser.add_argument("--rm-pth", default='none', choices=[
        'keep_last', 'keep_choice', 'rm_all', 'none'
    ])
    parser.add_argument("--keep-max", default=3, type=int)
    parser.add_argument("--apply", action='store_true')
    args = parser.parse_args()

    root = "/home/lyz/vdb/results/minigpt/mvtec_bak"
    exp_names = [p for p in os.listdir(root) if os.path.isdir(os.path.join(root, p))]
    rmdir_ops = []
    rmpth_ops = []
    rmpth_struc = {}  # TODO: 结构化打印
    count = 0
    for exp_name in exp_names:
        exp_path = os.path.join(root, exp_name)
        time_stamps = [
            p 
            for p in os.listdir(exp_path) 
            if os.path.isdir(os.path.join(exp_path, p)) and os.path.exists(os.path.join(exp_path, p, 'log.txt'))
        ]
        
        for run_id in time_stamps:
            count += 1
            run_path = os.path.join(exp_path, run_id)
            run_files = [p for p in os.listdir(run_path) if os.path.isfile(os.path.join(run_path, p))]
            if check_jsonl(run_files) or check_pth(run_files):
                # 说明是有东西的
                pth_ids = find_pths(run_files)
                if args.rm_pth == 'keep_last':
                    pth_ids = sorted(pth_ids, reverse=True)
                    pth_ids = pth_ids[args.keep_max:]
                elif args.rm_pth == 'keep_choice':
                    raise NotImplementedError("Not Implement keep_choice")
                elif args.rm_pth == 'rm_all':
                    if not check_jsonl(run_files):
                        pth_ids = sorted(pth_ids, reverse=True)
                        pth_ids = pth_ids[1:]
                elif args.rm_pth == 'none':
                    continue
                else:
                    raise NotImplementedError(f"Not Implement {args.rm_pth}")
                rmpth_ops.extend([
                    os.path.join(run_path, f'checkpoint_{ind}.pth')
                    for ind in pth_ids
                ])
            else:
                if args.empty:
                    rmdir_ops.append(run_path)
    
    print(f"Swap throught {len(exp_names)} experiments and {count} directory.")
    print(rmdir_ops)
    print(f"Try to Remove directory: {len(rmdir_ops)}")
    print(rmpth_ops)
    print(f"Try to Remove ckpt: {len(rmpth_ops)}")
    

    if len(rmdir_ops) > 0 and args.empty and args.apply:
        for p in rmdir_ops:
            shutil.rmtree(p)
    
    if len(rmpth_ops) > 0 and args.apply:
        from tqdm import tqdm
        for p in tqdm(rmpth_ops, total=len(rmpth_ops)):
            os.remove(p)