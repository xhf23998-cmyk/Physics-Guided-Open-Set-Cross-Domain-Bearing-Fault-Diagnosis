"""项目统一入口 — 按 experiment.txt 的 Step 编排运行实验"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="轴承故障诊断实验入口")
    parser.add_argument("--step", type=int, required=True,
                        help="实验步骤: 0=测试数据, 3=闭集分类, 5=跨域迁移, 6=开集诊断")
    parser.add_argument("--source", type=str, default="W1", help="源域工况")
    parser.add_argument("--target", type=str, default="W2", help="目标域工况")
    parser.add_argument("--unknown", type=str, default="Ball", help="开集未知类别")
    parser.add_argument("--dataset", type=str, default="self_collected",
                        choices=["self_collected", "cwru", "pu"], help="数据集")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=None, help="批大小")

    args = parser.parse_args()

    # 更新配置
    from configs import cfg
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.batch_size:
        cfg.train.batch_size = args.batch_size

    if args.step == 0:
        # 数据加载测试
        print("运行数据加载测试...")
        from test_data_loading import test_self_collected, test_cwru, test_pu, test_model
        test_self_collected()
        test_cwru()
        test_pu()
        test_model()

    elif args.step == 3:
        # Step 3: 同工况闭集分类
        cfg.experiment.task = "closed_set"
        cfg.experiment.dataset = args.dataset
        from train_closed_set import main as train_main
        train_main()

    elif args.step == 5:
        # Step 5: 跨工况闭集迁移
        cfg.experiment.task = "cross_domain"
        cfg.experiment.source_domain = args.source
        cfg.experiment.target_domain = args.target
        from train_cross_domain import main as train_main
        train_main()

    elif args.step == 6:
        # Step 6: 跨工况开集诊断
        cfg.experiment.task = "open_set"
        cfg.experiment.source_domain = args.source
        cfg.experiment.target_domain = args.target
        cfg.experiment.unknown_class = args.unknown
        from train_cross_domain import main as train_main
        train_main()

    else:
        print(f"未知步骤: {args.step}")
        print("支持: 0=测试, 3=闭集, 5=跨域闭集, 6=开集诊断")


if __name__ == "__main__":
    main()
