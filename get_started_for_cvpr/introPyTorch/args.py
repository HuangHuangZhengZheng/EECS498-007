import argparse

def main():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='计算给定数字的平方')

    # 添加位置参数
    parser.add_argument("number", type=int, help="要计算平方的数字")

    # 添加可选参数
    parser.add_argument("-v", "--verbosity", action="count", default=0,
                        help="增加输出的详细程度")

    # 解析命令行参数
    args = parser.parse_args()

    # 计算平方
    result = args.number ** 2

    # 根据 verbosity 输出不同的结果
    if args.verbosity >= 2:
        print(f"计算 {args.number} 的平方，结果是 {result}")
    elif args.verbosity >= 1:
        print(f"{args.number} 的平方是 {result}")
    else:
        print(result)

if __name__ == "__main__":
    main()