def cmn_cmd_progressBar(current, total, barLength = 50, message="Progress", only_one = False):
    total -= 1
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))

    if only_one:    # 1 - instance
        if current == 0:
            print(f"{message}: ")
        print(f' [{arrow}{spaces}] {int(percent)} %', end='\r')
    else:   # if many above logic fails
        print(f'{message}: [{arrow}{spaces}] {int(percent)} %', end='\r')