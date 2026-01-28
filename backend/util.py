def terminal_payload(state):
    returns = state.returns()
    print(returns)
    if returns[0] > returns[1]: 
        result = "black_win"
    elif returns[1] > returns[0]:
        result = "white_win"
    else: 
        result = "draw"

    return {
        "terminal": True,
        "result": result,
        "returns": returns,
    }

