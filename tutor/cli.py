from tutor.orchestrator import run_turn
def main():
    history = []
    while(True):
        try:
            prompt = input("> ")
            turn = run_turn(prompt, "Russian", history)
            history.append({"role": "user", "content": prompt})
            history.append({"role": "assistant", "content": turn.spoken})
            print(turn.spoken)
        except (KeyboardInterrupt, EOFError):
            print()
            break







if __name__ == "__main__": main()