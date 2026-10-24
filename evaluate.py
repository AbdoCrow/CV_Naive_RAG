from app import CVChatbot


def main():
    bot = CVChatbot()
    # Each inner list contains alternative spellings for one required fact.
    cases = [
        ("Who is the person?", [["abdalrahman essam mohamed"]]),
        ("What programming languages does the candidate know?",
         [["python"], ["c++"], ["dart"], ["kotlin"], ["java"], ["sql"]]),
        ("What is the candidate's GPA?", [["3.51"]]),
        ("Where did the candidate work as a data engineering intern?",
         [["summit technology solutions"]]),
        ("What methods were used in industrial machine sound classification?",
         [["log-mel", "log mel"], ["cnn", "convolutional neural network"]]),
        ("What is the candidate's passport number?",
         [["the information is not available in the cv"]]),
    ]
    passed = 0
    for question, expected in cases:
        bot.memory.clear()  # Standalone cases should not affect each other.
        answer = bot.ask(question)
        ok = all(any(word in answer.lower() for word in alternatives)
                 for alternatives in expected)
        passed += int(ok)
        print(f"{'PASS' if ok else 'FAIL'}: {question}\nAnswer: {answer}\n")

    # Check that a follow-up can refer to the previous turn.
    bot.memory.clear()
    bot.ask("Where did the candidate work as a data engineering intern?")
    answer = bot.ask("What was the name of that company again?")
    ok = "summit technology solutions" in answer.lower()
    passed += int(ok)
    print(f"{'PASS' if ok else 'FAIL'}: Conversation follow-up\nAnswer: {answer}\n")
    total = len(cases) + 1
    print(f"Passed {passed}/{total} checks.")
    print("These keyword checks are basic; read the answers to check their quality too.")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
