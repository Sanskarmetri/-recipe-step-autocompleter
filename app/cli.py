import argparse

from app.core import milestone_1, milestone_2, predict_next_word
def autocomplete():

    print("\n" + "=" * 45)
    print("       RECIPE STEP AUTOCOMPLETER")
    print("=" * 45)

    print("\nLoading recipe data...")

    corpus = milestone_1()
    model = milestone_2(corpus)

    print("Model trained successfully!")

    print("\nCommands:")
    print("  clear - Reset current instruction")
    print("  new   - Start a new instruction")
    print("  exit  - Quit application")

    current_step = ""

    while True:

        # Ask for input if there is no current instruction.
        if not current_step:

            user_input = input(
                "\nEnter recipe instruction:\n> "
            ).strip()

            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            if user_input.lower() == "clear":
                print("Nothing to clear.")
                continue

            if not user_input:
                continue

            current_step = user_input

        # Generate predictions.
        predictions = predict_next_word(
            model,
            current_step
        )

        if not predictions:
            print("\nNo suggestions found.")
            current_step = ""
            continue

        print("\nCurrent instruction:")
        print(current_step)

        print("\nSuggested next words:")

        for index, (word, probability) in enumerate(
            predictions,
            start=1
        ):
            print(
                f"{index}. {word} ({probability * 100:.2f}%)"
            )

        # Ask the user to select a suggestion.
        choice = input(
            "\nChoose a word (1-3), or enter a command:\n> "
        ).strip().lower()

        if choice in ["exit", "quit"]:
            print("Goodbye!")
            break

        elif choice == "clear":
            current_step = ""
            print("\nInstruction cleared!")

        elif choice == "new":
            current_step = ""
            print("\nStarting a new instruction...")

        elif choice.isdigit():

            selected_index = int(choice)

            if 1 <= selected_index <= len(predictions):

                selected_word = predictions[selected_index - 1][0]

                current_step += " " + selected_word

                print("\nUpdated instruction:")
                print(current_step)

            else:
                print("\nInvalid selection. Try again.")

        else:
            print("\nInvalid input. Enter a number or command.")
def main(argv: list[str] | None = None):

    parser = argparse.ArgumentParser(
        prog="recipe-step-autocompleter",
        description="Autocomplete incomplete recipe instructions."
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True
    )

    sub.add_parser("hello")

    sub.add_parser("autocomplete")

    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Recipe Step Autocompleter is ready.")

    elif args.command == "autocomplete":
        autocomplete()


if __name__ == "__main__":
    main()            