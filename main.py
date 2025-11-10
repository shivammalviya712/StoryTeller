from storyteller import StoryTeller

"""
Before submitting the assignment, describe here in a few sentences what you would have built next if you spent 2 more hours on this project:

"""

example_requests = "A story about a girl named Alice and her best friend Bob, who happens to be a cat."


def main():
    storyteller = StoryTeller()
    print("Type 'exit' or 'quit' to end the conversation.")
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        response = storyteller.tell_story(user_input)
        print(f"\nStoryteller: {response}\n")


if __name__ == "__main__":
    main()