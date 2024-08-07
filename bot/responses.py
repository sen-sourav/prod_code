def handle_response(message) -> str:
    p_message = message.lower()

    if p_message == 'hello':
        return 'Hey There! Type Intro to learn more about me'
    if message == 'Intro':
        return "Welcome to YOLO Music. I am Turbo Dave Bot and here to help you create amazing music"
    else:
        return "Sorry, I don't understand what you mean. Try Again"
