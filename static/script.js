document.getElementById('send-button').addEventListener('click', function() {
    const userInput = document.getElementById('user-input').value.trim();
    
    if (!userInput) return; // Prevent sending empty messages

    // Create a new message element for the user's input
    const chatBox = document.getElementById('chat-box');
    const userMessage = document.createElement('div');
    userMessage.classList.add('message', 'user'); // Add user class for styling
    userMessage.innerHTML = `${userInput}`; // No label
    chatBox.appendChild(userMessage);
    
    // Clear the input field
    document.getElementById('user-input').value = '';

    // Show typing indicator
    const typingIndicator = document.getElementById('typing-indicator');
    typingIndicator.classList.remove('hidden');

    // Record the start time
    const startTime = Date.now();

    // Simulate bot response delay
    setTimeout(() => {
        // Send user input to the chatbot
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ user_input: userInput }),
        })
        .then(response => response.json())
        .then(data => {
            // Hide typing indicator
            typingIndicator.classList.add('hidden');

            // Calculate response time
            const responseTime = Date.now() - startTime; // Time taken in milliseconds
            const seconds = (responseTime / 1000).toFixed(2); // Convert to seconds

            // Format bot response with book details
            const formattedResponse = formatBotResponse(data.response, seconds);

            // Create a new message element for the bot's response
            const botMessage = document.createElement('div');
            botMessage.classList.add('message', 'bot'); // Add bot class for styling
            botMessage.innerHTML = formattedResponse;
            chatBox.appendChild(botMessage);

            chatBox.scrollTop = chatBox.scrollHeight;  // Auto-scroll to bottom
        })
        .catch((error) => {
            console.error('Error:', error);
        });

    }, 1000); // Simulate a delay of 1 second
});

// Function to format bot response
function formatBotResponse(response, responseTime) {
    // Replace newlines (\n) with <br> for line breaks
    const formattedResponse = response.replace(/\n/g, '<br>'); // Format response to HTML
    return `${formattedResponse} <span class="response-time">(Response time: ${responseTime} seconds)</span>`;
}
