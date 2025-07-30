// src/components/ChatBox.jsx
// Chat interface component that displays conversation history with auto-scrolling functionality
import React, { useEffect, useRef } from 'react';  // React hooks for side effects and DOM references
import PropTypes from 'prop-types';  // Runtime type checking for component props
import Message from './Message'; // Import the new Message component for individual message rendering
import './ChatBox.css';  // Component-specific styling

/**
 * Renders the chat message history area and handles auto-scrolling.
 * Uses the Message component to render individual messages.
 * @param {object} props - Component props.
 * @param {Array<object>} props.messages - Array of message objects containing conversation history.
 */
function ChatBox({ messages }) {
    // Reference to the chat container for scroll manipulation
    const chatboxRef = useRef(null);

    // Auto-scroll to bottom when new messages arrive
    useEffect(() => {
        if (chatboxRef.current) {
            // Smoothly scroll to the bottom of the chat container
            chatboxRef.current.scrollTo({
                top: chatboxRef.current.scrollHeight,  // Scroll to maximum height
                behavior: 'smooth'  // Animated scrolling for better UX
            });
        }
    }, [messages]);  // Re-run effect when messages array changes

    return (
        <div className="chatbox" ref={chatboxRef}>
            {/* Map over messages and render the Message component for each conversation entry */}
            {messages.map((msg, index) => (
                <Message
                    key={index} // Still using index as key here - unique identifier for React rendering
                    message={msg} // Pass the whole message object as a prop to Message component
                />
            ))}
        </div>
    );
}

// PropTypes for runtime type checking and documentation
ChatBox.propTypes = {
    messages: PropTypes.arrayOf(PropTypes.shape({
        sender: PropTypes.oneOf(['user', 'ada']).isRequired,  // Message sender validation
        text: PropTypes.string.isRequired,  // Message content validation
    })).isRequired,  // Messages array is required
};

export default ChatBox;  // Export component for use in parent components