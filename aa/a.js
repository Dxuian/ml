const dotenv = require('dotenv');

// Load environment variables from .env file
dotenv.config();

// Access the environment variables
const ARLIAI_API_KEY = process.env.ARLIAI_API_KEY;

async function fetchChatCompletion() {
    try {
        const response = await fetch("https://api.arliai.com/v1/chat/completions", {
            method: "POST",
            headers: {
                "Authorization": `Bearer ${ARLIAI_API_KEY}`,
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                "model": "Meta-Llama-3.1-8B-Instruct",
                "messages": [
                    {"role": "system", "content": "Answer but have a vulgarity filter"},
                    {"role": "assistant", "content": "what is the square root of 25?"},
                    // {"role": "user", "content": question}
                ],
                "repetition_penalty": 1.1,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "max_tokens": 1024,
                "stream": true
            })
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let combinedContent = '';

        async function read() {
            const { done, value } = await reader.read();
            if (done) {
                console.log('Done reading');
                console.log('Combined Content:', combinedContent);
                return combinedContent;
            }

            let text = decoder.decode(value, { stream: true });
            console.log('Chunk:', text);
            const lines = text.trim().split('\n');

            lines.forEach(line => {
                try {
                    if (line.startsWith('data: ')) {
                        const modifiedLine = '{"' + line.slice(0, 4) + '"' + line.slice(4) + '}';
                        let content = modifiedLine.replace(/\\"/g, '');
                        content = JSON.parse(content);
                        if (content.data.choices[0].delta.content != null) {
                            combinedContent += content.data.choices[0].delta.content;
                        }
                    }
                } catch (error) {
                    // console.error('Error:', error);
                    
                }
            });

            return read();
        }

        await read();
        // const x = 123;
        // console.log('x:', x);
        return combinedContent;

    } catch (error) {
        console.error('There has been a problem with your fetch operation:', error);
    }
}

async function main() {
    const x = await fetchChatCompletion();
    console.log('x:', x);
}

main();