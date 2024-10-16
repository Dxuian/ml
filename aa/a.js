const dotenv = require('dotenv');

// Load environment variables from .env file
dotenv.config();

// Access the environment variables
const ARLIAI_API_KEY = process.env.ARLIAI_API_KEY;

async function fetchChatCompletion(messages, prompt) {
    try {
        const response = await fetch("https://api.arliai.com/v1/chat/completions", {
            method: "POST",
            headers: {
                "Authorization": `Bearer ${ARLIAI_API_KEY}`,
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                "model": "Meta-Llama-3.1-8B-Instruct",
                "messages": messages,
                "prompt": prompt,
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
                    console.error('Error:', error);
                }
            });

            return read();
        }

        await read();
        return combinedContent;

    } catch (error) {
        console.error('There has been a problem with your fetch operation:', error);
    }
}

async function main() {
    const initialMessages = [
        {"role": "system", "content": "You are a helpful assistant."},
    ];

    let prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>";

    // First question
    initialMessages.push({"role": "user", "content": "What is the diameter of the sun?"});
    prompt += "<|start_header_id|>user<|end_header_id|>\n\nWhat is the diameter of the sun?<|eot_id|>";
    let response1 = await fetchChatCompletion(initialMessages, prompt);
    console.log('Response 1:', response1);

    // Second question based on the first response
    initialMessages.push({"role": "assistant", "content": response1});
    initialMessages.push({"role": "user", "content": "How fast does it move?"});
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n" + response1 + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHow fast does it move?<|eot_id|>";
    let response2 = await fetchChatCompletion(initialMessages, prompt);
    console.log('Response 2:', response2);

    // Third question based on the second response
    initialMessages.push({"role": "assistant", "content": response2});
    initialMessages.push({"role": "user", "content": "How dense is it?"});
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n" + response2 + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHow dense is it?<|eot_id|>";
    let response3 = await fetchChatCompletion(initialMessages, prompt);
    console.log('Response 3:', response3);
}

main();