const { GoogleGenerativeAI } = require("@google/generative-ai");
const readline = require('readline');
const dotenv = require('dotenv');
const fs = require('fs');

// Get API key from .env and initialize genAI variable
dotenv.config();
const genAI = new GoogleGenerativeAI(process.env.KEY);

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

async function run() {
    // Initialize the model
    const model = genAI.getGenerativeModel({ model: "gemini-pro" });

    // Ask for AI generated text input
    console.log('Enter AI generated text to be humanized with your test prompt (finish with an empty line):');

    let text = '';
    rl.on('line', (input) => {
        if (input === '') {
            rl.close();
        } else {
            text += input + '\n';
        }
    });

    rl.on('close', async () => {
        // Read the prompt file and replace [text] with the user input
        let prompt_text = fs.readFileSync("prompt.txt", { "encoding": "utf-8" });
        prompt_text = prompt_text.replace(/\[text\]/g, text.trim());

        console.log("\n\nGemini Response (Might take a sec):\n\n");

        // Generate and output the AI's response
        const result = await model.generateContent(prompt_text);
        const response = await result.response;
        const data = await response.text();
        console.log(data);
    });
}

run();
