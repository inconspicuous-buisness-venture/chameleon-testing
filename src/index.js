const { GoogleGenerativeAI } = require("@google/generative-ai");
const prompt = require('prompt-sync');
const dotenv = require('dotenv');
const fs = require('fs');

// Get API key from .env and initialize genAI variable thingimabob
dotenv.config();
const genAI = new GoogleGenerativeAI(process.env.KEY);


async function run() {
    // You can also pass in restriction stuff here or use a different model
    const model = genAI.getGenerativeModel({ model: "gemini-pro" });

    // HERE is where you input the AI generated text that'll be inserted to your test prompt (prompt.txt)
    const text = require('prompt-sync')({sigint: true});

    // Rest of the logic just reads your prompt, replaces [text] with the text variable, and gets/outputs the AI's response
    let prompt = fs.readFileSync("prompt.txt", {"encoding": "utf-8"});
    prompt = prompt.replace(/\[text\]/g, text);

    console.log("Your Inputted Prompt:\n\n" + prompt + "\n\nResult (Might take a sec):\n\n");
    
    const result = await model.generateContent(prompt);
    const response = await result.response;
    const data = response.text();
    console.log(data);

} run();