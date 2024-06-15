# Prompt Testing
Use this repo for any AI prompt edits, testing, creating, etc.

- All prompts used and in development should also be stored here for API integration later or future updates-

When we want to review/test a seemingly good prompt we should make an issue for each of us to test it & share results to decide if it should be in API iteration

## Folder/File structure

- `[/src/index.js](/src/index.js)` - This is the script to run which uses the prompt.txt to get AI results.

- `[/src/prompt.txt](/src/prompt.txt)` - Here is the hot prompt file which is given to the AI model (Gemini Pro for now).

- `[/prompts/Temp/](/prompts/Temp/)` - Folder explained in its README

- `[/prompts/Garbage/](/prompts/Garbage/)` - Folder explained in it's README

<br>

- `[/notes.md](notes.md)` - Just notes you can add when testing prompts for others to see


### Config Files
The `package-lock.json` and `package.json` files should be ignored as they're just config files for node.

## Usage
write your prompt in `prompt.txt` run `npm install` (you only have to run that once per system) and run the `npm test` command to test the prompt in `prompt.txt`

# TODO/ISSUES
