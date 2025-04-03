import axios, { AxiosResponse } from 'axios';
import * as fs from 'fs/promises';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

// Create equivalents for __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Configure logging with colorized output
const colors = {
	reset: '\x1b[0m',
	bright: '\x1b[1m',
	dim: '\x1b[2m',
	underscore: '\x1b[4m',
	red: '\x1b[31m',
	green: '\x1b[32m',
	yellow: '\x1b[33m',
	blue: '\x1b[34m',
	magenta: '\x1b[35m',
	cyan: '\x1b[36m',
	white: '\x1b[37m',
};

const logger = {
	info: (message: string) => console.info(`${colors.blue}${new Date().toISOString()} - INFO - ${message}${colors.reset}`),
	error: (message: string) => console.error(`${colors.red}${new Date().toISOString()} - ERROR - ${message}${colors.reset}`),
	success: (message: string) => console.log(`${colors.green}${new Date().toISOString()} - SUCCESS - ${message}${colors.reset}`),
	warning: (message: string) => console.warn(`${colors.yellow}${new Date().toISOString()} - WARNING - ${message}${colors.reset}`),
	debug: (message: string) => console.debug(`${colors.dim}${new Date().toISOString()} - DEBUG - ${message}${colors.reset}`),
	important: (message: string) => console.log(`${colors.bright}${colors.magenta}${new Date().toISOString()} - IMPORTANT - ${message}${colors.reset}`),
	progress: (percent: number, message: string) => {
		const barLength = 30;
		const filled = Math.round(barLength * percent / 100);
		const bar = '█'.repeat(filled) + '░'.repeat(barLength - filled);
		console.log(`${colors.cyan}${new Date().toISOString()} - PROGRESS - [${bar}] ${percent.toFixed(1)}% - ${message}${colors.reset}`);
	}
};

// GPTZero API endpoint and configs
const gptzeroUrl = "https://api.gptzero.me/v2/predict/text";
const apiKeys = [
    "45eaf4bb82c147e8be9d07f217b47055",
    "d822b4fd061a4fed9f1bb76d7a7ac3f7",
    "45eaf4bb82c147e8be9d07f217b47055",
    "8861330a6bec4b05a048ced7c6698965",
    "fc7b0bb4bf9d46ac8c739d6db913f523",
    "428fe9fa5bf54bada967e8983d0b415a",
    "0bfab717af1f45cab1b2b87bbb8e0a5c",
    "ef690dcc40414db68c18b9683810e1a9",
    "6eee50b894f742238266bd5be6717b74",
    "69b58e2bb0eb4fea98b3d7678f11bd12",
    "1542a5af25e74ccdb3455bbb85a03eee",
    "0c37781a37be4e2b95ecb789bc30202d",
    "43c9c9744910471880c89aa33745f93b",
    "b3fe6a72f207452fac015cd649239a2f",
    "8bf8a3e4c41242f18fad0711c4c3af58",
	"c03a730f471d4a918523684ecdd1a96e",
	"877352fce13944d69f44eb67eb65a3c3",
	"5dab7b51f2eb491bac904fc7b2593e51",
	"c2c4cc70f97a41c3a640cfd82c892700",
	"7dc0685d41854ef9bace197173c90088",
    // Add more API keys here
];

// Track rate limit occurrences for each API key
const apiKeyRateLimits = new Map<string, number>();

// Interfaces
interface DetectionScores {
    gptzero: number;
    zerogpt: number;
    roberta: number;
}

interface Text {
    text: string;
    detectionScores: DetectionScores;
    wordFrequency: any;
}

interface Iteration extends Text {
    iteration: number;
}

interface Model {
    modelName: string;
    summarizedText: Text;
    iterations: Iteration[];
}

interface TextDataSet {
    id: number;
    originalText: Text;
    models: Record<string, Model>;
}

interface GptZeroResult {
    documents?: Array<{ completely_generated_prob: number }>;
    error?: string;
}

interface GptZeroError {
    error: string;
}

async function checkWithGPTZero(text: string): Promise<number> {
    const payload = { document: text };
    
    try {
        const result = await callGptZeroApi(payload);
        if ('error' in result) {
            logger.error(`GPTZero check failed: ${result.error}`);
            return -1;
        }
        const score = result.documents?.[0]?.completely_generated_prob || -1;
        return Number(score.toFixed(4));
    } catch (error) {
        logger.error(`Error processing text: ${error}`);
        return -1;
    }
}

async function processDataSet(data: TextDataSet[]): Promise<TextDataSet[]> {
    for (const dataset of data) {
        try {
            // Process original text
            if (dataset.originalText.detectionScores.gptzero === -1) {
                dataset.originalText.detectionScores.gptzero = await checkWithGPTZero(dataset.originalText.text);
				logger.success(`Processed original text for dataset ${dataset.id}: ${dataset.originalText.detectionScores.gptzero}`);
            }

            // Process each model's data
            for (const modelData of Object.values(dataset.models)) {
                // Process summarized text
                if (modelData.summarizedText.detectionScores.gptzero === -1) {
                    modelData.summarizedText.detectionScores.gptzero = await checkWithGPTZero(modelData.summarizedText.text);
					logger.success(`Processed summarized text for dataset ${dataset.id}, model ${modelData.modelName}: ${modelData.summarizedText.detectionScores.gptzero}`);
                }

                // Process iterations
                for (let i = 0; i < modelData.iterations.length; i++) {
					const iteration = modelData.iterations[i];
                    if (iteration.detectionScores.gptzero === -1) {
                        iteration.detectionScores.gptzero = await checkWithGPTZero(iteration.text);
						logger.success(`Processed iteration ${iteration.iteration} for dataset ${dataset.id}, model ${modelData.modelName}: ${iteration.detectionScores.gptzero}`);
                    }
                }
            }
            // Save progress after each dataset
            await saveProgress(data);
        } catch (error) {
            logger.error(`Error processing dataset ${dataset.id}: ${error}`);
            await saveProgress(data);
            throw error;
        }
    }
    return data;
}

async function saveProgress(data: TextDataSet[]): Promise<void> {
    const outputPath = path.join(__dirname, 'finalTextDataSets.json');
    await fs.writeFile(outputPath, JSON.stringify(data, null, 4));
    logger.info('Progress saved successfully');
}

async function main(): Promise<void> {
    try {
        const inputPath = path.join(__dirname, 'textDataSets.json');
        const finalPath = path.join(__dirname, 'finalTextDataSets.json');

        // Determine which file to use as input
        let data: TextDataSet[];
        try {
            const finalFileExists = await fs.access(finalPath)
                .then(() => true)
                .catch(() => false);

            if (finalFileExists) {
                const finalContent = await fs.readFile(finalPath, 'utf-8');
                data = JSON.parse(finalContent);
                logger.info('Continuing from finalTextDataSets.json');
            } else {
                const inputContent = await fs.readFile(inputPath, 'utf-8');
                data = JSON.parse(inputContent);
                logger.info('Starting fresh from textDataSets.json');
            }
        } catch (error) {
            logger.error(`Error reading input file: ${error}`);
            return;
        }

        // Process the data
        await processDataSet(data);
        logger.info('Processing completed successfully');

    } catch (error) {
        logger.error(`Main execution error: ${error}`);
    }
}

// Run the main function
if (require.main === module) {
    main().catch(err => {
        logger.error(`Unhandled error: ${err}`);
        process.exit(1);
    });
}

// Keep the existing callGptZeroApi function unchanged
async function callGptZeroApi(
    payload: Record<string, any>
): Promise<GptZeroResult | GptZeroError> {
    for (const apiKey of apiKeys) {
        // Skip keys that have been rate limited 3 or more times
        const rateLimitCount = apiKeyRateLimits.get(apiKey) || 0;
        if (rateLimitCount >= 3) {
            continue;
        }

        try {
            const headers = getGptZeroHeaders(apiKey);
            const response: AxiosResponse = await axios.post(gptzeroUrl, payload, { headers });
            return response.data;
        } catch (error) {
            if (axios.isAxiosError(error)) {
                if (error.response?.status === 429 || error.message.includes('too many requests')) {
                    // Increment the rate limit count for this key
                    const newCount = (apiKeyRateLimits.get(apiKey) || 0) + 1;
                    apiKeyRateLimits.set(apiKey, newCount);
                    
                    if (newCount >= 3) {
                        logger.error(`API key ${apiKey.substring(0, 8)}... has been rate limited 3 times, marking as exhausted`);
                    } else {
                        logger.error(`API key ${apiKey.substring(0, 8)}... rate limited (${newCount}/3), trying next key...`);
                    }
                    continue;
                }
                logger.error(`Error calling GPTZero API: ${error.message}`);
            }
            logger.error(`Unexpected error with GPTZero API: ${error}`);
        }
    }
    
    logger.error('All API keys exhausted. Saving progress and exiting...');
    const data = await fs.readFile(path.join(__dirname, 'finalTextDataSets.json'), 'utf-8')
        .then(JSON.parse)
        .catch(() => null);
    
    if (data) {
        await saveProgress(data);
    }
    process.exit(1);
}

function getGptZeroHeaders(apiKey: string) {
    return {
        'accept': 'application/json',
        'X-Api-Key': apiKey,
        'Content-Type': 'application/json'
    };
}