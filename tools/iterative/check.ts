import axios, { AxiosResponse } from 'axios';
import * as fs from 'fs/promises';
import * as path from 'path';

// Configure logging
const logger = {
	info: (message: string) => console.info(`${new Date().toISOString()} - INFO - ${message}`),
	error: (message: string) => console.error(`${new Date().toISOString()} - ERROR - ${message}`)
};

// API endpoint
const gptzeroUrl = "https://api.gptzero.me/v2/predict/text";

// Array of API keys
const apiKeys = [
    "45eaf4bb82c147e8be9d07f217b47055",
    // Add more API keys here
];

// Modified API configuration to accept different keys
const getGptzeroHeaders = (apiKey: string) => ({
    "x-api-key": apiKey,
    "Content-Type": "application/json",
    "Accept": "application/json"
});

// Interface for the JSON structure
interface TextDataSet {
    id: string;
    text: string;
    gptzero_detection_score: number;
    [key: string]: any;
}

/**
 * Makes an API call with retry logic for multiple API keys
 */
async function callApiWithRetry(
    url: string,
    payload: Record<string, any>,
    apiName: string
): Promise<ApiResult | ApiError> {
    for (const apiKey of apiKeys) {
        try {
            const headers = getGptzeroHeaders(apiKey);
            const response: AxiosResponse = await axios.post(url, payload, { headers });
            return response.data;
        } catch (error) {
            if (axios.isAxiosError(error)) {
                if (error.response?.status === 429 || error.message.includes('too many requests')) {
                    logger.error(`API key ${apiKey.substring(0, 8)}... rate limited, trying next key...`);
                    continue;
                }
                logger.error(`Error calling ${apiName} API: ${error.message}`);
                return { 
                    error: error.message, 
                    status: error.response?.status 
                };
            }
            logger.error(`Unexpected error with ${apiName} API: ${error}`);
            return { error: String(error) };
        }
    }
    return { error: "All API keys exhausted or rate limited" };
}

/**
 * Processes the textDataSets.json file and updates GPTZero scores
 */
async function processTextDataSets() {
    try {
        const filePath = path.join(__dirname, '..', '..', 'data', 'textDataSets.json');
        const fileContent = await fs.readFile(filePath, 'utf-8');
        const textDataSets: TextDataSet[] = JSON.parse(fileContent);
        let updated = false;

        for (const dataset of textDataSets) {
            if (dataset.gptzero_detection_score === -1) {
                logger.info(`Processing text ID: ${dataset.id}`);
                const result = await checkText(dataset.text);
                
                if ('completely_generated_prob' in result.gptzero_result) {
                    dataset.gptzero_detection_score = result.gptzero_result.completely_generated_prob;
                    updated = true;
                    logger.info(`Updated score for ID ${dataset.id}: ${dataset.gptzero_detection_score}`);
                } else {
                    logger.error(`Failed to get score for ID ${dataset.id}: ${JSON.stringify(result.gptzero_result)}`);
                }
                
                // Add a small delay to avoid overwhelming the API
                await new Promise(resolve => setTimeout(resolve, 1000));
            }
        }

        if (updated) {
            await fs.writeFile(filePath, JSON.stringify(textDataSets, null, 2), 'utf-8');
            logger.info('Updated textDataSets.json with new scores');
        }

        return textDataSets;
    } catch (error) {
        logger.error(`Error processing text data sets: ${error}`);
        throw error;
    }
}

// Modified checkText function to use the new retry logic
export async function checkText(text: string): Promise<TextCheckResult> {
    const gptzeroPayload = {
        document: text,
        multilingual: false
    };

    try {
        const gptzeroResult = await callApiWithRetry(gptzeroUrl, gptzeroPayload, "GPTZero");
        return {
            gptzero_result: gptzeroResult
        };
    } catch (error) {
        logger.error(`Error in execution: ${error}`);
        return { 
            gptzero_result: { error: String(error) }
        };
    }
}

// Modified main execution
if (require.main === module) {
    processTextDataSets()
        .then(() => logger.info('Processing completed'))
        .catch(error => logger.error(`Main execution error: ${error}`));
}