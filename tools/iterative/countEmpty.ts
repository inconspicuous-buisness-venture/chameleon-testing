import { readFileSync } from 'fs';
import { join } from 'path';

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

function countEmptyGPTZeroScores() {
    const filePath = join(__dirname, 'finalTextDataSets.json');
    const data: TextDataSet[] = JSON.parse(readFileSync(filePath, 'utf-8'));
    
    let emptyCount = 0;
    let totalCount = 0;

    for (const dataset of data) {
        // Check original text
        totalCount++;
        if (dataset.originalText.detectionScores.gptzero === -1) {
            emptyCount++;
        }

        // Check each model's data
        for (const modelData of Object.values(dataset.models)) {
            // Check summarized text
            totalCount++;
            if (modelData.summarizedText.detectionScores.gptzero === -1) {
                emptyCount++;
            }

            // Check iterations
            for (const iteration of modelData.iterations) {
                totalCount++;
                if (iteration.detectionScores.gptzero === -1) {
                    emptyCount++;
                }
            }
        }
    }

    console.log(`Empty GPTZero scores: ${emptyCount}/${totalCount} (${((emptyCount/totalCount)*100).toFixed(2)}%)`);
}

countEmptyGPTZeroScores();