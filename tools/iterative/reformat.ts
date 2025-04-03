import gemini2Data from '../iterativeSampling/3_checkingHumanity/data/gemini2FlashIterationsWithScores.json';
import gemini15Data from '../iterativeSampling/3_checkingHumanity/data/gemini15ProIterationsWithScores.json';
import gpt4oData from '../iterativeSampling/3_checkingHumanity/data/GPT4oIterationsWithScores.json';

interface WordFrequency {
	wordCounts: { [word: string]: number };
	frequencyDistribution: { [count: number]: number };
	totalWords: number;
}

interface DetectionScores {
	gptzero: number;
	zerogpt: number;
	roberta: number;
}

interface Text {
	text: string;
	detectionScores: DetectionScores;
	wordFrequency: WordFrequency;
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

function calculateWordFrequency(text: string): WordFrequency {
	const words = text.toLowerCase()
		.split(/\s+/)
		.map(word => word.replace(/^[^\w']+|[^\w']+$/g, ''))
		.filter(word => word.length > 0);

	const wordCounts: { [word: string]: number } = {};
	for (const word of words) {
		wordCounts[word] = (wordCounts[word] || 0) + 1;
	}

	const frequencyDistribution: { [count: number]: number } = {};
	Object.values(wordCounts).forEach(count => {
		frequencyDistribution[count] = (frequencyDistribution[count] || 0) + 1;
	});

	return {
		wordCounts,
		frequencyDistribution,
		totalWords: words.length
	};
}

const originalTextsRaw: string[] = [];
const originalMetricsRaw: DetectionScores[] = [];

for (const tds of gemini15Data) originalTextsRaw.push(tds.original_text);
// console.log(originalTextsRaw.length)

for (const tds of gemini15Data) {
	originalMetricsRaw.push({
		gptzero: Number((-1).toFixed(4)),
		zerogpt: Number((tds.original_text_metrics.detection_scores.zerogpt / 100).toFixed(4)),
		roberta: Number((tds.original_text_metrics.detection_scores.roberta / 100).toFixed(4)),
	});
}

const textDataSets: TextDataSet[] = [];
const models: string[] = ['gemini2', 'gemini15', 'gpt4o'];

for (let i = 0; i < originalTextsRaw.length; i++) {
	const originalText = originalTextsRaw[i];
	const textDataSet: TextDataSet = {
		id: i + 1,
		originalText: {
			text: originalText,
			detectionScores: originalMetricsRaw[i],
			wordFrequency: calculateWordFrequency(originalText)
		},
		models: {},
	};
	for (const model of models) {
		let modelDataRaw: any = null;
		let modelName: string = '';
		if (model === 'gemini2') {
			modelDataRaw = gemini2Data.find((data) => data.id === i + 1);
			modelName = 'gemini-2.0-flash';
		} else if (model === 'gemini15') {
			modelDataRaw = gemini15Data.find((data) => data.id === i + 1);
			modelName = 'gemini-1.5-pro';
		} else if (model === 'gpt4o') {
			modelDataRaw = gpt4oData.find((data) => data.id === i + 1);
			modelName = 'gpt-4o';
		}
		// if (modelData) {
		//  textDataSet.models[model] = modelData;
		// }

		// console.log('modelDataRaw', modelDataRaw);
		// console.log('modelName', modelName);
		// console.log('i', i);
		const summarizedText: Text = {
			text: modelDataRaw.summarized_text,
			detectionScores: {
				gptzero: Number((-1).toFixed(4)),
				zerogpt: Number((modelDataRaw.summarized_text_metrics.detection_scores.zerogpt / 100).toFixed(4)),
				roberta: Number((modelDataRaw.summarized_text_metrics.detection_scores.roberta / 100).toFixed(4)),
			},
			wordFrequency: calculateWordFrequency(modelDataRaw.summarized_text)
		};

		const iterations: Iteration[] = [];
		for (const iteration of modelDataRaw.iterations) {
			// console.log('iteration', iteration);
			iterations.push({
				iteration: iteration.iteration_number,
				text: iteration.rewritten_text,
				detectionScores: {
					gptzero: Number((-1).toFixed(4)),
					zerogpt: Number((iteration.detection_scores.zerogpt / 100).toFixed(4)),
					roberta: Number((iteration.detection_scores.roberta / 100).toFixed(4)),
				},
				wordFrequency: calculateWordFrequency(iteration.rewritten_text)
			});
		}

		const modelData: Model = {
			modelName: modelName,
			summarizedText: summarizedText,
			iterations: iterations,
		};
		textDataSet.models[modelName] = modelData;
	}
	textDataSets.push(textDataSet);
}

import { writeFileSync } from 'fs';
import { join } from 'path';

const outputPath = join(__dirname, 'textDataSets.json');
writeFileSync(outputPath, JSON.stringify(textDataSets, null, 4), 'utf-8');
console.log('File written successfully:', outputPath);
// console.log(textDataSets.length);
