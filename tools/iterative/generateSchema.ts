import { readFileSync, writeFileSync } from 'fs';
import { join } from 'path';

interface DetectionScores {
    gptzero: number;
    zerogpt: number;
    roberta: number;
}

interface Text {
    text: string;
    detectionScores: DetectionScores;
    wordFrequency: {
        wordCounts: { [word: string]: number };
        frequencyDistribution: { [count: number]: number };
        totalWords: number;
    };
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

const schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "TextDataSets Schema",
    "type": "array",
    "items": {
        "type": "object",
        "required": ["id", "originalText", "models"],
        "properties": {
            "id": {
                "type": "number"
            },
            "originalText": {
                "type": "object",
                "required": ["text", "detectionScores", "wordFrequency"],
                "properties": {
                    "text": {
                        "type": "string"
                    },
                    "detectionScores": {
                        "type": "object",
                        "required": ["gptzero", "zerogpt", "roberta"],
                        "properties": {
                            "gptzero": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1
                            },
                            "zerogpt": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1
                            },
                            "roberta": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1
                            }
                        }
                    },
                    "wordFrequency": {
                        "type": "object",
                        "required": ["wordCounts", "frequencyDistribution", "totalWords"],
                        "properties": {
                            "wordCounts": {
                                "type": "object",
                                "additionalProperties": {
                                    "type": "number"
                                }
                            },
                            "frequencyDistribution": {
                                "type": "object",
                                "additionalProperties": {
                                    "type": "number"
                                }
                            },
                            "totalWords": {
                                "type": "number",
                                "minimum": 0
                            }
                        }
                    }
                }
            },
            "models": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "required": ["modelName", "summarizedText", "iterations"],
                    "properties": {
                        "modelName": {
                            "type": "string"
                        },
                        "summarizedText": {
                            "$ref": "#/items/properties/originalText"
                        },
                        "iterations": {
                            "type": "array",
                            "items": {
                                "allOf": [
                                    { "$ref": "#/items/properties/originalText" },
                                    {
                                        "type": "object",
                                        "required": ["iteration"],
                                        "properties": {
                                            "iteration": {
                                                "type": "number",
                                                "minimum": 0
                                            }
                                        }
                                    }
                                ]
                            }
                        }
                    }
                }
            }
        }
    }
};

function main() {
    const outputPath = join(__dirname, 'textDataSets.schema.json');
    writeFileSync(outputPath, JSON.stringify(schema, null, 2));
    console.log(`Schema generated and saved to ${outputPath}`);
}

main();