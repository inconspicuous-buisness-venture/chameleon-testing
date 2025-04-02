import { checkText, sampleText } from './modules/check';

// Example usage
async function detectAIText() {
  const myText = "Okay, so you're ditching the car? Nice one! But get this, it's not just good for *you*, ya know? Loads of studies show that when we drive less, cool stuff happens. Think about awesome communities popping up all over the place. Plus, we'd see a HUGE drop in those nasty greenhouse gases messing with our planet. And hey, picture this: super-strong neighborhoods that everyone wants to live in!";
  const results = await checkText(myText);
  console.log(results.gptzero_result.documents);
}

detectAIText();