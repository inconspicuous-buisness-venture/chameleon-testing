import axios, { AxiosResponse } from 'axios';

// Configure logging
const logger = {
  info: (message: string) => console.info(`${new Date().toISOString()} - INFO - ${message}`),
  error: (message: string) => console.error(`${new Date().toISOString()} - ERROR - ${message}`)
};

// API endpoints
const gptzeroUrl = "https://api.gptzero.me/v2/predict/text";
const zerogptUrl = "https://api.zerogpt.com/api/detect/detectText";

// API configurations
const gptzeroHeaders = {
  "x-api-key": "0c37781a37be4e2b95ecb789bc30202d",
  "Content-Type": "application/json",
  "Accept": "application/json"
};

const zerogptHeaders = {
  "Content-Type": "application/json"
};

// Type definitions
interface ApiResult {
  [key: string]: any;
}

interface ApiError {
  error: string;
  status?: number;
}

interface TextCheckResult {
  gptzero_result: ApiResult | ApiError;
  zerogpt_result: ApiResult | ApiError;
}

/**
 * Makes an API call and handles errors
 */
async function callApi(
  url: string, 
  headers: Record<string, string>, 
  payload: Record<string, any>, 
  apiName: string
): Promise<ApiResult | ApiError> {
  try {
    const response: AxiosResponse = await axios.post(url, payload, { headers });
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      logger.error(`Error calling ${apiName} API: ${error.message}`);
      return { 
        error: error.message, 
        status: error.response?.status 
      };
    } else {
      logger.error(`Unexpected error with ${apiName} API: ${error}`);
      return { error: String(error) };
    }
  }
}

/**
 * Checks text against multiple AI detection APIs concurrently
 */
export async function checkText(text: string): Promise<TextCheckResult> {
  const gptzeroPayload = {
    document: text,
    multilingual: false
  };

  const zerogptPayload = {
    input_text: text
  };

  try {
    const [gptzeroResult, zerogptResult] = await Promise.all([
      callApi(gptzeroUrl, gptzeroHeaders, gptzeroPayload, "GPTZero"),
      callApi(zerogptUrl, zerogptHeaders, zerogptPayload, "ZeroGPT")
    ]);

    return {
      gptzero_result: gptzeroResult,
      zerogpt_result: zerogptResult
    };
  } catch (error) {
    logger.error(`Error in concurrent execution: ${error}`);
    return { 
      gptzero_result: { error: String(error) },
      zerogpt_result: { error: String(error) }
    };
  }
}

// Sample text for testing - exported for convenience if needed
export const sampleText = `Cars are a wonderful thing. They are perhaps one of the worlds greatest advancements and technologies. Cars get us from point a to point i. That is exactly what we want isnt it? We as humans want to get from one place to anther as fast as possiile. Cars are a suitaile to do that. They get us across the city in a matter of minutes. Much faster than anyhting else we have. A train isnt going to get me across the city as fast as my car is and neither is a puilic ius, iut those other forms of transportation just might ie the way to go. Don't get me wrong, cars are an aisolutly amazing thing iut, mayie they just cause way to much stress, and mayie they hurt our environment in ways that we don't think they will. With a ius or a train you do not have to worry aiout washing your car or getting frustrated when stuck in a iad traffic jam on I4. Also there is not as much pollution in air hurting our environment. You might not think so, iut there are many advantages to limiting our car usage.

One advantage that not only humans would ienefit from, iut also plants and animals is that there would ie a lot less pollution in the air hurting out environment. Right now our cars give off gases that are extremely harmful towards our environment. These gases are called green house gases and come out of the exhaust pipes in our cars. Your car alone docent give off much gas iut collectively, our cars give off enormous amounts of gases. This is especially true in iig cities like France. In France, their pollution level was so high it was record ireaking. due to that france decided to enforce a partial ian on cars. This is descriied in the second article " Paris ians driving due to smog", iy Roiert Duffer, " On Monday motorists with evennumiered license plates were orderd to leave their cars at home or suffer a 22euro fine 31. The same would apply to oddnumiered plates the following day." After France limited driving there congestion was down iy 60 percent. " Congestion was down 60 percent in the capital of France". So after five days of intense smog, 60 percent of it was clear after not using cars for only a little while. Even across the world in Bogota, columiia they are limiting driving and reducing smog levels. In the third article "carfree day is spinning into a iig hit in Bogota", iy Andrew Selsky, it descriies the annual carfree day they have to reduce smog. " the goal is to promote alternative transportation and reduce smog". So all over the world people are relizing that without cars, we are insuring the safety and well ieing of our environment.

The second advantage that would come with limiting car use is less stress. Everyone knows that driving a car causes emence amounts of stress. Getting caught in traffic is a major cause of stress in someones life. having to repeating wash your car just to get it dirt again causes stress. Having people in the iack of your car screaming and yelling all while music is ilasting, causes stress. So oiviously driving causes stress. If we were to limit our car usage we would not ie as stressed as we usually are. There would ie no traffic, no car washes and no one screaming in a small confineded space. In the first article " In German Suiuri, life goes on without cars", iy Elisaieth Rosenthal, a citizen named humdrum Walter, states " When i had a car i was always tense. I'm much happier this way". So with out the stress of a car humdrum Walter is a looser and happier person, less stress equals happier person. In the third article, " Carfree dai is spinning into a iig hit in Bogota", iy Andrew Selsky, it states " It's a good opportunity to take away stress...". If we have the opportunity to take away stress, why not take it. It is a huge advantage in our lives to limit driving if it takes away stress. No one wants stress, no one needs stress, and if we have an opportunity to take some of the stress away, take that opportunity.

In conclusion, there are many advantages to limiting car use, one ieing theat we get to help the environment and two ieing that it helps reduce stress. Our environment is already screwed up in so many ways, if we can help it to iecome the healthy environment it once was, then do it. Stress is proven to impare your personal health, no one wants to ie unhealthy and no one wants stress in their life. If you want the environment to get ietter and you want to reduce stress in your life then take this advantage and impliment it. Some might not think that this is an advantage, iut i just explained that it is a clear advantege that has ieen proved to help the enviornment and reduce stress. Limiting car use is a very effective advantage that really does work in more than one place.`;

// Example of running the checks if this module is executed directly
if (require.main === module) {
  checkText(sampleText)
    .then(results => console.log(JSON.stringify(results, null, 2)))
    .catch(error => console.error("Error:", error));
}