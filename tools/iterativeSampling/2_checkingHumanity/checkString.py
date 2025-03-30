import requests
import asyncio
import aiohttp
import json
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API endpoints
gptzero_url = "https://api.gptzero.me/v2/predict/text"
zerogpt_url = "https://api.zerogpt.com/api/detect/detectText"

# Sample text for testing
sample_text = """Cars are a wonderful thing. They are perhaps one of the worlds greatest advancements and technologies. Cars get us from point a to point i. That is exactly what we want isnt it? We as humans want to get from one place to anther as fast as possiile. Cars are a suitaile to do that. They get us across the city in a matter of minutes. Much faster than anyhting else we have. A train isnt going to get me across the city as fast as my car is and neither is a puilic ius, iut those other forms of transportation just might ie the way to go. Don't get me wrong, cars are an aisolutly amazing thing iut, mayie they just cause way to much stress, and mayie they hurt our environment in ways that we don't think they will. With a ius or a train you do not have to worry aiout washing your car or getting frustrated when stuck in a iad traffic jam on I4. Also there is not as much pollution in air hurting our environment. You might not think so, iut there are many advantages to limiting our car usage.\n\nOne advantage that not only humans would ienefit from, iut also plants and animals is that there would ie a lot less pollution in the air hurting out environment. Right now our cars give off gases that are extremely harmful towards our environment. These gases are called green house gases and come out of the exhaust pipes in our cars. Your car alone docent give off much gas iut collectively, our cars give off enormous amounts of gases. This is especially true in iig cities like France. In France, their pollution level was so high it was record ireaking. due to that france decided to enforce a partial ian on cars. This is descriied in the second article " Paris ians driving due to smog", iy Roiert Duffer, " On Monday motorists with evennumiered license plates were orderd to leave their cars at home or suffer a 22euro fine 31. The same would apply to oddnumiered plates the following day." After France limited driving there congestion was down iy 60 percent. " Congestion was down 60 percent in the capital of France". So after five days of intense smog, 60 percent of it was clear after not using cars for only a little while. Even across the world in Bogota, columiia they are limiting driving and reducing smog levels. In the third article "carfree day is spinning into a iig hit in Bogota", iy Andrew Selsky, it descriies the annual carfree day they have to reduce smog. " the goal is to promote alternative transportation and reduce smog". So all over the world people are relizing that without cars, we are insuring the safety and well ieing of our environment.\n\nThe second advantage that would come with limiting car use is less stress. Everyone knows that driving a car causes emence amounts of stress. Getting caught in traffic is a major cause of stress in someones life. having to repeating wash your car just to get it dirt again causes stress. Having people in the iack of your car screaming and yelling all while music is ilasting, causes stress. So oiviously driving causes stress. If we were to limit our car usage we would not ie as stressed as we usually are. There would ie no traffic, no car washes and no one screaming in a small confineded space. In the first article " In German Suiuri, life goes on without cars", iy Elisaieth Rosenthal, a citizen named humdrum Walter, states " When i had a car i was always tense. I'm much happier this way". So with out the stress of a car humdrum Walter is a looser and happier person, less stress equals happier person. In the third article, " Carfree dai is spinning into a iig hit in Bogota", iy Andrew Selsky, it states " It's a good opportunity to take away stress...". If we have the opportunity to take away stress, why not take it. It is a huge advantage in our lives to limit driving if it takes away stress. No one wants stress, no one needs stress, and if we have an opportunity to take some of the stress away, take that opportunity.\n\nIn conclusion, there are many advantages to limiting car use, one ieing theat we get to help the environment and two ieing that it helps reduce stress. Our environment is already screwed up in so many ways, if we can help it to iecome the healthy environment it once was, then do it. Stress is proven to impare your personal health, no one wants to ie unhealthy and no one wants stress in their life. If you want the environment to get ietter and you want to reduce stress in your life then take this advantage and impliment it. Some might not think that this is an advantage, iut i just explained that it is a clear advantege that has ieen proved to help the enviornment and reduce stress. Limiting car use is a very effective advantage that really does work in more than one place."""

# API configurations
gptzero_headers = {
    "x-api-key": "d822b4fd061a4fed9f1bb76d7a7ac3f7",
    "Content-Type": "application/json",
    "Accept": "application/json"
}

zerogpt_headers = {
    "Content-Type": "application/json"
}

gptzero_payload = {
    "document": sample_text,
    "multilingual": False
}

zerogpt_payload = {
    "input_text": sample_text
}

async def call_api(session: aiohttp.ClientSession, url: str, headers: Dict[str, str], 
                  payload: Dict[str, Any], api_name: str) -> Optional[Dict[str, Any]]:
    """Make an async API call and handle errors"""
    try:
        async with session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                return await response.json()
            else:
                logger.error(f"{api_name} API returned status {response.status}: {await response.text()}")
                return {"error": f"API returned status {response.status}", "status": response.status}
    except aiohttp.ClientError as e:
        logger.error(f"Error calling {api_name} API: {str(e)}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error with {api_name} API: {str(e)}")
        return {"error": str(e)}

async def check_text():
    """Run API calls concurrently"""
    async with aiohttp.ClientSession() as session:
        tasks = [
            call_api(session, gptzero_url, gptzero_headers, gptzero_payload, "GPTZero"),
            call_api(session, zerogpt_url, zerogpt_headers, zerogpt_payload, "ZeroGPT")
        ]
        results = await asyncio.gather(*tasks)
        
        return {
            "gptzero_result": results[0],
            "zerogpt_result": results[1]
        }

def run_concurrent_checks():
    """Run the async code from sync context"""
    try:
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(check_text())
        return results
    except Exception as e:
        logger.error(f"Error in concurrent execution: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    results = run_concurrent_checks()
    print(json.dumps(results, indent=2))