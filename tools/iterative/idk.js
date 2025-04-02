console.log(await fetch("https://api.gptzero.me/v3/ai/text", {
    "credentials": "include",
    "headers": {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:136.0) Gecko/20100101 Firefox/136.0",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.5",
        "Content-Type": "application/json",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
        "Sec-GPC": "1",
        "Priority": "u=0"
    },
    "referrer": "https://app.gptzero.me/",
    "body": "{\"document\":\"I believe a shift towards veganism is something we should seriously consider. Consuming meat has negative consequences for both the planet and our health. For one, animal agriculture demands an immense amount of water to produce feed. Just consider the fact that producing a single pound of beef requires approximately 1,000 gallons of water. Isn't that astonishing? That water could be used to help individuals facing water scarcity. Furthermore, the land requirements for raising livestock are substantial. Cattle require a significant amount of space, and their waste contributes to soil and air pollution. A plant-based diet, on the other hand, benefits both our health and the environment. Plus, plants can be delicious! Have you ever tasted tofu? It's incredibly versatile and can be incorporated into countless dishes. I personally love tofu stir-fry with broccoli and carrots; it's absolutely delicious.\\n\",\"writing_stats_required\":false,\"interpretability_required\":false,\"multilingual\":true}",
    "method": "POST",
    "mode": "cors"
}))