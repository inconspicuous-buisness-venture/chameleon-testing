import asyncio
from playwright.async_api import async_playwright
import re

async def delay(time_seconds):
    await asyncio.sleep(time_seconds)

async def main(input_text):
    
    print('\n\nScraping Detection Results for ZeroGPT (The easy one)... this may take around 5 seconds\n\n')
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        await page.set_viewport_size({"width": 1303, "height": 949})
        
        await page.goto('https://www.zerogpt.com/')
        
        await page.wait_for_selector('#textArea')
        await page.click('#textArea')

        await delay(1)

        try: 
            await page.type('#textArea', input_text)
        except:
            print("bruh it didnt work")
        
        await page.wait_for_selector('div.features-and-textarea button')
        await page.click('div.features-and-textarea button')
        
        await delay(4)
        
        header_text = await page.evaluate('''() => {
            const elements = document.querySelectorAll('.header-text.text-center');
            return Array.from(elements).map(el => el.textContent).join(' ');
        }''')
        
        numbers_only = re.sub(r'\D', '', header_text)
        print(numbers_only)
        
        await browser.close()

        return numbers_only