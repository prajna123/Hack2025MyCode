from playwright.async_api import async_playwright
import subprocess  



class OpenWebsite:
    @staticmethod
    def check_network_connectivity(host="8.8.8.8"):  # Google's public DNS
        try:
            subprocess.run(["ping", host], check=True)
            print("Network is reachable")
            return "✅ Network is reachable!"
        except subprocess.CalledProcessError:
            print("Network is unreachable")
            return "❌ Network is unreachable!"



    @staticmethod
    async def open_website():
        urls = ["https://www.google.com", "https://www.google.com/maps", "https://news.google.com"]
        successresponse = ""
        failresponse = ""
        for url in urls:
            async with async_playwright() as p:
                print("sync_playwright started:")
                #browser = await p.chromium.launch(headless=False, executable_path="D:/Playwright_Browsers/chromium-1161/chrome-win/chrome.exe")
                browser = await p.chromium.launch(headless=False)
                print("Browser launched")
                page = await browser.new_page()
                print("Browser new Page")
                try:
                    await page.goto(url)
                    successresponse = successresponse + "Successfully opened {url}!"
                    #return f"✅ Successfully opened {url}!"
                except Exception as e:
                    failresponse = failresponse + "Successfully opened {url}!"
                    #return f"❌ Failed to open {url}: {e}"
                finally:
                    await browser.close()
        
        if successresponse: 
            print("String is not empty")  # ✅ Output: String is not empty
            return f"✅ Successfully opened {urls}!"

        if failresponse:
            print("String is empty")
            return f"❌ Failed to open {urls}: {e}"