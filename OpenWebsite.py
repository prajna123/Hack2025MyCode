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
                browser = await p.chromium.launch(headless=False, executable_path="D:/Playwright_Browsers/chromium-1161/chrome-win/chrome.exe")
                #browser = await p.chromium.launch(headless=False)
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
        
    @staticmethod
    async def search_splunk():
        async with async_playwright() as p:
            # index = data.get("index", "N/A")  # Default to "N/A" if key is missing
            # host = data.get("host", "N/A")
            # source = data.get("source", "N/A")
            # sourcetype = data.get("sourcetype", "N/A")

            # print("Index:", index)
            # print("Host:", host)
            # print("Source:", source)
            # print("Sourcetype:", sourcetype)

            # query_parts = []  # List to hold query key-value pairs

            # for key, value in data.items():
            #     if value:  # Only include non-empty values
            #         query_parts.append(f'{key}="{value}"')
            #         print(f"Key: {key}, Value: {value}")
            #         print("Query parts:", query_parts)

            # print("Final Query parts:", query_parts)
            # search_query = " ".join(query_parts)

            browser = await p.chromium.launch(headless=False, executable_path="D:/Playwright_Browsers/chromium-1161/chrome-win/chrome.exe", timeout=60000)
            page = await browser.new_page()
        
            # Open Splunk URL
            await page.goto("http://127.0.0.1:8000/en-US/app/search/search")
        
            # Login
            await page.fill("input[name='username']", "Hackathon2025")  # Change to your username
            await page.fill("input[name='password']", "Hack@2025")  # Change to your password
            await page.press("input[name='password']", "Enter")

            # await page.wait_for_selector("button[type='submit']", state="visible")
            # await page.click("button[type='submit']", force=True)

            #await page.wait_for_selector("button[value='Sign In']", state="visible")
            #await page.click("button[value='Sign In']", force=True)
        
            # Wait for page load after login
            #page.wait_for_selector("div.ace_invisible.ace_emptyMessage", timeout=5000)
            #search_element = page.locator("div.ace_invisible.ace_emptyMessage")
            search_element = page.locator("text='enter search here...'")
            await search_element.wait_for(state="visible", timeout=5000)
        
            # Enter search query
            #search_query = 'source="access.log" host="www2" index="web" sourcetype="access_combined"'

            is_ace_editor = await page.evaluate("() => !!window.ace")

            # if is_ace_editor:
            #     print("Ace Editor detected!")
            #     await page.evaluate("""
            #         let editor = ace.edit(document.querySelector('.ace_editor'));
            #         editor.setValue('source="access.log" host="www2" index="web" sourcetype="access_combined"');
            #         editor.session.getSelection().clearSelection();  // Ensure text is registered
            #         editor.focus();  // Bring focus to editor
            #         editor.renderer.updateFull();  // Force UI update
            #         editor.session.setScrollTop(0);  // Ensure query is visible
            # """)
            # else:
            #     print("Ace Editor not detected! Try another method.")

            # query_text = await page.evaluate("""
            #     return ace.edit(document.querySelector('.ace_editor')).getValue();
            # """)
            # print("Entered Query:", query_text)

            search_box = page.locator(".ace_editor")  # Adjust selector if needed
            await search_box.click()
            await page.keyboard.type('source="access.log" host="www2" index="web" sourcetype="access_combined"')
            #await page.keyboard.press("Enter") 

            
            #await search_element.fill(search_query)
            #await page.fill("#searchElement", search_query)

            # await search_element.click()
            # await page.keyboard.type(search_query)
        
            # Set time range to 'All Time'

            await page.locator("span.time-label").click()
            await page.locator("text='All time (real-time)'").wait_for(state="visible", timeout=5000)  # Ensure dropdown is visible
            await page.locator("text='All time (real-time)'").click()

            await page.locator("[aria-label='Search Button']").click()


            try:
                await page.locator("div.search-results-events-container").wait_for(state="visible", timeout=30000)
                print("Data found!")  # Replace with actual data extraction logic
            except Exception as e:
                print("No data found")
                return f"❌ No Data Found"
            finally:
                    await browser.close()
            
            print("Search results loaded!")

            table = await page.locator("//table[contains(@class, 'events-results-table')]")
            row_count = await table.locator("tr").count()

            print("Row count: ", row_count)
        
            # Check if results appear
            if row_count > 0:
                print("Data is coming in search results!")
            else:
                print("No data found.")
        
            # Close browser
            browser.close()