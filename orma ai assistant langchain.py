import os
import requests
import json
from ddgs import DDGS  # Updated import
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import speech_recognition as sr
import pyttsx3
import threading
import time

# Configuration
os.environ["OPENROUTER_API_KEY"] = "YOUR API KEY HERE"

class VoiceEnabledAIAssistant:
    def __init__(self):
        self.api_key = os.environ["OPENROUTER_API_KEY"]
        self.model = "deepseek/deepseek-r1:free"  # DeepSeek R1 free model
        self.conversation_history = []
        
        # Initialize voice components
        self.setup_voice()
        
    def setup_voice(self):
        """Initialize speech recognition and text-to-speech"""
        try:
            # Initialize speech recognition
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            # Initialize text-to-speech
            self.tts_engine = pyttsx3.init()
            
            # Initialize thread lock for TTS
            self.tts_lock = threading.Lock()
            
            # Configure TTS for calming voice
            self.configure_calming_voice()
            
            # Adjust microphone for ambient noise
            print("🎤 Adjusting microphone for ambient noise... Please wait.")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
            print("✅ Voice setup complete!")
            
        except Exception as e:
            print(f"⚠️ Voice setup error: {e}")
            print("Voice features may not work properly.")
            self.tts_engine = None
            self.recognizer = None
    
    def configure_calming_voice(self):
        """Configure TTS engine for a calming voice"""
        if self.tts_engine:
            # Get available voices
            voices = self.tts_engine.getProperty('voices')
            
            # Try to find a female voice (generally perceived as more calming)
            female_voice = None
            for voice in voices:
                if 'female' in voice.name.lower() or 'zira' in voice.name.lower() or 'susan' in voice.name.lower():
                    female_voice = voice
                    break
            
            # Set voice (female if available, otherwise first available)
            if female_voice:
                self.tts_engine.setProperty('voice', female_voice.id)
                print(f"🎵 Using calming voice: {female_voice.name}")
            else:
                # Use first available voice and configure it to be calming
                if voices:
                    self.tts_engine.setProperty('voice', voices[0].id)
                    print(f"🎵 Using voice: {voices[0].name}")
            
            # Set calming speech parameters
            self.tts_engine.setProperty('rate', 160)    # Slower speech rate (default ~200)
            self.tts_engine.setProperty('volume', 0.8)  # Slightly lower volume
            
            # Test the voice
            print("🔊 Testing voice...")
            self.speak("Hello! I'm your AI assistant. How can I help you today?")
    
    def speak(self, text):
        """Convert text to speech with calming voice"""
        if self.tts_engine:
            try:
                # Use a lock to prevent concurrent TTS calls
                if not hasattr(self, 'tts_lock'):
                    self.tts_lock = threading.Lock()
                
                with self.tts_lock:
                    # Stop any current speech
                    self.tts_engine.stop()
                    # Add the text and run
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                
            except Exception as e:
                print(f"🔇 TTS Error: {e}")
                # Try alternative approach if main method fails
                try:
                    import subprocess
                    import platform
                    
                    # Fallback to system TTS
                    if platform.system() == "Windows":
                        # Use Windows SAPI - simplified version
                        clean_text = text.replace("'", "").replace('"', '')
                        cmd = f'powershell -Command "Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Speak(\'{clean_text}\')"'
                        subprocess.run(cmd, shell=True, capture_output=True)
                except:
                    pass  # Silent fallback failure
    
    def listen(self, timeout=5, phrase_time_limit=10):
        """Listen for voice input"""
        if not self.recognizer or not self.microphone:
            return None
            
        try:
            print("🎤 Listening... (speak now)")
            with self.microphone as source:
                # Listen for audio with timeout
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            
            print("🔄 Processing speech...")
            
            # Use Google's speech recognition (free)
            text = self.recognizer.recognize_google(audio)
            print(f"👂 You said: {text}")
            return text
            
        except sr.WaitTimeoutError:
            print("⏰ Listening timeout. No speech detected.")
            return None
        except sr.UnknownValueError:
            print("❓ Sorry, I couldn't understand what you said.")
            self.speak("Sorry, I couldn't understand what you said. Could you please repeat?")
            return None
        except sr.RequestError as e:
            print(f"🌐 Speech recognition service error: {e}")
            self.speak("I'm having trouble with speech recognition right now.")
            return None
        except Exception as e:
            print(f"🎤 Voice input error: {e}")
            return None
    
    def call_model(self, messages):
        """Call the OpenRouter API"""
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "your-site-url",  # Optional
                    "X-Title": "Voice AI Assistant"  # Optional
                },
                data=json.dumps({
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 1000,
                    "temperature": 0.7
                }),
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                return f"API Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error calling model: {str(e)}"
    
    def web_search(self, query):
        """Perform web search using ddgs"""
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))
                
            if not results:
                return f"No search results found for '{query}'"
            
            # Format the results
            formatted_results = f"Search results for '{query}':\n\n"
            for i, result in enumerate(results, 1):
                formatted_results += f"{i}. {result['title']}\n"
                formatted_results += f"   {result['body']}\n"
                formatted_results += f"   URL: {result['href']}\n\n"
            
            return formatted_results
            
        except Exception as e:
            return f"Search error: {str(e)}"
    
    def analyze_pdf(self, file_path, question):
        """Analyze PDF content"""
        try:
            if not os.path.exists(file_path):
                return f"Error: PDF file '{file_path}' not found. Please make sure the file exists in the current directory."
            
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            
            if not docs:
                return "Error: Could not load PDF content. Make sure it's a valid PDF file."
            
            # Combine all PDF content
            pdf_content = "\n".join([doc.page_content for doc in docs[:5]])  # Increased to first 5 pages
            
            # Create messages for the model WITH conversation context
            messages = [
                {"role": "system", "content": "You are a helpful assistant that answers questions about PDF documents. Provide detailed and accurate answers based on the content provided. Use conversation history for context."}
            ] + self.conversation_history[-4:] + [  # Include last 2 exchanges for context
                {
                    "role": "user", 
                    "content": f"Based on this PDF content from '{file_path}', please answer the question.\n\nPDF Content:\n{pdf_content[:4000]}\n\nQuestion: {question}"
                }
            ]
            
            return self.call_model(messages)
            
        except Exception as e:
            return f"Error processing PDF: {str(e)}"
    
    def extract_pdf_path(self, user_input):
        """Extract PDF path from user input"""
        # Look for common path patterns
        import re
        
        # Pattern 1: Path in quotes
        quote_pattern = r'["\']([^"\']*\.pdf)["\']'
        match = re.search(quote_pattern, user_input, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # Pattern 2: Path starting with drive letter (Windows) or / (Linux/Mac)
        path_pattern = r'([A-Za-z]:[\\\/][^\\\/\s]*\.pdf|\/[^\/\s]*\.pdf|[\\\/][^\\\/\s]*\.pdf)'
        match = re.search(path_pattern, user_input, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # Pattern 3: Simple filename.pdf
        filename_pattern = r'(\S+\.pdf)'
        match = re.search(filename_pattern, user_input, re.IGNORECASE)
        if match:
            return match.group(1)
        
        return None
    
    def extract_image_path_or_url(self, user_input):
        """Extract image path or URL from user input"""
        import re
        
        # Pattern 1: HTTP/HTTPS URLs
        url_pattern = r'(https?://[^\s]+\.(?:jpg|jpeg|png|gif|bmp|webp|svg))'
        url_match = re.search(url_pattern, user_input, re.IGNORECASE)
        if url_match:
            return url_match.group(1), 'url'
        
        # Pattern 2: Path in quotes
        quote_pattern = r'["\']([^"\']*\.(?:jpg|jpeg|png|gif|bmp|webp|svg))["\']'
        path_match = re.search(quote_pattern, user_input, re.IGNORECASE)
        if path_match:
            return path_match.group(1), 'path'
        
        # Pattern 3: Path starting with drive letter (Windows) or / (Linux/Mac)
        path_pattern = r'([A-Za-z]:[\\\/][^\\\/\s]*\.(?:jpg|jpeg|png|gif|bmp|webp|svg)|\/[^\/\s]*\.(?:jpg|jpeg|png|gif|bmp|webp|svg)|[\\\/][^\\\/\s]*\.(?:jpg|jpeg|png|gif|bmp|webp|svg))'
        path_match = re.search(path_pattern, user_input, re.IGNORECASE)
        if path_match:
            return path_match.group(1), 'path'
        
        # Pattern 4: Simple filename with image extension
        filename_pattern = r'(\S+\.(?:jpg|jpeg|png|gif|bmp|webp|svg))'
        filename_match = re.search(filename_pattern, user_input, re.IGNORECASE)
        if filename_match:
            return filename_match.group(1), 'path'
        
        return None, None
    
    def convert_local_image_to_data_url(self, image_path):
        """Convert local image to data URL for API"""
        try:
            import base64
            from PIL import Image
            import io
            
            # Normalize and expand path
            image_path = image_path.strip('\'"')
            image_path = os.path.normpath(image_path)
            
            if image_path.startswith('~'):
                image_path = os.path.expanduser(image_path)
            elif not os.path.isabs(image_path):
                image_path = os.path.abspath(image_path)
            
            if not os.path.exists(image_path):
                return None, f"Error: Image file '{image_path}' not found."
            
            # Open and convert image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'LA'):
                    img = img.convert('RGB')
                
                # Resize if too large (to save tokens)
                max_size = (1024, 1024)
                if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # Convert to base64
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                img_str = base64.b64encode(buffer.getvalue()).decode()
                
                return f"data:image/jpeg;base64,{img_str}", None
                
        except Exception as e:
            return None, f"Error processing image: {str(e)}"
    
    def analyze_image(self, image_source, question, source_type='url'):
        """Analyze image from URL or local path"""
        try:
            if source_type == 'path':
                # Convert local image to data URL
                data_url, error = self.convert_local_image_to_data_url(image_source)
                if error:
                    return error
                image_url = data_url
                source_info = f"local file '{image_source}'"
            else:
                image_url = image_source
                source_info = f"URL '{image_source}'"
            
            # Try with a vision-capable model WITH conversation context
            vision_model = "deepseek/deepseek-r1:free"  # Try with same model first
            
            # Prepare messages with context
            context_messages = []
            if self.conversation_history:
                context_messages = [
                    {"role": "system", "content": "You are a helpful AI assistant analyzing images. Use conversation history for context."}
                ] + self.conversation_history[-4:]  # Last 2 exchanges for context
            
            current_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Analyzing image from {source_info}. {question}"},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
            
            all_messages = context_messages + [current_message]
            
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "your-site-url",  # Optional
                    "X-Title": "Voice AI Assistant"  # Optional
                },
                data=json.dumps({
                    "model": vision_model,
                    "messages": all_messages,
                    "max_tokens": 1000
                }),
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                # Fallback: try with a different vision model
                try:
                    fallback_response = requests.post(
                        url="https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        data=json.dumps({
                            "model": "google/gemini-pro-vision",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": f"Analyzing image from {source_info}. {question}"},
                                        {"type": "image_url", "image_url": {"url": image_url}}
                                    ]
                                }
                            ],
                            "max_tokens": 1000
                        }),
                        timeout=30
                    )
                    
                    if fallback_response.status_code == 200:
                        return fallback_response.json()['choices'][0]['message']['content']
                    else:
                        return f"Image analysis not available. Error: {response.status_code}"
                        
                except:
                    return f"Image analysis not available with current model. Error: {response.status_code}"
                
        except Exception as e:
            return f"Error analyzing image: {str(e)}"
    
    def process_query(self, user_input):
        """Process user query and determine what action to take"""
        user_input_lower = user_input.lower()
        
        # Check if user wants to search
        if any(keyword in user_input_lower for keyword in ['search', 'find', 'look up', 'current', 'news', 'recent']):
            search_query = user_input.replace('search for', '').replace('find', '').replace('look up', '').strip()
            search_results = self.web_search(search_query)
            
            # Now ask the model to summarize the search results WITH conversation context
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant. Use the conversation history to provide contextual responses."}
            ] + self.conversation_history + [
                {"role": "user", "content": f"Please summarize and answer based on these search results:\n\n{search_results}\n\nOriginal question: {user_input}"}
            ]
            return self.call_model(messages)
        
        # Check if user wants PDF analysis
        elif 'pdf' in user_input_lower:
            pdf_path = self.extract_pdf_path(user_input)
            
            if pdf_path is None:
                return """📄 **PDF Analysis Help:**
                
To analyze a PDF, include the file path in your message. Examples:

**Windows:**
- `analyze "C:\\Users\\YourName\\Documents\\report.pdf" what is the summary?`
- `what does C:/Documents/file.pdf say about sales?`

**Mac/Linux:**
- `analyze "/home/user/documents/report.pdf" what are the key points?`
- `what does ~/Downloads/document.pdf contain?`

**Current folder:**
- `analyze document.pdf what is it about?`

Just mention the PDF path and ask your question!"""
            
            else:
                # Clean up the path (remove extra quotes, normalize slashes)
                pdf_path = pdf_path.strip('\'"')
                pdf_path = os.path.normpath(pdf_path)
                
                # Handle relative paths and tilde expansion
                if pdf_path.startswith('~'):
                    pdf_path = os.path.expanduser(pdf_path)
                elif not os.path.isabs(pdf_path):
                    pdf_path = os.path.abspath(pdf_path)
                
                return self.analyze_pdf(pdf_path, user_input)
        
        # Check if user wants image analysis
        elif 'image' in user_input_lower or 'picture' in user_input_lower or 'photo' in user_input_lower:
            image_source, source_type = self.extract_image_path_or_url(user_input)
            
            if image_source is None:
                return """🖼️ **Image Analysis Help:**
                
To analyze an image, include the image path or URL in your message. Examples:

**Image URLs:**
- `analyze https://example.com/image.jpg what do you see?`
- `describe this image: https://site.com/photo.png`

**Windows paths:**
- `analyze "C:\\Users\\Name\\Pictures\\photo.jpg" what's in this image?`
- `what does C:/Images/screenshot.png show?`

**Mac/Linux paths:**
- `analyze "/home/user/pictures/image.jpg" describe this`
- `what's in ~/Desktop/photo.png?`

**Current folder:**
- `analyze image.jpg what do you see?`

**Supported formats:** JPG, JPEG, PNG, GIF, BMP, WEBP, SVG

Just mention the image path/URL and ask your question!"""
            
            else:
                return self.analyze_image(image_source, user_input, source_type)
        
        # Regular conversation WITH FULL MEMORY
        else:
            # Add system message and conversation history
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant. Remember our conversation and provide contextual responses based on our chat history."}
            ] + self.conversation_history + [
                {"role": "user", "content": user_input}
            ]
            return self.call_model(messages)
    
    def voice_chat_loop(self):
        """Voice-only chat loop"""
        print("\n🎤 Voice Chat Mode Active!")
        print("- Say 'stop voice mode' to return to text mode")
        print("- Say 'exit' or 'quit' to end conversation")
        print("- The assistant will speak all responses")
        
        self.speak("Voice chat mode is now active. How can I help you?")
        
        while True:
            try:
                # Listen for voice input
                user_input = self.listen(timeout=10, phrase_time_limit=15)
                
                if user_input is None:
                    continue
                    
                # Check for mode switches
                if any(phrase in user_input.lower() for phrase in ['stop voice mode', 'text mode', 'switch to text']):
                    print("📝 Switching back to text mode...")
                    self.speak("Switching back to text mode.")
                    break
                
                if user_input.lower() in ["exit", "quit", "goodbye"]:
                    print("👋 Goodbye!")
                    self.speak("Goodbye! Have a wonderful day!")
                    return True  # Signal to exit main program
                
                # Process the query
                print("🤔 Processing...")
                response = self.process_query(user_input)
                
                # Display and speak the response
                print(f"🤖 Assistant: {response}\n")
                self.speak(response)
                
                # Update conversation history
                self.conversation_history.append({"role": "user", "content": user_input})
                self.conversation_history.append({"role": "assistant", "content": response})
                
                # Keep only last 10 exchanges
                if len(self.conversation_history) > 20:
                    self.conversation_history = self.conversation_history[-20:]
                
            except KeyboardInterrupt:
                print("\n📝 Returning to text mode...")
                self.speak("Returning to text mode.")
                break
            except Exception as e:
                print(f"❌ Voice chat error: {e}")
                self.speak("I encountered an error. Let me try again.")
        
        return False  # Don't exit main program
    
    def chat(self):
        """Main chat loop with voice capabilities"""
        print("🤖 Voice-Enabled AI Assistant with DeepSeek R1 initialized!")
        print("Available capabilities:")
        print("- 💬 General conversation (with full memory)")
        print("- 🎤 Voice input and 🔊 voice output (calming voice)")
        print("- 🔍 Web search (use keywords like 'search', 'find', 'current news')")
        print("- 📄 PDF analysis (specify full path to any PDF file)")
        print("- 🖼️  Image analysis (specify image path or URL)")
        print("- 🧠 Remembers conversation context across all interactions")
        print("\nCommands:")
        print("- Type 'voice' or 'voice mode' to switch to voice-only mode")
        print("- Type 'exit' or 'quit' to end the conversation")
        print("- Type 'clear' to clear conversation history")
        print("- Type 'memory' to see current conversation history")
        print("- For PDF analysis, include the full file path in your question")
        print("- For image analysis, include the image path or URL in your question\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ["exit", "quit", ""]:
                    print("👋 Goodbye!")
                    self.speak("Goodbye! Have a wonderful day!")
                    break
                
                if user_input.lower() in ["voice", "voice mode"]:
                    should_exit = self.voice_chat_loop()
                    if should_exit:
                        break
                    continue
                
                if user_input.lower() == "clear":
                    self.conversation_history = []
                    print("🧹 Conversation history cleared!")
                    self.speak("Conversation history cleared!")
                    continue
                
                if user_input.lower() == "memory":
                    if not self.conversation_history:
                        print("📝 No conversation history yet.")
                        self.speak("No conversation history yet.")
                    else:
                        print(f"📝 Conversation History ({len(self.conversation_history)//2} exchanges):")
                        for i, msg in enumerate(self.conversation_history):
                            role = "You" if msg["role"] == "user" else "Assistant"
                            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                            print(f"   {role}: {content}")
                        self.speak(f"I have {len(self.conversation_history)//2} conversation exchanges in memory.")
                    continue
                
                print("🤔 Thinking...")
                response = self.process_query(user_input)
                print(f"🤖 Assistant: {response}\n")
                
                # Speak the response
                self.speak(response)
                
                # Update conversation history (keep last 10 exchanges)
                self.conversation_history.append({"role": "user", "content": user_input})
                self.conversation_history.append({"role": "assistant", "content": response})
                
                # Keep only last 10 exchanges to avoid token limits
                if len(self.conversation_history) > 20:
                    self.conversation_history = self.conversation_history[-20:]
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                self.speak("Goodbye! Have a wonderful day!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("Please try again with a different query.\n")

# Example usage
if __name__ == "__main__":
    # Install required packages message
    print("📦 Required packages for voice features:")
    print("   pip install SpeechRecognition pyttsx3 pyaudio")
    print("   Note: pyaudio might need system dependencies on Linux/Mac")
    print("   Alternative: pip install speechrecognition pyttsx3 pipwin && pipwin install pyaudio")
    print()
    
    try:
        assistant = VoiceEnabledAIAssistant()
        assistant.chat()
    except ImportError as e:
        print(f"❌ Missing voice dependencies: {e}")
        print("Please install: pip install SpeechRecognition pyttsx3 pyaudio")
        print("Running in text-only mode...")
        
        # Fallback to original assistant
        import sys
        from original_assistant import SimpleAIAssistant  # You'd need to save original as separate file
        SimpleAIAssistant().chat()
