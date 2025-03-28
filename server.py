from flask import Flask, request, jsonify, send_from_directory
import logging
import sys
import os
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# Import functions from app.py
try:
    from app import get_hadith, get_quran, get_response, chain
    logger.info("Successfully imported functions from app.py")
except ImportError as e:
    logger.error(f"Error importing from app.py: {e}")
    logger.error(traceback.format_exc())

app = Flask(__name__, static_url_path='')

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'index.html')

@app.route('/api/query', methods=['POST'])
def handle_query():
    """Handle POST requests from the frontend and process using app.py functions"""
    try:
        # Get form data
        input_text = request.form.get('input_text')
        deep_search = request.form.get('deep_search', 'false').lower() == 'true'
        source = 'both'  # Always use both sources
        
        logger.info(f"Received query: {input_text}, Deep Search: {deep_search}")
        
        # Special case for creator question
        if "created" in input_text.lower() and "sirat" in input_text.lower():
            response = "SiratGPT was created by Zaid, a visionary AI engineer and entrepreneur who is passionate about fusing technology with knowledge. As the Founder of HatchUp.ai, Zaid built SiratGPT to bring deep Islamic insights to the digital world, combining modern AI techniques with timeless wisdom. His expertise in AI, app development, and automation drives this project, making SiratGPT a unique and intelligent guide for seekers of knowledge."
            return jsonify({"response": response})
        
        # Get data from both sources
        hadith = None
        quran = None
        
        logger.info("Starting to fetch data from sources...")
        
        try:
            logger.info("Fetching hadith data...")
            hadith = get_hadith(input_text)
            logger.info(f"Hadith data fetched: {hadith is not None}")
        except Exception as e:
            logger.error(f"Error fetching hadith data: {str(e)}")
            logger.error(traceback.format_exc())
            
        try:
            logger.info("Fetching quran data...")
            quran = get_quran(input_text)
            logger.info(f"Quran data fetched: {quran is not None}")
        except Exception as e:
            logger.error(f"Error fetching quran data: {str(e)}")
            logger.error(traceback.format_exc())
        
        combined_data = ""
        if hadith:
            combined_data += hadith + "\n"
        if quran:
            combined_data += quran + "\n"
        
        logger.info(f"Combined data length: {len(combined_data)}")
        
        # If we don't have any data, return a fallback response
        if not combined_data:
            logger.warning("No data found in sources, using fallback response")
            return jsonify({"response": f"I couldn't find specific information about '{input_text}' in my database. Please try asking about another Islamic topic like Ramadan, prayer times, Zakat, or Hajj."})
        
        # Process using LangChain
        logger.info("Running LLM chain...")
        try:
            output = chain.run(dataset=combined_data, input=input_text)
            logger.info("LLM chain completed")
        except Exception as e:
            logger.error(f"Error running LLM chain: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({"response": f"I encountered an issue processing your query. Here's what I found: {combined_data}"})
        
        # Clean the response if needed
        if "Response:" in output:
            response = output.split("Response:")[-1].strip()
        else:
            response = output
        
        # If deep search is enabled, add additional info
        if deep_search:
            try:
                logger.info("Performing deep search...")
                deep_search_result = get_response(input_text)
                response += "\n\n--- Deep Search Results ---\n" + deep_search_result
                logger.info("Deep search completed")
            except Exception as e:
                logger.error(f"Error in deep search: {str(e)}")
                logger.error(traceback.format_exc())
                response += "\n\n--- Deep Search Results ---\nUnable to perform deep search due to an error."
        
        logger.info("Sending response to client")
        return jsonify({"response": response})
    
    except Exception as e:
        logger.error(f"Unhandled error in handle_query: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"response": f"I apologize, but I encountered an error processing your request. Please try again with a different question. Error details: {str(e)}"})

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

@app.route('/test')
def test():
    """Test endpoint to check if server is running"""
    return jsonify({"status": "ok", "message": "Server is running correctly"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting server on port {port}")
    app.run(debug=True, host='0.0.0.0', port=port)