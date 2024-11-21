from flask import Flask, render_template, request, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import json
from openai import OpenAI
#from dotenv import load_dotenv
import os
import base64
import requests
import logging
from waitress import serve
from flask_cors import CORS
import io


# Firebase Imports
import firebase_admin
from firebase_admin import auth, credentials, storage, firestore, functions
import functions_framework
from firebase_functions import https_fn

app = Flask(__name__, template_folder='../public')

# Enable CORS for all routes
CORS(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# logger.info("Current working directory: %s", os.getcwd())

# Initalize Firebase Admin SDK
cred = credentials.Certificate('cook-ai-4fc88-fd2283c2e331.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'cook-ai-4fc88.appspot.com'
})

#load_dotenv()
api_key = os.getenv('API_KEY')

@app.route('/response/<filename>')
def response(filename):
    return send_from_directory('response', filename)

def validate_json(data):
    try:    
        if all(key in data for key in ["title", "serving_size", "ingredients", "instructions"]):
            return True
        return False
    except Exception as e:
        print(f"Validation error: {e}")
        return False

# Improved flatten_ingredients function with underscore handling
def flatten_ingredients(ingredients):
    if isinstance(ingredients, list):
        return [f"{item['ingredient'].replace('_', ' ')}: {item['measurement']}" for item in ingredients if 'ingredient' in item and 'measurement' in item]
    return ingredients

@app.route('/generate', methods=['POST'])
def generate():
    logger.info("Entering generate function.")

    data = request.get_json()
    logger.debug(f"Request JSON data: {data}")

    if not data:
        logger.error("No JSON data received")
        return jsonify({"error": "No JSON data received"}), 400

    #api_key = data.get('api_key')
    user_input = data.get('user_input')
    #api_key = os.getenv('API_KEY')

    #if not api_key:
    #    logger.error("API key is required")
    #    return jsonify({"error": "API key is required"}), 400
    if not user_input:
        logger.error("User input is required")
        return jsonify({"error": "User input is required"}), 400

    root_filepath = data.get('rootPath')
    haul_filepath = root_filepath + '/haul.json'

    bucket = storage.bucket()
    blob = bucket.blob(haul_filepath)

    if not blob.exists():
        logger.error("No ingredients found. Please add items to your fridge first.")
        return jsonify({"error": "No ingredients found. Please add items to your fridge first."}), 400

    # Download ingredients data from Firebase Storage
    try:
        ingredients_data = json.loads(blob.download_as_text())
        ingredients_list = ingredients_data.get('ingredients', [])
        if not ingredients_list:
            logger.error("No ingredients found in haul.json")
            return jsonify({"error": "No ingredients found in haul.json"}), 400
    except Exception as e:
        return jsonify(error=f"Error reading haul.json: {str(e)}"), 500

    # Construct the prompt for the OpenAI API
    # Construct the prompt for the OpenAI API
    prompt_parts = [
        "We are creating a recipe. The output should be in JSON format with specific sections with no additional comments.",
        "The JSON structure should include 'title', 'serving_size', 'ingredients', and 'instructions' in that order.",
        "Start with a 'title' section, followed by 'serving_size'. Then, 'serving_size' should be 4.",
        "Each ingredient must have a measurement indicating how much to use of that ingredient.",
        (
            "Finally, provide 'instructions' as an ordered list. You do not have to use all the ingredients in the following list. "
            "The 'instructions' section needs to include the minutes it will take to cook each ingredient in plain sentence format. "
            "Only use what you need for the recipe. If the following list does not include an ingredient, do NOT include it. "
            f"Here is the list of ingredients to choose from: {', '.join(ingredients_list)}"
        ),
        "Please format the ingredients as a JSON object.",
        f"The food I want to make is: {user_input}"
    ]
    final_prompt = " ".join(prompt_parts)

    max_attempts = 15
    attempt_counter = 0
    recipe_filepath = f"{root_filepath}/recipes.json"

    api_key = os.getenv('API_KEY')

    while attempt_counter < max_attempts:
        attempt_counter += 1
        try:
            # Create an OpenAI client instance
            client = OpenAI(api_key=api_key)

            # Generate a chat completion
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": final_prompt}],
                #model="gpt-3.5-turbo"
                model="gpt-4o-mini"
            )

            # Extract the generated text
            generated_text = response.choices[0].message.content

            # Save the generated text to a JSON file in the user's Firebase Storage folder
            try:
                recipe_data = json.loads(generated_text)
                
                if validate_json(recipe_data):
                         
                    recipe_data['ingredients'] = flatten_ingredients(recipe_data['ingredients'])

                    # Save JSON to Firebase Storage
                    recipe_blob = bucket.blob(recipe_filepath)
                    recipe_blob.upload_from_string(json.dumps(recipe_data, indent=2), content_type='application/json')
                    logger.info("Successfully saved recipe data to Firebase Storage")

                    logger.info("GENERATED: %s", recipe_data)

                    break

                else:
                    print(f"Attempt {attempt_counter} failed. Invalid JSON format. Retrying...")

            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error on attempt {attempt_counter}: {e}")
                continue
            except ValueError as ve:
                logger.error(f"Invalid JSON structure on attempt {attempt_counter}: {ve}")
                continue

        except Exception as e:
            logger.error(f"Attempt {attempt_counter} failed with error: {str(e)}")

    if attempt_counter == max_attempts:
        return "Failed to generate a valid recipe after multiple attempts.", 500

    # Retrieve the generated recipe data from Firebase Storage
    recipe_data = json.loads(recipe_blob.download_as_text())

    # Render the template with the recipe data
    #return render_template('generate.html', recipe=recipe_data)
    return jsonify(recipe=recipe_data)

@app.route('/upload-haul', methods=['POST'])
def upload_haul():
    logger.info("Entering upload_haul function.")

    data = request.get_json()

    file_data = data.get('fileData')  # The base64 file data
    file_name = data.get('fileName')  # The original file name
    file_type = data.get('fileType')  # The original file type (e.g., image/jpeg)

    if not file_data:
        return "No file part", 400

    # Decode the base64 string
    #file_bytes = base64.b64decode(file_data)
    try:
        file_bytes = base64.b64decode(file_data)
        logger.info(f"Successfully decoded base64 data for {file_name}")
    except Exception as e:
        logger.error(f"Failed to decode base64 data: {e}")
        return "Invalid base64 data", 400

    #api_key = data.get('api_key')

    if file_name == '':
        return "No selected file", 400
    
    # Create a reference to the user's haul.jpeg file (where uploaded image will go in Firebase Storage)
    root_filepath = data.get('rootPath')
    logger.info("Root File Path: %s", root_filepath)
    haul_image_filepath = root_filepath + '/haul.jpeg'

    bucket = storage.bucket()
    image_blob = bucket.blob(haul_image_filepath)
    
    if file_bytes: #and api_key:
        # Convert the file bytes into BytesIO filelike object (to use with Firebase functions)
        file_stream = io.BytesIO(file_bytes)

        # Upload image to Firebase
        image_blob.upload_from_file(file_stream, content_type=file_type)

        # Get the public URL for the uploaded image
        image_url = image_blob.public_url

        # Download the image from Firebase and convert to base64
        image_content = image_blob.download_as_bytes()
        base64_image = base64.b64encode(image_content).decode('utf-8')

        api_key = os.getenv('API_KEY')
        if not api_key:
            logger.error("API Key not found in environment")
            return "API key not configured", 500

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Put a name to all of these ingredients in a JSON formatted list with no extra jargon. Include the label 'ingredients'"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                    # "content": f"Put a name to all of these ingredients in a JSON formatted list with no extra jargon. Include the label 'ingredients'. Here is the image in base64: data:image/jpeg;base64,{base64_image}"

                }
            ],
            "max_tokens": 300
        } 

        response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 200:
            json_response = response.json()
            content = json_response['choices'][0]['message']['content']
            
            # Extract the JSON part from the content
            start = content.find('{')
            end = content.rfind('}') + 1
            json_data = content[start:end]
            
            # Debug prints to check the extracted JSON data
            print(f"Full content:\n{content}\n")
            print(f"Extracted JSON data:\n{json_data}\n")
            
            if json_data.strip():  # Check if json_data is not empty
                # json_filepath = os.path.join(directory, 'haul.json')

                new_ingredients = json.loads(json_data).get('ingredients', [])

                haul_filepath = root_filepath + '/haul.json'

                # Create a reference to the user's haul.json file
                bucket = storage.bucket()
                blob = bucket.blob(haul_filepath)

                # Get the existing ingredients from the user's haul.json file
                if blob.exists():
                    existing_data = json.loads(blob.download_as_text())
                    existing_ingredients = existing_data.get('ingredients', [])
                else:
                    existing_ingredients = []

                # Append new ingredients to existing ones
                combined_ingredients = existing_ingredients + new_ingredients

                # Remove duplicates while preserving order
                seen = set()
                combined_ingredients = [item for item in combined_ingredients if not (item in seen or seen.add(item))]

                # Upload the updated fridge back to Storage
                updated_json = json.dumps({"ingredients": combined_ingredients}, indent=2)
                blob.upload_from_string(updated_json, content_type='application/json')

                return jsonify({'ingredients': combined_ingredients})
            else:
                return "Extracted JSON data is empty", 500
        else:
            return f"OpenAI API request failed: {response.text}", 500

    return "File upload failed", 500


@app.route('/generate-recipes', methods=['POST'])
def generate_recipes():
    logger.info("Entering generate_recipes function.")

    try:
        #api_key = request.json.get('api_key_recipes')
        api_key = os.getenv('API_KEY')
        if not api_key:
            return jsonify(error="API key is required"), 400 

        # Create a reference to the user's haul.json file
        data = request.json
        root_filepath = data.get('rootPath')
        haul_filepath = root_filepath + '/haul.json'

        bucket = storage.bucket()
        blob = bucket.blob(haul_filepath)

        if not blob.exists():
            return jsonify(error="No ingredients found. Please add items to your fridge first."), 400

        try:
            # Download the JSON data from the blob
            ingredients_data = json.loads(blob.download_as_text())
            ingredients_list = ingredients_data.get('ingredients', [])
            if not ingredients_list:
                return jsonify(error="No ingredients found in haul.json"), 400
        except Exception as e:
            return jsonify(error=f"Error reading haul.json: {str(e)}"), 500

        # Ensure ingredients_list is a list of strings
        if isinstance(ingredients_list, list):
            ingredients_list = [ingredient if isinstance(ingredient, str) else str(ingredient) for ingredient in ingredients_list]

        # Construct the prompt for the OpenAI API
        prompt_parts = [
            f"Generate 5 different meals I can make with this set of food: {', '.join(ingredients_list)}.",
            "Please format the recipes as a JSON array with each recipe containing 'meal', 'serving_size', 'ingredients', and 'instructions'."
        ]
        final_prompt = " ".join(prompt_parts)

        api_key = os.getenv('API_KEY')
        # Create an OpenAI client instance
        client = OpenAI(api_key=api_key)

        # Generate a chat completion
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": final_prompt}],
            model="gpt-4o-mini"
        )

        # Extract the generated text
        generated_text = response.choices[0].message.content

        # Debug print to check the raw API response
        print(f"Raw API response: {generated_text}")

        # Find the JSON object within the text
        start_index = generated_text.find('[')
        end_index = generated_text.rfind(']')
        if start_index == -1 or end_index == -1:
            return jsonify(error="Failed to find JSON in response"), 500

        json_text = generated_text[start_index:end_index + 1]

        # Convert the extracted JSON text to a Python object
        recipes = json.loads(json_text)

        # Save the extracted JSON to recipes.json

        # json_filepath = os.path.join(app.root_path, 'haul', 'recipes.json')
        # with open(json_filepath, 'w') as json_file:
        #     json.dump(recipes, json_file, indent=2)

        recipe_filepath = root_filepath + '/recipes.json'
        recipe_blob = bucket.blob(recipe_filepath)
        recipe_blob.upload_from_string(json.dumps(recipes, indent=2), content_type='application/json')

        return jsonify(recipes=recipes)

    except Exception as e:
        return jsonify(error=f"Failed to generate recipes: {str(e)}"), 500

# Route to get the current food list
@app.route('/get-food-list', methods=['POST'])
def get_food_list():
    logger.info("Entering get_food_list function.")

    try:
        # Create a reference to the user's haul.json file
        data = request.json
        haul_filepath = data.get('filePath')

        bucket = storage.bucket()
        blob = bucket.blob(haul_filepath)

        if not blob.exists():
            return jsonify({"ingredients": []}), 200

        # Download the JSON data from the blob
        json_data = json.loads(blob.download_as_text())

        logger.info(f"File contents: {json_data}")

        # Check if haul.json is empty
        if json_data == {}:
            return jsonify({"ingredients": []}), 200
        
        return jsonify(json_data), 200

    except Exception as e:
        logger.error("An error occurred: %s", e)
        return jsonify({"error": str(e)}), 500

# Route to save an edited food item
@app.route('/save-food', methods=['POST'])
def save_food():
    logger.info("Entering save_food (edit) function.")

    try:
        data = request.json
        index = data['index']
        new_name = data['new_name']

        # Create a reference to the user's haul.json file
        haul_filepath = data.get('filePath')
        bucket = storage.bucket()
        blob = bucket.blob(haul_filepath)

        # Download the haul.json fridge contents
        json_content = blob.download_as_text()
        food_data = json.loads(json_content)

        # Edit the name of the food at index
        if 0 <= index < len(food_data['ingredients']):
            food_data['ingredients'][index] = new_name
        else:
            return jsonify({"error": "Index out of range"}), 400
        
        # Upload the updated fridge back to Storage
        updated_json = json.dumps(food_data)
        blob.upload_from_string(updated_json, content_type='application/json')

        return jsonify({"message": "Food name updated successfully!"})
            
    except Exception as e:
        logger.error("An error occurred: %s", e)
        return jsonify({"error": str(e)}), 500

# Route to delete a food item
@app.route('/delete-food', methods=['POST'])
def delete_food():
    logger.info("Entering delete_food function.")

    try:
        data = request.json
        index = data['index']

        # Create a reference to the user's haul.json file
        haul_filepath = data.get('filePath')
        bucket = storage.bucket()
        blob = bucket.blob(haul_filepath)

        # Download the haul.json fridge contents
        json_content = blob.download_as_text()
        food_data = json.loads(json_content)  

        # Delete the food at index
        if 0 <= index < len(food_data['ingredients']):
            food_data['ingredients'].pop(index)
        else:
            return jsonify({"error": "Index out of range"}), 400
        
        # Upload the updated fridge back to Storage
        updated_json = json.dumps(food_data, indent=2)
        blob.upload_from_string(updated_json, content_type='application/json')

        return jsonify({"message": "Food deleted successfully!"})
    
    except Exception as e:
        logger.error("An error occurred: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route('/add-food', methods=['POST'])
def add_food():
    logger.info("Entering add_food function.")

    try:
        data = request.json
        new_food = data['newFood']

        # Create a reference to the user's haul.json file
        haul_filepath = data.get('filePath')

        bucket = storage.bucket()
        blob = bucket.blob(haul_filepath)

        # Check the file for JSON data and create it if not
        if not blob.exists():
            food_data = {'ingredients': []}
        else:
            json_content = blob.download_as_text()
            food_data = json.loads(json_content)

        # If the JSON data exists but is empty, start the ingredients list
        if food_data == {}:
            food_data = {'ingredients': []}            

        # Check for duplicates before adding
        if new_food in food_data['ingredients']:
            return jsonify({"error": "Food item already exists"}), 400

        # Add the new food to the fridge
        food_data['ingredients'].append(new_food)

        # Upload the updated fridge back to Storage
        updated_json = json.dumps(food_data, indent=2)
        blob.upload_from_string(updated_json, content_type='application/json')

        return jsonify({"message": "Food added successfully!"})
    
    except Exception as e:
        logger.error("An error occurred: %s", e)
        return jsonify({"error": str(e)}), 500

# Route to clear the food list
@app.route('/clear-food-list', methods=['POST'])
def clear_food_list():
    logger.info("Entering clear_food_list function.")

    try:
        # Create a reference to the user's haul.json file
        data = request.json
        haul_filepath = data.get('filePath')

        bucket = storage.bucket()
        blob = bucket.blob(haul_filepath)

        # Download the haul.json fridge contents
        json_content = blob.download_as_text()
        food_data = json.loads(json_content)

        # Clear the food list by overwriting with an empty list
        food_data = {'ingredients': []}

        # Upload the cleared fridge back to Storage
        updated_json = json.dumps(food_data, indent=2)
        blob.upload_from_string(updated_json, content_type='application/json')

        return jsonify({"message": "All food items cleared successfully!"})
    
    except Exception as e:
        logger.error("An error occurred: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route('/check-haul', methods=['GET'])
def check_haul():
    logger.debug("Received request to /check-haul")
    try:
        haul_filepath = request.args.get('path')

        bucket = storage.bucket()
        blob = bucket.blob(haul_filepath)

        if blob.exists():
            logger.debug("Haul file exists")
            return jsonify({"exists": True})
        else:
            logger.debug("Haul file does not exist")
            return jsonify({"exists": False})
    except Exception as e:
        logger.exception("Error during haul check")
        return jsonify({"error": str(e)}), 500
    
@app.route('/get-recipe', methods=['POST'])
def get_recipe():
    try:
        # Create a reference to the user's haul.json file
        data = request.json
        recipe_filepath = data.get('recipePath')

        bucket = storage.bucket()
        blob = bucket.blob(recipe_filepath)

        # Download the recipes.json contents
        recipe_data = blob.download_as_text()

        return jsonify({"recipe": recipe_data})

    except Exception as e:
        logger.error("An error occurred: %s", e)
        return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
    # app.run(debug=True))
    # serve(app, host='0.0.0.0', port=8080)
    # print("Starting Flask app...")
    # serve(app, host="127.0.0.1", port=5000)

@https_fn.on_request(max_instances=1)
def flask_app(req: https_fn.Request) -> https_fn.Response:
    with app.request_context(req.environ):
        return app.full_dispatch_request()