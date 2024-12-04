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
from firebase_functions import https_fn
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
        # Retrieve API key
        api_key = os.getenv('API_KEY')
        if not api_key:
            logger.error("API key is missing.")
            return jsonify({"error": "API key is required"}), 400

        # Retrieve root filepath from request data
        data = request.json
        root_filepath = data.get('rootPath')
        if not root_filepath:
            logger.error("Root filepath is missing from request data.")
            return jsonify({"error": "Root filepath is required"}), 400

        # Define paths
        haul_filepath = f"{root_filepath}/haul.json"
        haul_conditions_path = f"{root_filepath}/haulConditions.json"

        # Ensure haul.json exists
        bucket = storage.bucket()
        haul_blob = ensure_file_exists(bucket, haul_filepath, default_content={"ingredients": []})

        # Load ingredients from haul.json
        try:
            ingredients_data = json.loads(haul_blob.download_as_text())
            ingredients_list = ingredients_data.get('ingredients', [])
        except Exception as e:
            logger.error(f"Error reading haul.json: {e}")
            return jsonify({"error": f"Error reading haul.json: {e}"}), 500

        if not ingredients_list:
            logger.error("No ingredients found in haul.json.")
            return jsonify({"error": "No ingredients found in haul.json."}), 400

        # Ensure haulConditions.json exists
        haul_conditions_blob = ensure_file_exists(
            bucket,
            haul_conditions_path,
            default_content={"diets": [], "allergies": [], "spice_tolerance": None, "serving_size": 4, "prep_time": 20}
        )

        # Load conditions from haulConditions.json
        try:
            haul_conditions = json.loads(haul_conditions_blob.download_as_text())
            recipe_type = haul_conditions.get('diets')
            allergies = haul_conditions.get('allergies')
            serving_size = haul_conditions.get('serving_size')
            prep_time = haul_conditions.get('prep_time')
            spiciest_food = haul_conditions.get('spice_tolerance')
        except Exception as e:
            logger.error(f"Error reading haulConditions.json: {e}")
            return jsonify({"error": f"Error reading haulConditions.json: {e}"}), 500


        # Construct OpenAI prompt
        prompt_parts = [
            f"Generate 5 different meals I can make with this set of food: {', '.join(ingredients_list)}."
        ]
        if serving_size:
            prompt_parts.append(f"Each recipe should serve {serving_size} people.")
        if prep_time:
            prompt_parts.append(f"Each recipe should take no longer than {prep_time} minutes to prepare.")
        if recipe_type:
            prompt_parts.append(f"Each recipe should be suitable for a {', '.join(recipe_type)} diet.")
        if allergies:
            prompt_parts.append(f"Exclude any ingredients that may trigger these allergies: {', '.join(allergies)}.")
        if spiciest_food:
            prompt_parts.append(f"Each recipe should have a determined spice level (like mild, medium, or hot) based on the user's tolerance, which is set to: {spiciest_food}.")
        prompt_parts.append(f"Please format the recipes as a JSON array with each recipe containing 'meal', 'serving_size', 'type (diet)', 'allergies', 'prep_time', 'ingredients' with quantities, 'instructions' as an array, 'spice_level', and 'based_on'. The 'based_on' field should reflect the spiciest ingredient, which is: {spiciest_food}.")
        final_prompt = " ".join(prompt_parts)

        # OpenAI API call
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": final_prompt}],
            model="gpt-4o-mini"
        )

        # Parse OpenAI response
        generated_text = response.choices[0].message.content
        start_index = generated_text.find('[')
        end_index = generated_text.rfind(']')
        if start_index == -1 or end_index == -1:
            logger.error("Failed to find JSON in response.")
            return jsonify({"error": "Invalid response format from OpenAI."}), 500

        json_text = generated_text[start_index:end_index + 1]
        try:
            recipes = json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            return jsonify({"error": f"Invalid JSON format in OpenAI response: {e}"}), 500

        # Save recipes to recipes.json
        recipes_filepath = f"{root_filepath}/recipes.json"
        recipes_blob = bucket.blob(recipes_filepath)
        recipes_blob.upload_from_string(json.dumps(recipes, indent=2), content_type='application/json')

        logger.info("Recipes generated and saved successfully.")
        return jsonify({"recipes": recipes}), 200

    except Exception as e:
        logger.error(f"Failed to generate recipes: {e}")
        return jsonify({"error": f"Failed to generate recipes: {e}"}), 500



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

def ensure_file_exists(bucket, filepath, default_content=None):
    """
    Ensures that a file exists in Firebase Storage. If it doesn't exist, creates it with the specified default content.
    """
    blob = bucket.blob(filepath)
    if not blob.exists():
        logger.info(f"{filepath} does not exist. Creating a new one.")
        # Create a default content if provided, else an empty object
        default_content = default_content or {}
        blob.upload_from_string(json.dumps(default_content, indent=2), content_type="application/json")
        logger.info(f"Created new file at {filepath} with default content.")
    else:
        logger.info(f"{filepath} already exists.")
    return blob

@app.route('/get-allergy-list', methods=['POST'])
def get_allergy_list():
    logger.info("Entering get_allergy_list function.")
    try:
        data = request.json
        haul_conditions_filepath = data.get('filePath')
        logger.info("File Path: %s", haul_conditions_filepath)

        bucket = storage.bucket()
        blob = ensure_file_exists(bucket, haul_conditions_filepath, default_content={"allergies": [], "diets": [], "spice_tolerance": None, "serving_size": [4], "prep_time": [20]})

        json_data = json.loads(blob.download_as_text())
        logger.info(f"Retrieved data: {json_data}")

        allergies = json_data.get('allergies', [])
        return jsonify({"allergies": allergies}), 200
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/save-allergy', methods=['POST'])
def save_allergy():
    logger.info("Entering save_allergy function.")
    try:
        data = request.json
        index = data['index']
        new_allergy = data['newAllergy']
        haul_conditions_filepath = data.get('filePath')

        bucket = storage.bucket()
        blob = ensure_file_exists(bucket, haul_conditions_filepath, default_content={"allergies": [], "diets": [], "spice_tolerance": None, "serving_size": [4], "prep_time": [20]})

        json_content = blob.download_as_text()
        logger.debug(f"Blob content before update: {json_content}")
        allergy_data = json.loads(json_content)

        # Edit the allergy at the specified index
        if 0 <= index < len(allergy_data['allergies']):
            allergy_data['allergies'][index] = new_allergy
        else:
            return jsonify({"error": "Index out of range"}), 400

        updated_json = json.dumps(allergy_data, indent=2)

        blob.upload_from_string(updated_json, content_type='application/json')
        logger.info("Allergy updated successfully.")
        return jsonify({"message": "Allergy updated successfully!"})
    except Exception as e:
        logger.error("An error occurred: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route('/delete-allergy', methods=['POST'])
def delete_allergy():
    logger.info("Entering delete_allergy function.")
    try:
        data = request.json
        index = data['index']
        haul_conditions_filepath = data.get('filePath')

        bucket = storage.bucket()
        blob = ensure_file_exists(bucket, haul_conditions_filepath, default_content={"allergies": [], "diets": [], "spice_tolerance": None, "serving_size": [4], "prep_time": [20]})

        json_content = blob.download_as_text()
        logger.debug(f"Blob content before deletion: {json_content}")
        allergy_data = json.loads(json_content)

        # Delete the allergy
        if 0 <= index < len(allergy_data['allergies']):
            allergy_data['allergies'].pop(index)
        else:
            return jsonify({"error": "Index out of range"}), 400

        updated_json = json.dumps(allergy_data, indent=2)
        blob.upload_from_string(updated_json, content_type='application/json')
        logger.info("Allergy deleted successfully.")
        return jsonify({"message": "Allergy deleted successfully!"})
    except Exception as e:
        logger.error("An error occurred: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route('/add-allergy', methods=['POST'])
def add_allergy():
    logger.info("Entering add_allergy function.")
    try:
        data = request.json
        new_allergy = data.get('new_allergy')
        haul_conditions_filepath = data.get('filePath')

        bucket = storage.bucket()
        blob = ensure_file_exists(bucket, haul_conditions_filepath, default_content={"allergies": [], "diets": [], "spice_tolerance": None, "serving_size": [4], "prep_time": [20]})

        json_content = blob.download_as_text()
        logger.debug(f"Blob content before addition: {json_content}")
        allergy_data = json.loads(json_content)

        # Check for duplicates
        if new_allergy in allergy_data['allergies']:
            return jsonify({"error": "Allergy already exists"}), 400

        # Add the new allergy
        allergy_data['allergies'].append(new_allergy)
        updated_json = json.dumps(allergy_data, indent=2)
        logger.debug(f"Blob content after update: {updated_json}")
        blob.upload_from_string(updated_json, content_type='application/json')
        logger.info("Allergy added successfully.")
        return jsonify({"message": "Allergy added successfully!"})
    except Exception as e:
        logger.error("An error occurred: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route('/clear-allergy-list', methods=['POST'])
def clear_allergy_list():
    logger.info("Entering clear_allergy_list function.")
    try:
        data = request.json
        haul_conditions_filepath = data.get('filePath')

        bucket = storage.bucket()
        blob = ensure_file_exists(bucket, haul_conditions_filepath, default_content={"allergies": [], "diets": [], "spice_tolerance": None, "serving_size": [4], "prep_time": [20]})

        json_content = blob.download_as_text()
        conditions_data = json.loads(json_content)
        logger.debug(f"Blob content before clearing allergies: {json_content}")

        # Clear only the allergies field while keeping the other fields unchanged
        conditions_data["allergies"] = []

        # Upload the updated data back to Storage
        updated_json = json.dumps(conditions_data, indent=2)
        blob.upload_from_string(updated_json, content_type="application/json")
        logger.info("All allergies cleared successfully.")
        return jsonify({"message": "All allergies cleared successfully!"})
    except Exception as e:
        logger.error("An error occurred: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route('/get-diet-list', methods=['POST'])
def get_diet_list():
    logger.info("Entering get_diet_list function.")
    try:
        data = request.json
        haul_conditions_filepath = data.get('filePath')
        logger.info("File Path: %s", haul_conditions_filepath)

        bucket = storage.bucket()
        blob = ensure_file_exists(bucket, haul_conditions_filepath, default_content={"allergies": [], "diets": [], "spice_tolerance": None, "serving_size": [4], "prep_time": [20]})

        json_data = json.loads(blob.download_as_text())
        logger.info(f"Retrieved data: {json_data}")

        diets = json_data.get('diets', [])
        return jsonify({"diets": diets}), 200
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/save-diet', methods=['POST'])
def save_diet():
    logger.info("Entering save_diet function.")
    try:
        data = request.json
        index = data['index']
        new_diet = data['newDiet']
        haul_conditions_filepath = data.get('filePath')

        bucket = storage.bucket()
        blob = ensure_file_exists(bucket, haul_conditions_filepath, default_content={"allergies": [], "diets": [], "spice_tolerance": None, "serving_size": [4], "prep_time": [20]})

        json_content = blob.download_as_text()
        logger.debug(f"Blob content before update: {json_content}")
        diet_data = json.loads(json_content)

        # Edit the diet at the specified index
        if 0 <= index < len(diet_data['diets']):
            diet_data['diets'][index] = new_diet
        else:
            return jsonify({"error": "Index out of range"}), 400

        updated_json = json.dumps(diet_data, indent=2)

        blob.upload_from_string(updated_json, content_type='application/json')
        logger.info("Diet updated successfully.")

        return jsonify({"message": "Diet updated successfully!"})
    except Exception as e:
        logger.error("An error occurred: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route('/delete-diet', methods=['POST'])
def delete_diet():
    logger.info("Entering delete_diet function.")
    try:
        data = request.json
        index = data['index']
        haul_conditions_filepath = data.get('filePath')

        bucket = storage.bucket()
        blob = ensure_file_exists(bucket, haul_conditions_filepath, default_content={"allergies": [], "diets": [], "spice_tolerance": None, "serving_size": [4], "prep_time": [20]})

        json_content = blob.download_as_text()
        logger.debug(f"Blob content before deletion: {json_content}")
        diet_data = json.loads(json_content)

        # Delete the diet
        if 0 <= index < len(diet_data['allergies']):
            diet_data['diets'].pop(index)
        else:
            return jsonify({"error": "Index out of range"}), 400

        updated_json = json.dumps(diet_data, indent=2)
        blob.upload_from_string(updated_json, content_type='application/json')
        logger.info("Diet deleted successfully.")

        return jsonify({"message": "Diet deleted successfully!"})
    except Exception as e:
        logger.error("An error occurred: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route('/add-diet', methods=['POST'])
def add_diet():
    logger.info("Entering add_diet function.")
    try:
        data = request.json
        new_diet = data.get('new_diet')
        haul_conditions_filepath = data.get('filePath')

        bucket = storage.bucket()
        blob = ensure_file_exists(bucket, haul_conditions_filepath, default_content={"allergies": [], "diets": [], "spice_tolerance": None, "serving_size": [4], "prep_time": [20]})

        json_content = blob.download_as_text()
        logger.debug(f"Blob content before addition: {json_content}")
        diet_data = json.loads(json_content)

        # Check for duplicates
        if new_diet in diet_data['allergies']:
            return jsonify({"error": "Diet already exists"}), 400

        # Add the new diet
        diet_data['diets'].append(new_diet)
        updated_json = json.dumps(diet_data, indent=2)
        logger.debug(f"Blob content after update: {updated_json}")
        blob.upload_from_string(updated_json, content_type='application/json')
        logger.info("Diet added successfully.")

        return jsonify({"message": "Diet added successfully!"})
    except Exception as e:
        logger.error("An error occurred: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route('/clear-diet-list', methods=['POST'])
def clear_diet_list():
    logger.info("Entering clear_diets_list function.")
    try:
        data = request.json
        haul_conditions_filepath = data.get('filePath')

        bucket = storage.bucket()
        blob = ensure_file_exists(bucket, haul_conditions_filepath, default_content={"allergies": [], "diets": [], "spice_tolerance": None, "serving_size": [4], "prep_time": [20]})

        json_content = blob.download_as_text()
        conditions_data = json.loads(json_content)
        logger.debug(f"Blob content before clearing diets: {json_content}")

        # Clear only the diets field while keeping the other fields unchanged
        conditions_data["diets"] = []

        # Upload the updated data back to Storage
        updated_json = json.dumps(conditions_data, indent=2)
        blob.upload_from_string(updated_json, content_type="application/json")
        logger.info("All diets cleared successfully.")

        return jsonify({"message": "All diets cleared successfully!"})
    except Exception as e:
        logger.error("An error occurred: %s", e)
        return jsonify({"error": str(e)}), 500
     
@app.route('/save-spice-tolerance', methods=['POST'])
def save_spice_tolerance():
    logger.info("Entering save_spice_tolerance function.")

    try:
        data = request.json
        spice_tolerance = data.get('spice_tolerance')

        haul_conditions_filepath = data.get('filePath')

        bucket = storage.bucket()
        blob = ensure_file_exists(bucket, haul_conditions_filepath, default_content={"allergies": [], "diets": [], "spice_tolerance": None, "serving_size": [4], "prep_time": [20]})

        json_content = blob.download_as_text()
        logger.debug(f"Blob content before addition: {json_content}")
        spice_data = json.loads(json_content)

        # Convert to a list if None
        if spice_data['spice_tolerance'] is None:
            spice_data['spice_tolerance'] = []

        # Check for duplicates
        elif spice_tolerance in spice_data['spice_tolerance']:
            return jsonify({"error": "Spice tolerance already exists"}), 400

        # Add the new spice tolerance
        spice_data['spice_tolerance'] = [spice_tolerance]
        updated_json = json.dumps(spice_data, indent=2)
        logger.debug(f"Blob content after update: {updated_json}")
        blob.upload_from_string(updated_json, content_type='application/json')

        logger.info("Spice tolerance edited successfully.")
        return jsonify({"message": "Spice tolerance edited successfully!"})
    
    except Exception as e:
        logger.error("An error occurred: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route('/save-serving-size', methods=['POST'])
def save_serving_size():
    logger.info("Entering save_serving_size function.")

    try:
        data = request.json
        serving_size = data.get('serving_size')

        haul_conditions_filepath = data.get('filePath')

        bucket = storage.bucket()
        blob = ensure_file_exists(bucket, haul_conditions_filepath, default_content={"allergies": [], "diets": [], "spice_tolerance": None, "serving_size": [4], "prep_time": [20]})

        json_content = blob.download_as_text()
        logger.debug(f"Blob content before addition: {json_content}")
        size_data = json.loads(json_content)

        # Add serving size header if it doesn't exist
        if 'serving_size' not in size_data:
            size_data['serving_size'] = []

        # Check for duplicates
        elif serving_size in size_data['serving_size']:
            return jsonify({"error": "Serving size already exists"}), 400

        # Add the new spice tolerance
        size_data['serving_size'] = [int(serving_size)]
        updated_json = json.dumps(size_data, indent=2)
        logger.debug(f"Blob content after update: {updated_json}")
        blob.upload_from_string(updated_json, content_type='application/json')

        logger.info("Serving size edited successfully.")
        return jsonify({"message": "Serving size edited successfully!"})
    
    except Exception as e:
        logger.error("An error occurred: %s", e)
        return jsonify({"error": str(e)}), 500
    
@app.route('/save-prep-time', methods=['POST'])
def save_prep_time():
    logger.info("Entering save_prep_time function.")

    try:
        data = request.json
        prep_time = data.get('prep_time')

        haul_conditions_filepath = data.get('filePath')

        bucket = storage.bucket()
        blob = ensure_file_exists(bucket, haul_conditions_filepath, default_content={"allergies": [], "diets": [], "spice_tolerance": None, "serving_size": [4], "prep_time": [20]})

        json_content = blob.download_as_text()
        logger.debug(f"Blob content before addition: {json_content}")
        time_data = json.loads(json_content)

        # Add prep time header if it doesn't exist
        if 'prep_time' not in time_data:
            time_data['prep_time'] = []

        # Check for duplicates
        elif prep_time in time_data['prep_time']:
            return jsonify({"error": "Prep time already exists"}), 400

        # Add the new spice tolerance
        time_data['prep_time'] = [int(prep_time)]
        updated_json = json.dumps(time_data, indent=2)
        logger.debug(f"Blob content after update: {updated_json}")
        blob.upload_from_string(updated_json, content_type='application/json')

        logger.info("Prep time edited successfully.")
        return jsonify({"message": "Prep time edited successfully!"})
    
    except Exception as e:
        logger.error("An error occurred: %s", e)
        return jsonify({"error": str(e)}), 500
 
@app.route('/modify-ingredients', methods=['POST'])
def modify_ingredients():
    logger.info("Entering modify_ingredients function.")
    try:
        # Parse the incoming request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request data is required"}), 400

        # Extract the root path and API key from the request data
        root_filepath = data.get('rootPath')
        api_key = os.getenv('API_KEY')  # API key is now managed through environment variables
        if not api_key:
            logger.error("API key is missing.")
            return jsonify({"error": "API key is required"}), 400

        # Define file paths
        haul_filepath = f"{root_filepath}/haul.json"
        haul_conditions_path = f"{root_filepath}/haulConditions.json"

        bucket = storage.bucket()
        haul_blob = bucket.blob(haul_filepath)
        haul_conditions_blob = bucket.blob(haul_conditions_path)

        # Ensure the haul.json file exists
        if not haul_blob.exists():
            logger.error("haul.json file does not exist.")
            return jsonify({"error": "No ingredients found. Please add items to your fridge first."}), 400

        # Load conditions from haulConditions.json
        if not haul_conditions_blob.exists():
            logger.error("haulConditions.json file does not exist.")
            return jsonify({"error": "Conditions file not found. Please set up dietary preferences and allergies first."}), 400

        try:
            haul_conditions_data = json.loads(haul_conditions_blob.download_as_text())
            dietary_preferences = haul_conditions_data.get('diets', [])
            allergies = haul_conditions_data.get('allergies', [])
            spiciest_food = haul_conditions_data.get('spice_tolerance', '')
        except Exception as e:
            logger.error(f"Error reading haulConditions.json: {e}")
            return jsonify({"error": f"Error reading conditions file: {e}"}), 500

        # Load the current ingredient list from haul.json
        try:
            ingredients_data = json.loads(haul_blob.download_as_text())
            ingredients_list = ingredients_data.get('ingredients', [])
        except Exception as e:
            logger.error(f"Error reading haul.json: {e}")
            return jsonify({"error": f"Error reading ingredients file: {e}"}), 500

        if not ingredients_list:
            return jsonify({"error": "No ingredients found in haul.json."}), 400

        # Construct the OpenAI prompt
        prompt = (
            f"Edit the ingredient list to fit the following criteria: {', '.join(dietary_preferences)} "
            f"and free of: {', '.join(allergies)}. "
            f"Here is the current ingredient list: {', '.join(ingredients_list)}. "
            f"Additionally, please identify and remove any ingredients spicier than {spiciest_food}. "
            f"Return the output in a valid JSON format, and only include the updated list of ingredients under an 'ingredients' key."
        )

        # Call OpenAI API
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o-mini"
        )

        # Handle OpenAI response
        if not response or not response.choices or not response.choices[0].message.content:
            logger.error("OpenAI API returned an invalid response.")
            return jsonify({"error": "Failed to get a valid response from OpenAI."}), 500

        generated_text = response.choices[0].message.content.strip("```json").strip("```").strip()
        logger.debug(f"Generated text from OpenAI: {generated_text}")

        # Parse the generated JSON
        try:
            new_ingredients = json.loads(generated_text).get('ingredients', [])
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OpenAI response as JSON: {e}")
            return jsonify({"error": "OpenAI response is not valid JSON."}), 500

        # Save the updated ingredient list back to haul.json
        updated_ingredients = {"ingredients": new_ingredients}
        haul_blob.upload_from_string(json.dumps(updated_ingredients, indent=2), content_type='application/json')
        logger.info("Updated ingredients saved successfully.")

        return jsonify({"message": "Ingredient list modified successfully!", "ingredients": new_ingredients}), 200

    except Exception as e:
        logger.exception("An error occurred during ingredient modification.")
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