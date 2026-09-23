from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    jsonify
)
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_cors import CORS
import pandas as pd
import re
import os
import shutil
from datetime import datetime, timedelta
import secrets
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from ultralytics import YOLO
from werkzeug.utils import secure_filename
# ==========================================
# SAFE CSV READER - AUTO ENCODING
# ==========================================
def read_csv_safe(file_path, **kwargs):
    """
    Safely read CSV files with automatic encoding fallback.

    The caller may pass encoding=... .
    It is removed from kwargs before pandas.read_csv() is called,
    so the encoding argument can never be passed twice.
    """
    requested_encoding = kwargs.pop("encoding", None)

    fallback_encodings = [
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin1"
    ]

    if requested_encoding:
        encodings = [requested_encoding]

        for enc in fallback_encodings:
            if enc.lower() != str(requested_encoding).lower():
                encodings.append(enc)
    else:
        encodings = fallback_encodings

    last_error = None

    for encoding in encodings:
        try:
            df = pd.read_csv(
                file_path,
                encoding=encoding,
                **kwargs
            )

            print(
                f"CSV loaded successfully: {file_path} "
                f"| encoding={encoding}"
            )

            return df

        except UnicodeDecodeError as e:
            last_error = e

            print(
                f"CSV encoding failed: {encoding} "
                f"| {file_path}"
            )

            continue

    if last_error is not None:
        raise last_error

    raise ValueError(
        f"Unable to read CSV file: {file_path}"
    )



# ==========================================
# Load Dataset
# ==========================================


from ultralytics import YOLO

from werkzeug.utils import secure_filename
app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "http://127.0.0.1:5173"], supports_credentials=True)

# ==========================================
# APP CONFIGURATION
# ==========================================

app.config["SECRET_KEY"] = "smart_cooking_assistant_secret"

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# ==========================================
# GMAIL SMTP CONFIGURATION
# ==========================================

import os

GMAIL_ADDRESS = os.environ.get("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD")

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
UPLOAD_FOLDER = "uploads"

PREDICTION_FOLDER = "predictions"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.config["PREDICTION_FOLDER"] = PREDICTION_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

os.makedirs(PREDICTION_FOLDER, exist_ok=True)
# Single ingredient model
single_model = YOLO("models/best_single.pt")

# Multiple ingredient model
multiple_model = YOLO("models/best_multiple.pt")
def automatic_detection(image_path):

    # First check with multiple model
    multiple_results = multiple_model(
        image_path,
        conf=0.10
    )

    # Find different detected classes
    detected_classes = set()

    for result in multiple_results:

        if result.boxes is None:
            continue

        for cls in result.boxes.cls:
            detected_classes.add(int(cls))

    # 2 or more different ingredients
    if len(detected_classes) >= 2:

        print("🍲 Multiple ingredients detected")
        print("Using MULTIPLE model")

        return multiple_results, "multiple"

    # 0 or 1 ingredient
    print("🥕 Single ingredient detected")
    print("Using SINGLE model")

    single_results = single_model(
        image_path,
        conf=0.10
    )

    return single_results, "single"

def load_dataset(language):

    if language == "urdu":

        return {

            # Files
            "ingredients":
            "data/Pakistani_Food_urdu/pakistani_ingredients_urdu.csv",

            "steps":
            "data/Pakistani_Food_urdu/pakistani_steps_urdu.csv",

            "nutrition":
            "data/Pakistani_Food_urdu/pakistani_nutrition_urdu.csv",

            "recipes":
            "data/Pakistani_Food_urdu/pakistani_recipes_urdu.csv",

            # Urdu Column Names
            "recipe_col":
            "رسیپی_آئی_ڈی",

            "ingredient_col":
            "جزو_کا_نام",

            "quantity_col":
            "مقدار",

            "step_no":
            "قدم_نمبر",

            "step_col":
            "ہدایت"

        }

    else:

        return {

            # Files
            "ingredients":
            "data/Pakistani_Food/pakistani_ingredients.csv",

            "steps":
            "data/Pakistani_Food/pakistani_steps.csv",

            "nutrition":
            "data/Pakistani_Food/pakistani_nutrition.csv",

            "recipes":
            "data/Pakistani_Food/pakistani_recipes.csv",

            # English Column Names
            "recipe_col":
            "recipe_id",

            "ingredient_col":
            "ingredient_name",

            "quantity_col":
            "quantity",

            "step_no":
            "step_number",

            "step_col":
            "instruction"

        }
# ==========================================
# Scale Quantity
# ==========================================

def scale_quantity(text, persons, base=2):

    if pd.isna(text):
        return ""

    text = str(text).strip()

    scale = persons / base

    # ==========================
    # Fraction Support
    # ==========================

    fractions = {

        "1/4": 0.25,
        "1/2": 0.50,
        "3/4": 0.75

    }

    for f, v in fractions.items():

        if f in text:

            text = text.replace(
                f,
                str(round(v * scale, 2))
            )

    # ==========================
    # Scale Numbers
    # ==========================

    pattern = r'(?<![\d.])\d+(\.\d+)?'

    def replace(match):

        value = float(match.group())

        value = round(value * scale, 2)

        # Integer
        if value.is_integer():
            value = int(value)

        return str(value)

    text = re.sub(pattern, replace, text)

    # ==========================
    # Gram → Kilogram
    # ==========================

    kg_pattern = r'(\d+(\.\d+)?)\s*(g|gram|grams|گرام)'

    def convert_kg(match):

        value = float(match.group(1))

        unit = match.group(3)

        if value >= 1000:

            kg = round(value / 1000, 2)

            if kg.is_integer():
                kg = int(kg)

            if unit == "گرام":
                return f"{kg} کلوگرام"

            return f"{kg} kg"

        return match.group(0)

    text = re.sub(
        kg_pattern,
        convert_kg,
        text,
        flags=re.IGNORECASE
    )

    return text

# ==========================================
# Scale Instructions
# ==========================================

def scale_instruction(step,persons):

    return scale_quantity(step,persons)
# ==========================================
# Apply Health Changes
# ==========================================

def modify_health(
    ingredients,
    steps,
    nutrition,
    config,
    health,
    language
):

    ingredient_col = config["ingredient_col"]
    quantity_col = config["quantity_col"]
    step_col = config["step_col"]
    step_no = config["step_no"]
    recipe_col = config["recipe_col"]

    # ==============================
    # Messages
    # ==============================

    messages = {

        "english":{

            "healthy":"Original recipe is recommended.",

            "diabetes":"Recipe modified for Diabetes.",

            "blood_pressure":"Recipe modified for High Blood Pressure.",

            "weight_loss":"Recipe modified for Weight Loss.",

            "weight_gain":"Recipe modified for Weight Gain."

        },

        "urdu":{

            "healthy":"اصل نسخہ آپ کے لیے مناسب ہے۔",

            "diabetes":"ذیابیطس کے لیے نسخے میں تبدیلی کی گئی ہے۔",

            "blood_pressure":"بلند فشار خون کے لیے نسخے میں تبدیلی کی گئی ہے۔",

            "weight_loss":"وزن کم کرنے کے لیے نسخے میں تبدیلی کی گئی ہے۔",

            "weight_gain":"وزن بڑھانے کے لیے نسخے میں تبدیلی کی گئی ہے۔"

        }

    }

    message = messages[language][health]

    # =====================================
    # Diabetes
    # =====================================

    if health == "diabetes":

        for index,row in ingredients.iterrows():

            name = str(row[ingredient_col]).lower()

            if "sugar" in name or "چینی" in name:

                ingredients.at[index,ingredient_col] = (

                    "Stevia"

                    if language=="english"

                    else

                    "اسٹیویا"

                )

                ingredients.at[index,quantity_col] = (

                    "As Required"

                    if language=="english"

                    else

                    "حسبِ ضرورت"

                )

        if language=="english":

            steps[step_col]=steps[step_col].str.replace(

                "Sugar",

                "Stevia",

                case=False,

                regex=False

            )

    # =====================================
    # Blood Pressure
    # =====================================

    elif health=="blood_pressure":

        for index,row in ingredients.iterrows():

            name=str(row[ingredient_col]).lower()

            if "salt" in name or "نمک" in name:

                ingredients.at[index,quantity_col]=(

                    "1/2 tsp"

                    if language=="english"

                    else

                    "آدھا چائے کا چمچ"

                )

            elif "oil" in name or "تیل" in name:

                ingredients.at[index,quantity_col]=(

                    "2 tbsp"

                    if language=="english"

                    else

                    "2 کھانے کے چمچ"

                )

            elif "ghee" in name or "گھی" in name:

                ingredients.at[index,quantity_col]=(

                    "1 tbsp"

                    if language=="english"

                    else

                    "1 کھانے کا چمچ"

                )

        if language=="english":

            steps[step_col]=steps[step_col].str.replace(

                "1 tsp salt",

                "1/2 tsp salt",

                case=False,

                regex=False

            )

            steps[step_col]=steps[step_col].str.replace(

                "5 tbsp oil",

                "2 tbsp oil",

                case=False,

                regex=False

            )
                # =====================================
    # Weight Loss
    # =====================================

    elif health == "weight_loss":

        for index, row in ingredients.iterrows():

            name = str(row[ingredient_col]).lower()

            if "oil" in name or "تیل" in name:

                ingredients.at[index, quantity_col] = (

                    "2 tbsp"

                    if language == "english"

                    else

                    "2 کھانے کے چمچ"

                )

        if language == "english":

            steps[step_col] = steps[step_col].str.replace(

                "Cream",

                "Low Fat Yogurt",

                case=False,

                regex=False

            )

    # =====================================
    # Weight Gain
    # =====================================

    elif health == "weight_gain":

        new_row = {

            recipe_col:

            ingredients.iloc[0][recipe_col],

            ingredient_col:

            "Boiled Egg"

            if language == "english"

            else

            "ابلا ہوا انڈا",

            quantity_col:

            "2 piece"

            if language == "english"

            else

            "2 عدد"

        }

        ingredients = pd.concat(

            [

                ingredients,

                pd.DataFrame([new_row])

            ],

            ignore_index=True

        )

        new_step = {

            recipe_col:

            ingredients.iloc[0][recipe_col],

            step_no:

            len(steps)+1,

            step_col:

            "Serve with 2 boiled eggs for extra protein."

            if language == "english"

            else

            "اضافی پروٹین کے لیے 2 ابلے ہوئے انڈوں کے ساتھ پیش کریں۔"

        }

        steps = pd.concat(

            [

                steps,

                pd.DataFrame([new_step])

            ],

            ignore_index=True

        )

        # ==========================
        # Nutrition Update
        # ==========================

        if nutrition:

            if language == "english":

                nutrition["protein_g"] = (

                    nutrition.get("protein_g",0)

                    +12

                )

                nutrition["calories"] = (

                    nutrition.get("calories",0)

                    +160

                )

            else:

                nutrition["پروٹین_گرام"] = (

                    nutrition.get("پروٹین_گرام",0)

                    +12

                )

                nutrition["کیلوریز"] = (

                    nutrition.get("کیلوریز",0)

                    +160

                )

    # =====================================
    # Return Values
    # =====================================

    return (

        ingredients,

        steps,

        nutrition,

        message

    )
# ==========================================
# Scale Nutrition
# ==========================================

def scale_nutrition(nutrition_df, persons, language, base=2):

    if nutrition_df.empty:
        return {}

    nutrition = nutrition_df.iloc[0].to_dict()

    scale = persons / base

    for key in nutrition:

        try:
            nutrition[key] = round(
                float(nutrition[key]) * scale,
                2
            )
        except:
            pass

    # Urdu Mapping
    if language == "urdu":

        nutrition = {

            "calories":
                nutrition.get("کیلوریز",0),

            "protein_g":
                nutrition.get("پروٹین_گرام",0),

            "carbohydrates_g":
                nutrition.get("کاربوہائیڈریٹس_گرام",0),

            "fat_g":
                nutrition.get("چکنائی_گرام",0),

            "fiber_g":
                nutrition.get("فائبر_گرام",0),

            "sugar_g":
                nutrition.get("شوگر_گرام",0),

            "sodium_mg":
                nutrition.get("سوڈیم_ملی_گرام",0),

            "cholesterol_mg":
                nutrition.get("کولیسٹرول_ملی_گرام",0)

        }

    return nutrition

# ==========================================
# Health Recommendation
# ==========================================

@app.route("/health_recommendation", methods=["POST"])
def health_recommendation():

    data = request.get_json()

    recipe_id = str(data["recipe_id"]).strip()

    persons = int(data["persons"])

    health = data["health"]

    language = data.get(
        "language",
        "english"
    )

    config = load_dataset(language)

    # ==========================
    # Read Dataset
    # ==========================

    ingredients = read_csv_safe(
        config["ingredients"],
        dtype=str
    )

    steps = read_csv_safe(
        config["steps"],
        dtype=str
    )

    nutrition_df = read_csv_safe(
        config["nutrition"],
        dtype=str
    )

    for df in [
        ingredients,
        steps,
        nutrition_df
    ]:

        df.columns = df.columns.str.strip()

        df[
            config["recipe_col"]
        ] = (

            df[
                config["recipe_col"]
            ]

            .astype(str)

            .str.strip()

        )

    # ==========================
    # Filter Recipe
    # ==========================

    ingredients = ingredients[

        ingredients[
            config["recipe_col"]
        ] == recipe_id

    ].copy()

    steps = steps[

        steps[
            config["recipe_col"]
        ] == recipe_id

    ].copy()

    nutrition_df = nutrition_df[

        nutrition_df[
            config["recipe_col"]
        ] == recipe_id

    ].copy()

    # ==========================
    # Scale Ingredients
    # ==========================

    ingredients[
        config["quantity_col"]
    ] = (

        ingredients[
            config["quantity_col"]
        ]

        .apply(

            lambda x:

            scale_quantity(
                x,
                persons
            )

        )

    )

    # ==========================
    # Scale Instructions
    # ==========================

    steps[
        config["step_col"]
    ] = (

        steps[
            config["step_col"]
        ]

        .apply(

            lambda x:

            scale_instruction(
                x,
                persons
            )

        )

    )

    # ==========================
    # Nutrition
    # ==========================

    nutrition = scale_nutrition(

        nutrition_df,

        persons,

        language

    )

    # ==========================
    # Health Changes
    # ==========================

    ingredients, steps, nutrition, message = modify_health(

        ingredients,

        steps,

        nutrition,

        config,

        health,

        language

    )

    # ==========================
    # Sort Steps
    # ==========================

    if config["step_no"] in steps.columns:

        try:

            steps[
                config["step_no"]
            ] = pd.to_numeric(

                steps[
                    config["step_no"]
                ]

            )

            steps = steps.sort_values(

                config["step_no"]

            )

        except:
            pass

    # ==========================
    # Return JSON
    # ==========================

    return jsonify({

        "ingredients":

        ingredients[

            [

                config["ingredient_col"],

                config["quantity_col"]

            ]

        ].rename(

            columns={

                config["ingredient_col"]:

                "ingredient_name",

                config["quantity_col"]:

                "quantity"

            }

        ).to_dict("records"),

        "steps":

        steps[

            [

                config["step_no"],

                config["step_col"]

            ]

        ].rename(

            columns={

                config["step_no"]:

                "step_number",

                config["step_col"]:

                "instruction"

            }

        ).to_dict("records"),

        "nutrition":

        nutrition,

        "message":

        message

    })
# ==========================
# Quantity Formatter
# ==========================

def format_quantity(quantity):

    quantity = str(quantity)

    try:

        parts = quantity.split()

        value = float(parts[0])

        unit = " ".join(parts[1:])

        # Gram to KG

        if unit.lower() == "g":

            if value >= 1000:

                return f"{round(value/1000,2)} kg"

            else:

                return f"{value} g"


        # ml to litre

        elif unit.lower() == "ml":

            if value >= 1000:

                return f"{round(value/1000,2)} litre"

            else:

                return f"{value} ml"


        else:

            return quantity

    except:

        return quantity
# ==========================================
# Common Recipe Recommendation Function
# ==========================================

def get_recipe_recommendations(selected):

    print("\n========== RECIPE RECOMMENDATION ==========")
    print("SELECTED:", selected)

    # =====================================================
    # LOAD DATASETS
    # =====================================================

    ingredients_df = read_csv_safe(
        "data/Pakistani_Food/pakistani_ingredients.csv"
    )

    recipes_df = read_csv_safe(
        "data/Pakistani_Food/pakistani_recipes.csv"
    )

    if ingredients_df is None or recipes_df is None:
        print("ERROR: Recipe datasets could not be loaded.")
        return []

    # =====================================================
    # NORMALIZE SELECTED INGREDIENTS
    # =====================================================

    selected = [
        str(item).strip().lower()
        for item in selected
        if str(item).strip()
    ]

    selected = list(dict.fromkeys(selected))

    print("SELECTED:", selected)

    # =====================================================
    # CATEGORY MAP
    # =====================================================

    category_map = {

        "rice": "Rice",
        "chicken": "Chicken",

        "daal": "Daal",
        "lentils": "Daal",

        "corn": "Corn",
        "makai": "Corn",

        "bhindi": "Bhindi",
        "okra": "Bhindi",

        "egg": "Egg",
        "eggs": "Egg",

        "potato": "Potato",
        "carrot": "Carrot",
        "spinach": "Spinach",

        "chickpeas": "Chickpeas",
        "chana": "Chickpeas",

        "cauliflower": "Cauliflower",
        "cabbage": "Cabbage",

        "yogurt": "Yogurt",

        "capsicum": "Capsicum",
        "onion": "Onion",
        "garlic": "Garlic",
        "ginger": "Ginger",
        "tomato": "Tomato",
        "lemon": "Lemon",
        "peas": "Peas",
        "flour": "Flour",
        "bread": "Bread"
    }

    # =====================================================
    # INGREDIENT NORMALIZATION
    # =====================================================

    replacements = {

        "eggs": "egg",
        "egg": "egg",

        "tomatoes": "tomato",
        "tomato": "tomato",

        "garlic paste": "garlic",
        "garlic": "garlic",

        "ginger paste": "ginger",
        "ginger": "ginger",

        "ginger-garlic": "ginger-garlic",
        "ginger garlic": "ginger-garlic",

        "yogurt (dahi)": "yogurt",
        "dahi": "yogurt",
        "yogurt": "yogurt",

        "sweet corn (makai)": "corn",
        "sweet corn": "corn",
        "makai": "corn",
        "corn": "corn",

        "capsicum (shimla mirch)": "capsicum",
        "shimla mirch": "capsicum",
        "capsicum": "capsicum",

        "spinach (palak)": "spinach",
        "palak": "spinach",
        "spinach": "spinach",

        "cauliflower (gobi)": "cauliflower",
        "gobi": "cauliflower",
        "cauliflower": "cauliflower",

        "cabbage (band gobi)": "cabbage",
        "band gobi": "cabbage",
        "cabbage": "cabbage",

        "potato (aloo)": "potato",
        "aloo": "potato",
        "potato": "potato",

        "okra (bhindi)": "bhindi",
        "okra": "bhindi",
        "bhindi": "bhindi",

        "chickpeas (chana)": "chickpeas",
        "chana": "chickpeas",
        "chickpeas": "chickpeas",

        "lentils (daal)": "daal",
        "lentils": "daal",
        "daal": "daal",

        "chana daal": "daal",
        "mash daal": "daal",
        "moong daal": "daal",
        "mixed daal": "daal",

        "lemon": "lemon",
        "onion": "onion",
        "carrot": "carrot",
        "peas": "peas",

        "flour": "flour",
        "wheat flour": "flour",
        "gram flour": "flour",
        "corn flour": "flour",
        "rice flour": "flour",

        "bread": "bread",

        "chicken stock": "chicken stock"
    }

    # =====================================================
    # NORMALIZE INGREDIENT
    # =====================================================

    def normalize_ingredient(name):

        name = str(name).strip().lower()

        name = re.sub(
            r"\s+",
            " ",
            name
        )

        # Exact match
        if name in replacements:
            return replacements[name]

        # Partial match
        for old, new in replacements.items():

            if old in name:

                name = name.replace(
                    old,
                    new
                )

        return name.strip()

    # =====================================================
    # NORMALIZED SELECTED INGREDIENTS
    # =====================================================

    normalized_selected = []

    for item in selected:

        normalized = normalize_ingredient(item)

        if normalized not in normalized_selected:

            normalized_selected.append(
                normalized
            )

    print(
        "NORMALIZED SELECTED:",
        normalized_selected
    )

    # =====================================================
    # SELECTED CATEGORIES
    # =====================================================

    selected_categories = []

    for item in normalized_selected:

        if item in category_map:

            category = category_map[item]

            if category not in selected_categories:

                selected_categories.append(
                    category
                )

    print(
        "SELECTED CATEGORIES:",
        selected_categories
    )

    # =====================================================
    # STOP IF NO MAIN INGREDIENT
    # =====================================================

    if len(selected_categories) == 0:

        print("NO MAIN INGREDIENT FOUND")

        return []

    # =====================================================
    # PRIMARY INGREDIENT
    # =====================================================

    recipes_df["primary_ingredient"] = (

        recipes_df["primary_ingredient"]
        .astype(str)
        .str.strip()

    )

    # =====================================================
    # PRIMARY MAP
    # =====================================================

    primary_map = {

        "chicken": "chicken",
        "rice": "rice",

        "daal": "daal",
        "lentils": "daal",

        "corn": "corn",

        "okra": "bhindi",
        "bhindi": "bhindi",

        "egg": "egg",

        "potato": "potato",
        "carrot": "carrot",
        "spinach": "spinach",

        "chickpeas": "chickpeas",

        "cauliflower": "cauliflower",
        "cabbage": "cabbage",

        "yogurt": "yogurt",
        "bread": "bread"
    }

    # =====================================================
    # REQUIRED COMBINATIONS
    # =====================================================

    required_combinations = {

        "chicken biryani": [
            "chicken",
            "rice"
        ],

        "chicken pulao": [
            "chicken",
            "rice"
        ],

        "yakhni pulao": [
            "chicken",
            "rice"
        ],

        "sindhi biryani": [
            "chicken",
            "rice",
            "potato"
        ],

        "chana pulao": [
            "chickpeas",
            "rice"
        ],

        "corn pulao": [
            "corn",
            "rice"
        ],

        "vegetable pulao": [
            "rice"
        ],

        "masala rice": [
            "rice"
        ],

        "chicken corn soup": [
            "chicken",
            "corn"
        ],

        "hot and sour soup": [
            "chicken"
        ],

        "chicken palak": [
            "chicken",
            "spinach"
        ],

        "palak chicken": [
            "chicken",
            "spinach"
        ],

        "daal palak": [
            "daal",
            "spinach"
        ],

        "palak daal": [
            "daal",
            "spinach"
        ],

        "aloo gobi": [
            "potato",
            "cauliflower"
        ],

        "aloo palak": [
            "potato",
            "spinach"
        ],

        "mix vegetable": [
            "carrot",
            "potato",
            "cauliflower"
        ],

        "vegetable handi": [
            "carrot",
            "potato",
            "cauliflower"
        ],

        "creamy vegetable soup": [
            "carrot",
            "potato",
            "cauliflower"
        ],

        "dahi aloo": [
            "potato",
            "yogurt"
        ],

        "dahi chicken": [
            "chicken",
            "yogurt"
        ],

        "palak paneer": [
            "spinach"
        ]
    }

    # =====================================================
    # DEFAULT COMBINATIONS
    # =====================================================

    default_combinations = {

        "chicken biryani":
            "Raita, Kachumber, Salad",

        "chicken pulao":
            "Raita, Salad, Chutney",

        "yakhni pulao":
            "Raita, Salad",

        "sindhi biryani":
            "Raita, Kachumber, Salad",

        "chana pulao":
            "Raita, Salad, Pickle",

        "corn pulao":
            "Raita, Salad, Chutney",

        "vegetable pulao":
            "Raita, Salad, Pickle",

        "masala rice":
            "Raita, Salad, Chutney",

        "chicken corn soup":
            "Bread, Salad",

        "hot and sour soup":
            "Bread, Spring Rolls",

        "chicken palak":
            "Raita, Salad",

        "palak chicken":
            "Raita, Salad",

        "daal palak":
            "Rice, Roti, Salad",

        "palak daal":
            "Rice, Roti, Salad",

        "aloo gobi":
            "Roti, Raita, Salad",

        "aloo palak":
            "Roti, Raita, Salad",

        "mix vegetable":
            "Roti, Raita, Salad",

        "vegetable handi":
            "Naan, Raita, Salad",

        "creamy vegetable soup":
            "Bread, Salad",

        "dahi aloo":
            "Roti, Salad, Pickle",

        "dahi chicken":
            "Naan, Raita, Salad",

        "palak paneer":
            "Roti, Naan, Salad"
    }

    # =====================================================
    # SIMILAR RECIPE GROUPS
    #
    # Recipes in the same group are treated as duplicates.
    #
    # Example:
    # Daal Palak + Palak Daal
    # Only ONE will be returned.
    # =====================================================

    similar_recipe_groups = [

        {
            "daal palak",
            "palak daal"
        },

        {
            "chicken palak",
            "palak chicken"
        },

        {
            "aloo palak",
            "palak aloo"
        },

        {
            "chicken biryani"
        },

        {
            "chicken pulao"
        }
    ]

    # =====================================================
    # CREATE RECIPE GROUP KEY
    # =====================================================

    def get_recipe_group(recipe_name):

        recipe_name = (
            str(recipe_name)
            .lower()
            .strip()
            .replace(".", "")
        )

        # Check predefined groups
        for group in similar_recipe_groups:

            if recipe_name in group:

                return "|".join(
                    sorted(group)
                )

        # -------------------------------------------------
        # Generic normalization
        # -------------------------------------------------

        words = recipe_name.split()

        # For two-word recipes where order is reversed
        # treat them as same recipe.
        if len(words) == 2:

            return "|".join(
                sorted(words)
            )

        return recipe_name

    # =====================================================
    # RESULTS
    # =====================================================

    results = []

    # =====================================================
    # LOOP RECIPES
    # =====================================================

    for _, recipe in recipes_df.iterrows():

        recipe_id = str(
            recipe["recipe_id"]
        ).strip()

        recipe_name = str(
            recipe["recipe_name"]
        ).strip()

        recipe_name_clean = (

            recipe_name
            .lower()
            .strip()
            .replace(".", "")

        )

        # =================================================
        # PRIMARY INGREDIENT
        # =================================================

        primary_raw = str(
            recipe["primary_ingredient"]
        ).strip()

        primary_parts = [

            x.strip()

            for x in primary_raw.split(",")

            if x.strip()

        ]

        primary_ingredients = []

        for primary_part in primary_parts:

            normalized_primary = normalize_ingredient(
                primary_part
            )

            if normalized_primary in primary_map:

                normalized_primary = primary_map[
                    normalized_primary
                ]

            primary_ingredients.append(
                normalized_primary
            )

        # =================================================
        # PRIMARY MUST BE AVAILABLE
        # =================================================

        primary_available = False

        for primary in primary_ingredients:

            if primary in normalized_selected:

                primary_available = True

                break

        if not primary_available:

            continue

        # =================================================
        # RECIPE INGREDIENTS
        # =================================================

        recipe_ing = ingredients_df[

            ingredients_df["recipe_id"]
            .astype(str)
            .str.strip()
            == recipe_id

        ]["ingredient_name"] \
            .astype(str) \
            .str.strip() \
            .tolist()

        # =================================================
        # NORMALIZE RECIPE INGREDIENTS
        # =================================================

        normalized_recipe = []

        original_recipe_ingredients = []

        for ing in recipe_ing:

            original_name = str(
                ing
            ).strip()

            normalized = normalize_ingredient(
                original_name
            )

            if normalized:

                if normalized not in normalized_recipe:

                    normalized_recipe.append(
                        normalized
                    )

                    original_recipe_ingredients.append(
                        original_name
                    )

        # =================================================
        # RECIPE KEYWORDS
        # =================================================

        keyword_list = []

        if "recipe_keywords" in recipes_df.columns:

            keyword_text = str(
                recipe.get(
                    "recipe_keywords",
                    ""
                )
            )

            for keyword in keyword_text.split(","):

                keyword = normalize_ingredient(
                    keyword
                )

                if keyword:

                    keyword_list.append(
                        keyword
                    )

        # =================================================
        # REQUIRED COMBINATION
        # =================================================

        required = required_combinations.get(
            recipe_name_clean,
            []
        )

        # =================================================
        # REQUIRED INGREDIENT CHECK
        # =================================================

        missing_required = []

        for required_item in required:

            if required_item not in normalized_selected:

                missing_required.append(
                    required_item
                )

        # =================================================
        # REQUIRED INGREDIENT MISSING
        # =================================================

        if len(missing_required) > 0:

            continue

        # =================================================
        # MATCH INGREDIENTS
        # =================================================

        matched_items = []

        for selected_item in normalized_selected:

            # ---------------------------------------------
            # Recipe ingredient matching
            # ---------------------------------------------

            for recipe_item in normalized_recipe:

                if (

                    selected_item == recipe_item

                    or selected_item in recipe_item

                    or recipe_item in selected_item

                ):

                    if selected_item not in matched_items:

                        matched_items.append(
                            selected_item
                        )

                    break

            # ---------------------------------------------
            # Keyword matching
            # ---------------------------------------------

            if selected_item not in matched_items:

                for keyword in keyword_list:

                    if (

                        selected_item == keyword

                        or selected_item in keyword

                        or keyword in selected_item

                    ):

                        matched_items.append(
                            selected_item
                        )

                        break

        # =================================================
        # MATCH COUNT
        # =================================================

        matched = len(
            matched_items
        )

        # =================================================
        # TOTAL INGREDIENTS
        # =================================================

        total_ingredients = len(
            set(normalized_recipe)
        )

        if total_ingredients == 0:

            total_ingredients = 1

        # =================================================
        # MATCH PERCENTAGE
        # =================================================

        percentage = (

            matched /
            total_ingredients
        ) * 100

        # =================================================
        # REQUIRED MATCH COUNT
        # =================================================

        required_matched = 0

        for required_item in required:

            if required_item in normalized_selected:

                required_matched += 1

        # =================================================
        # FINAL SCORE
        # =================================================

        final_score = percentage

        # Small bonus for satisfying required ingredients
        final_score += (
            required_matched * 10
        )

        # =================================================
        # RECOMMENDED INGREDIENTS
        # =================================================

        recommended_ingredients = []

        for index, normalized_item in enumerate(
            normalized_recipe
        ):

            if normalized_item not in normalized_selected:

                if index < len(
                    original_recipe_ingredients
                ):

                    display_name = (
                        original_recipe_ingredients[index]
                    )

                else:

                    display_name = normalized_item

                if display_name not in recommended_ingredients:

                    recommended_ingredients.append(
                        display_name
                    )

        # Maximum 15
        recommended_ingredients = (
            recommended_ingredients[:15]
        )

        # =================================================
        # COMBINATION
        # =================================================

        combination = str(
            recipe.get(
                "combination",
                ""
            )
        ).strip()

        if not combination:

            combination = default_combinations.get(
                recipe_name_clean,
                ""
            )

        # =================================================
        # RECIPE GROUP
        # =================================================

        recipe_group = get_recipe_group(
            recipe_name_clean
        )

        # =================================================
        # ADD RESULT
        # =================================================

        results.append({

            "recipe_id":
                recipe_id,

            "recipe_name":
                recipe_name,

            "score":
                f"{matched}/{total_ingredients}",

            "matched_ingredients":
                matched,

            "total_ingredients":
                total_ingredients,

            "percentage":
                round(
                    percentage,
                    2
                ),

            "final_score":
                round(
                    final_score,
                    2
                ),

            "combination":
                combination,

            "matched":
                matched_items,

            "required":
                required,

            "missing":
                missing_required,

            "recommended_ingredients":
                recommended_ingredients,

            "primary_ingredient":
                primary_raw,

            "primary_available":
                primary_available,

            "_recipe_group":
                recipe_group

        })

        print(
            recipe_name,
            "| Matched:",
            f"{matched}/{total_ingredients}",
            "| Percentage:",
            round(percentage, 2),
            "| Final:",
            round(final_score, 2)
        )

    # =====================================================
    # SORT
    #
    # IMPORTANT:
    # First priority = NUMBER OF MATCHED INGREDIENTS
    # Second priority = MATCH PERCENTAGE
    # Third priority = FINAL SCORE
    # =====================================================

    results = sorted(

        results,

        key=lambda x: (

            x["matched_ingredients"],

            x["percentage"],

            x["final_score"]

        ),

        reverse=True

    )

    # =====================================================
    # REMOVE DUPLICATE / SIMILAR RECIPES
    # =====================================================

    unique_results = []

    used_groups = set()

    for result in results:

        group = result["_recipe_group"]

        if group in used_groups:

            print(
                "DUPLICATE/SIMILAR REMOVED:",
                result["recipe_name"]
            )

            continue

        used_groups.add(group)

        unique_results.append(
            result
        )

        # -----------------------------------------------
        # We only need TOP 3
        # -----------------------------------------------

        if len(unique_results) >= 3:

            break

    # =====================================================
    # REMOVE INTERNAL GROUP KEY
    # =====================================================

    for result in unique_results:

        result.pop(
            "_recipe_group",
            None
        )

    # =====================================================
    # FINAL OUTPUT
    # =====================================================

    print(
        "\n========== TOP 3 RECOMMENDATIONS =========="
    )

    for index, result in enumerate(
        unique_results,
        start=1
    ):

        print(
            index,
            ".",
            result["recipe_name"],
            "| Match:",
            result["score"],
            "| Percentage:",
            result["percentage"],
            "%"
        )

    print(
        "TOTAL RETURNED:",
        len(unique_results)
    )

    return unique_results
# -------------------------------
# Configuration
# -------------------------------
app.config["SECRET_KEY"] = "smart_cooking_assistant_secret"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# -------------------------------
# Database Model
# -------------------------------
# ==========================================
# USER DATABASE MODEL
# ==========================================

class User(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    name = db.Column(
        db.String(100),
        nullable=False
    )

    email = db.Column(
        db.String(120),
        unique=True,
        nullable=False
    )

    password = db.Column(
        db.String(255),
        nullable=False
    )

    reset_token = db.Column(
        db.String(255),
        nullable=True
    )

    reset_token_expiry = db.Column(
        db.DateTime,
        nullable=True
    )
    # ==========================================
# DATABASE MIGRATION
# ==========================================

from sqlalchemy import inspect, text

with app.app_context():

    db.create_all()

    inspector = inspect(db.engine)

    columns = [
        column["name"]
        for column in inspector.get_columns("user")
    ]

    with db.engine.begin() as connection:

        if "reset_token" not in columns:

            connection.execute(
                text(
                    "ALTER TABLE user "
                    "ADD COLUMN reset_token VARCHAR(255)"
                )
            )

            print("✅ reset_token added")

        if "reset_token_expiry" not in columns:

            connection.execute(
                text(
                    "ALTER TABLE user "
                    "ADD COLUMN reset_token_expiry DATETIME"
                )
            )

            print("✅ reset_token_expiry added")

    print("✅ Database ready")
# ==========================================
# STRONG PASSWORD VALIDATION
# ==========================================

def validate_password(password):

    errors = []

    # Minimum 8 characters
    if len(password) < 8:
        errors.append(
            "Password must be at least 8 characters long."
        )

    # Uppercase
    if not re.search(r"[A-Z]", password):
        errors.append(
            "Password must contain at least one uppercase letter."
        )

    # Lowercase
    if not re.search(r"[a-z]", password):
        errors.append(
            "Password must contain at least one lowercase letter."
        )

    # Number
    if not re.search(r"[0-9]", password):
        errors.append(
            "Password must contain at least one number."
        )

    # Special character
    if not re.search(r"[^A-Za-z0-9]", password):
        errors.append(
            "Password must contain at least one special character."
        )

    return errors
# ==========================================
# SEND PASSWORD RESET EMAIL
# ==========================================

def send_reset_email(receiver_email, reset_link):

    try:

        message = MIMEMultipart()

        message["From"] = GMAIL_ADDRESS

        message["To"] = receiver_email

        message["Subject"] = (
            "AI Smart Cooking Assistant - "
            "Password Reset"
        )

        body = f"""
Hello,

We received a request to reset your password
for AI Based Smart Cooking Assistant.

Click the link below to reset your password:

{reset_link}

This link will expire in 15 minutes.

If you did not request a password reset,
please ignore this email.

Regards,
AI Based Smart Cooking Assistant
"""

        message.attach(
            MIMEText(
                body,
                "plain",
                "utf-8"
            )
        )

        # Connect Gmail SMTP
        server = smtplib.SMTP(
            SMTP_SERVER,
            SMTP_PORT
        )

        server.starttls()

        # Login using Gmail App Password
        server.login(
            GMAIL_ADDRESS,
            GMAIL_APP_PASSWORD
        )

        # Send email
        server.sendmail(
            GMAIL_ADDRESS,
            receiver_email,
            message.as_string()
        )

        server.quit()

        print(
            "Password reset email sent to:",
            receiver_email
        )

        return True

    except Exception as e:

        print(
            "Email sending error:",
            e
        )

        return False
import os
import cv2
import pandas as pd
from flask import request, jsonify
from werkzeug.utils import secure_filename


# ==========================================
# INGREDIENT DETECTION
# ==========================================

@app.route("/detect", methods=["POST"])
def detect():

    print("\n======================================")
    print("        INGREDIENT DETECTION")
    print("======================================")

    # =========================================================
    # CHECK IMAGE
    # =========================================================

    if "image" not in request.files:

        return jsonify({
            "status": "error",
            "message": "No image uploaded."
        }), 400

    file = request.files["image"]

    if file.filename == "":

        return jsonify({
            "status": "error",
            "message": "No file selected."
        }), 400

    # =========================================================
    # SAVE IMAGE
    # =========================================================

    filename = secure_filename(file.filename)

    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(image_path)

    print("Image saved:", image_path)

    # =========================================================
    # FIRST: MULTIPLE MODEL
    # =========================================================

    try:

        multiple_results = multiple_model.predict(
            source=image_path,
            conf=0.30,
            verbose=False
        )
    
    except Exception as e:

        print("Multiple model error:", e)

        return jsonify({
            "status": "error",
            "message": "Multiple ingredient model failed."
        }), 500

    # =========================================================
    # FIND DIFFERENT DETECTED CLASSES
    # =========================================================

    detected_classes = set()

    for result in multiple_results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_id = int(box.cls[0])

            detected_classes.add(class_id)

    print(
        "Multiple model detected classes:",
        detected_classes
    )

    # =========================================================
    # MODEL SELECTION
    # =========================================================

    if len(detected_classes) >= 2:

        model = multiple_model

        results = multiple_results

        model_type = "multiple"

        print("--------------------------------------")
        print("🍲 MULTIPLE INGREDIENT IMAGE")
        print("Using MULTIPLE model")
        print("--------------------------------------")

    else:

        model = single_model

        try:

            results = single_model.predict(
                source=image_path,
                conf=0.75,
                verbose=False
            )

        except Exception as e:

            print("Single model error:", e)

            return jsonify({
                "status": "error",
                "message": "Single ingredient model failed."
            }), 500

        model_type = "single"

        print("--------------------------------------")
        print("🥕 SINGLE INGREDIENT IMAGE")
        print("Using SINGLE model")
        print("--------------------------------------")

    # =========================================================
    # CHECK DETECTION
    # =========================================================

    if not results:

        return jsonify({
            "status": "error",
            "message": "No detection result."
        }), 200

    if results[0].boxes is None:

        return jsonify({
            "status": "error",
            "message": "No ingredients detected."
        }), 200

    if len(results[0].boxes) == 0:

        return jsonify({
            "status": "error",
            "message": "No ingredients detected."
        }), 200

    # =========================================================
    # SAVE ANNOTATED IMAGE
    # =========================================================

    os.makedirs(
        "static",
        exist_ok=True
    )

    annotated = results[0].plot()

    result_path = os.path.join(
        "static",
        "result.jpg"
    )

    cv2.imwrite(
        result_path,
        annotated
    )

    # =========================================================
    # GET DETECTED INGREDIENTS
    # =========================================================

    ingredients = []

    for box in results[0].boxes:

        class_id = int(box.cls[0])

        confidence = float(box.conf[0])

        ingredient = model.names[class_id]

        ingredients.append({

            "name": ingredient,

            "confidence": round(
                confidence * 100,
                2
            )

        })

    # =========================================================
    # REMOVE DUPLICATES
    # KEEP HIGHEST CONFIDENCE
    # =========================================================

    unique = {}

    for item in ingredients:

        ingredient_name = item["name"]

        if ingredient_name not in unique:

            unique[ingredient_name] = item

        elif (
            item["confidence"]
            >
            unique[ingredient_name]["confidence"]
        ):

            unique[ingredient_name] = item

    ingredients = list(
        unique.values()
    )

    # =========================================================
    # SORT BY CONFIDENCE
    # =========================================================

    ingredients.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    # =========================================================
    # DETECTED NAMES
    # =========================================================

    detected = [

        item["name"]
        .lower()
        .strip()

        for item in ingredients

    ]

    print("--------------------------------------")
    print("DETECTED INGREDIENTS:")
    print(detected)
    print("--------------------------------------")

    # =========================================================
    # RECIPE RECOMMENDATION
    # =========================================================

    try:

        recipes = get_recipe_recommendations(
            detected
        )

    except Exception as e:

        print("Recipe recommendation error:", e)

        return jsonify({

            "status": "error",

            "message":
                "Ingredients detected, but recipe recommendation failed.",

            "ingredients":
                ingredients,

            "image":
                "/static/result.jpg"

        }), 500

    # =========================================================
    # FORMAT RECIPES
    # =========================================================

    formatted_recipes = []

    for recipe in recipes:

        recipe_name = recipe.get(
            "recipe_name",
            recipe.get(
                "name",
                "Unknown Recipe"
            )
        )

        score = recipe.get(
            "score",
            recipe.get(
                "match_score",
                0
            )
        )

        matched_ingredients = recipe.get(
            "matched_ingredients",
            recipe.get(
                "matched",
                0
            )
        )

        total_ingredients = recipe.get(
            "total_ingredients",
            recipe.get(
                "total",
                0
            )
        )

        recommended_ingredients = recipe.get(
            "recommended_ingredients",
            []
        )

        combination = recipe.get(
            "combination",
            recipe.get(
                "best_combination",
                ""
            )
        )

        # --------------------------------------
        # Recommended ingredients → list
        # --------------------------------------

        if isinstance(
            recommended_ingredients,
            str
        ):

            recommended_ingredients = [

                x.strip()

                for x in
                recommended_ingredients.split(",")

                if x.strip()

            ]

        # --------------------------------------
        # Combination → string
        # --------------------------------------

        if isinstance(
            combination,
            list
        ):

            combination = ", ".join(
                str(x)
                for x in combination
            )

        formatted_recipes.append({

            "recipe_name":
                recipe_name,

            "score":
                score,

            "matched_ingredients":
                matched_ingredients,

            "total_ingredients":
                total_ingredients,

            "recommended_ingredients":
                recommended_ingredients,

            "combination":
                combination

        })

    # =========================================================
    # DEBUG
    # =========================================================

    print("--------------------------------------")
    print("RECOMMENDED RECIPES:")

    for recipe in formatted_recipes:

        print(
            recipe["recipe_name"],
            "|",
            recipe["score"],
            "| Matched:",
            recipe["matched_ingredients"],
            "/",
            recipe["total_ingredients"]
        )

    print("--------------------------------------")

    # =========================================================
    # RETURN JSON
    # =========================================================

    return jsonify({

        "status":
            "success",

        "model":
            model_type,

        "image":
            "/static/result.jpg",

        "ingredients":
            ingredients,

        "recipes":
            formatted_recipes

    })
# Home
# -------------------------------
@app.route("/")
def home():
    return redirect(url_for("login"))

# -------------------------------
# Register
# -------------------------------
# ==========================================
# REGISTER
# ==========================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if request.method == "POST":

        name = request.form.get(
            "name",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        # -----------------------------
        # Check existing user
        # -----------------------------

        user = User.query.filter_by(
            email=email
        ).first()

        if user:

            return render_template(
                "register.html",
                error="Email already exists."
            )

        # -----------------------------
        # Strong password
        # -----------------------------

        errors = validate_password(
            password
        )

        if errors:

            return render_template(
                "register.html",
                errors=errors
            )

        # -----------------------------
        # Hash password
        # -----------------------------

        hashed_password = (
            bcrypt
            .generate_password_hash(password)
            .decode("utf-8")
        )

        # -----------------------------
        # Create user
        # -----------------------------

        new_user = User(
            name=name,
            email=email,
            password=hashed_password
        )

        db.session.add(
            new_user
        )

        db.session.commit()

        return redirect(
            url_for("login")
        )

    return render_template(
        "register.html"
    )

# -------------------------------
# Login
# -------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form["email"]
        password = request.form["password"]

        user = User.query.filter_by(email=email).first()

        if user and bcrypt.check_password_hash(user.password, password):

            session["username"] = user.name

            return redirect(url_for("dashboard"))

        else:
            return "Invalid Email or Password"

    return render_template("login.html")
# ==========================================
# FORGOT PASSWORD
# ==========================================

@app.route(
    "/forgot-password",
    methods=["GET", "POST"]
)
def forgot_password():

    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        user = User.query.filter_by(
            email=email
        ).first()

        # -----------------------------
        # User does not exist
        # -----------------------------

        if not user:

            return render_template(
                "forgot_password.html",
                error="No account found with this email address."
            )

        # -----------------------------
        # Generate secure token
        # -----------------------------

        token = secrets.token_urlsafe(32)

        # -----------------------------
        # Token expiry = 15 minutes
        # -----------------------------

        expiry = (
            datetime.utcnow()
            + timedelta(minutes=15)
        )

        user.reset_token = token

        user.reset_token_expiry = expiry

        db.session.commit()

        # -----------------------------
        # Create reset link
        # -----------------------------

        reset_link = url_for(
            "reset_password",
            token=token,
            _external=True
        )

        # -----------------------------
        # Send email
        # -----------------------------

        email_sent = send_reset_email(
            user.email,
            reset_link
        )

        if not email_sent:

            return render_template(
                "forgot_password.html",
                error=(
                    "Unable to send email. "
                    "Please try again later."
                )
            )

        return render_template(
            "forgot_password.html",
            success=(
                "Password reset link has been "
                "sent to your email."
            )
        )

    return render_template(
        "forgot_password.html"
    )
# ==========================================
# RESET PASSWORD
# ==========================================

@app.route(
    "/reset-password/<token>",
    methods=["GET", "POST"]
)
def reset_password(token):

    # -----------------------------
    # Find user
    # -----------------------------

    user = User.query.filter_by(
        reset_token=token
    ).first()

    # -----------------------------
    # Invalid token
    # -----------------------------

    if not user:

        return """
        <h2>Invalid Password Reset Link</h2>
        <p>This password reset link is invalid.</p>
        """

    # -----------------------------
    # Check expiry
    # -----------------------------

    if (
        user.reset_token_expiry is None
        or
        datetime.utcnow()
        > user.reset_token_expiry
    ):

        return """
        <h2>Password Reset Link Expired</h2>
        <p>Please request a new password reset link.</p>
        """

    # -----------------------------
    # POST
    # -----------------------------

    if request.method == "POST":

        password = request.form.get(
            "password",
            ""
        )

        confirm_password = request.form.get(
            "confirm_password",
            ""
        )

        # -----------------------------
        # Password match
        # -----------------------------

        if password != confirm_password:

            return render_template(
                "reset_password.html",
                error="Passwords do not match.",
                token=token
            )

        # -----------------------------
        # Strong password validation
        # -----------------------------

        errors = validate_password(
            password
        )

        if errors:

            return render_template(
                "reset_password.html",
                errors=errors,
                token=token
            )

        # -----------------------------
        # Hash new password
        # -----------------------------

        hashed_password = (
            bcrypt
            .generate_password_hash(password)
            .decode("utf-8")
        )

        user.password = hashed_password

        # -----------------------------
        # Remove reset token
        # -----------------------------

        user.reset_token = None

        user.reset_token_expiry = None

        db.session.commit()

        return redirect(
            url_for("login")
        )

    return render_template(
        "reset_password.html",
        token=token
    )

# -------------------------------
# Dashboard
# -------------------------------
@app.route("/dashboard")
def dashboard():

    if "username" not in session:
        return redirect(url_for("login"))

    return render_template(
        "dashboard.html",
        username=session["username"]
    )
@app.route("/balti")
def balti_dish():
    return render_template("balti.html")

# -------------------------------
# Logout
# -------------------------------
@app.route("/logout")
def logout():

    session.pop("username", None)

    return redirect(url_for("login"))
@app.route("/recipe/<food>")
def recipe(food):

    import pandas as pd

    # ==========================
    # Read English CSV
    # ==========================

    english_df = read_csv_safe(
    "data/recipes.csv",
    encoding="latin1",
    dtype=str
   )

    urdu_df = read_csv_safe(
    "data/recipes_urdu.csv",
    encoding="utf-8-sig",
    dtype=str
)

    # ==========================
    # Remove Spaces
    # ==========================

    english_df.columns = english_df.columns.str.strip()
    urdu_df.columns = urdu_df.columns.str.strip()

    english_df["Food Name"] = (
        english_df["Food Name"]
        .astype(str)
        .str.strip()
    )

    urdu_df["Food Name"] = (
        urdu_df["Food Name"]
        .astype(str)
        .str.strip()
    )

    english_df["S.No"] = (
        english_df["S.No"]
        .astype(str)
        .str.strip()
    )

    urdu_df["S.No"] = (
        urdu_df["S.No"]
        .astype(str)
        .str.strip()
    )

    food = food.strip()

    # ==========================
    # Find English Recipe
    # ==========================

    recipe = english_df[
        english_df["Food Name"].str.lower() == food.lower()
    ]

    if recipe.empty:

        print("Recipe Not Found")
        print("URL Food :", food)

        return "Recipe not found"

    recipe = recipe.iloc[0]

    # ==========================
    # Get Recipe Number
    # ==========================

    recipe_no = recipe["S.No"]

    # ==========================
    # Find Urdu Recipe
    # ==========================

    urdu_recipe = urdu_df[
        urdu_df["S.No"] == recipe_no
    ]

    if urdu_recipe.empty:

        urdu_food = recipe["Food Name"]

    else:

        urdu_food = urdu_recipe.iloc[0]["Food Name"]

    # ==========================
    # Debug
    # ==========================

    print("========== DEBUG ==========")
    print("Food :", food)
    print("Recipe No :", recipe_no)
    print("English :", recipe["Food Name"])
    print("Urdu :", urdu_food)
    print("===========================")

    # ==========================
    # Open HTML
    # ==========================

    return render_template(

        "recipe.html",

        food=recipe["Food Name"],

        urdu_food=urdu_food,

        recipe_no=recipe_no

    )
@app.route("/get_recipe/<recipe_no>")
def get_recipe(recipe_no):

    # ==========================
    # Current Language
    # ==========================

    language = request.args.get(
        "language",
        "english"
    )

    # ==========================
    # Select CSV
    # ==========================

    if language == "urdu":
        file_name = "data/recipes_urdu.csv"
    else:
        file_name = "data/recipes.csv"

    # ==========================
    # Read CSV Safely
    # ==========================

    recipe_df = read_csv_safe(
        file_name,
        dtype=str
    )

    recipe_df.columns = recipe_df.columns.astype(str).str.strip()

    # ==========================
    # Find Recipe using S.No
    # ==========================

    for _, row in recipe_df.iterrows():

        row = {
            str(key).strip(): str(value).strip()
            for key, value in row.items()
        }

        if str(row.get("S.No", "")).strip() == str(recipe_no).strip():

            return jsonify({
                "Food Name": row.get("Food Name", ""),
                "ingredients": row.get("Ingredients", ""),
                "instruction": row.get("Instruction", "")
            })

    # ==========================
    # Recipe Not Found
    # ==========================

    return jsonify({
        "error": "Recipe not found"
    })
@app.route("/ingredients")
def ingredients():
    return render_template("ingredients.html")
@app.route("/find_recipe", methods=["POST"])
def find_recipe():

    print("\n========== FIND RECIPE ==========")

    data = request.get_json()

    selected = [

        str(i)
        .replace("✕", "")
        .strip()
        .lower()

        for i in data.get("ingredients", [])

    ]

    print("SELECTED:", selected)

    # ==========================
    # Common Recommendation Function
    # ==========================

    recipes = get_recipe_recommendations(selected)

    # ==========================
    # Return Result
    # ==========================

    return jsonify({

        "recipes": recipes

    })

@app.route("/missing_ingredients", methods=["POST"])
def missing_ingredients():

    data = request.get_json()

    selected = [
        i.lower().strip()
        for i in data["ingredients"]
    ]

    foods = data["foods"]


    ingredients_df = read_csv_safe(
        "data/Pakistani_Food/pakistani_ingredients.csv"
    )

    recipes_df = read_csv_safe(
        "data/Pakistani_Food/pakistani_recipes.csv"
    )


    result = {}


    for food in foods:

        recipe = recipes_df[
            recipes_df["recipe_name"] == food
        ]


        if not recipe.empty:

            recipe_id = recipe.iloc[0]["recipe_id"]

            required = ingredients_df[

                ingredients_df["recipe_id"] == recipe_id

            ]["ingredient_name"].tolist()


            missing = []


            for item in required:

                if item.lower() not in selected:

                    missing.append(item)


            result[food] = missing


    return jsonify(result)
@app.route("/pakistani_recipe/<food>")
def pakistani_recipe(food):

    # Number of persons selected from popup
    persons = int(request.args.get("persons", 2))

# Maximum persons limit

    if persons > 10:
      persons = 10

# Minimum limit

    if persons < 1:
      persons = 1

    # Base recipe is for 2 persons
    base_persons = 2

    recipes_df = read_csv_safe(
        "data/Pakistani_Food/pakistani_recipes.csv"
    )

    ingredients_df = read_csv_safe(
        "data/Pakistani_Food/pakistani_ingredients.csv"
    )

    steps_df = read_csv_safe(
        "data/Pakistani_Food/pakistani_steps.csv"
    )

    recipe = recipes_df[
        recipes_df["recipe_name"].str.strip() == food.strip()
    ]

    if recipe.empty:
        return "Recipe Not Found"

    recipe_id = recipe.iloc[0]["recipe_id"]

    ingredients = ingredients_df[
        ingredients_df["recipe_id"] == recipe_id
    ].copy()


    # ==========================
    # Change Ingredient Quantity
    # ==========================

    for index, row in ingredients.iterrows():

        quantity = str(row["quantity"])

        try:

            parts = quantity.split()

            value = float(parts[0])

            unit = " ".join(parts[1:])

            # Person scaling

            new_value = round(
                (value * persons) / base_persons,
                2
            )

            converted_quantity = f"{new_value} {unit}"

            # Convert g -> kg, ml -> litre

            ingredients.at[index, "quantity"] = format_quantity(
                converted_quantity
            )

        except:

            # Keep text values unchanged
            ingredients.at[index, "quantity"] = quantity



    steps = steps_df[
        steps_df["recipe_id"] == recipe_id
    ].sort_values("step_number")

    return render_template(

    "pakistani_recipe.html",

    food_name=food,

    recipe_id=recipe_id,

    persons=persons,

    ingredients=ingredients.to_dict("records"),

    steps=steps.to_dict("records")

    )
from flask import request, jsonify

# ==========================================
# Health Recommendation
# ==========================================

@app.route("/ingredient_detection")
def ingredient_detection():

    return render_template(
        "ingredient_detection.html"
    )
@app.route('/pakistani-recipes')
def pakistani_recipes():

    return render_template(
        "pakistani_recipesright.html"
    )

# ===============================
# Search Recipe From CSV
# ===============================

@app.route("/get-recipe", methods=["POST"])
def search_recipe():

    data = request.get_json()

    food = data["food"].strip().lower()

    # Read CSV files

    recipes_df = read_csv_safe(
        "data/Pakistani_Food/pakistani_recipes.csv"
    )

    ingredients_df = read_csv_safe(
        "data/Pakistani_Food/pakistani_ingredients.csv"
    )

    steps_df = read_csv_safe(
        "data/Pakistani_Food/pakistani_steps.csv"
    )

    # Find Recipe

    recipe = recipes_df[
        recipes_df["recipe_name"]
        .str.lower()
        .str.contains(food)
    ]

    if recipe.empty:

        return jsonify({

            "status":"error",

            "message":"Recipe not found"

        })

    recipe = recipe.iloc[0]

    recipe_id = recipe["recipe_id"]


    # Get Ingredients

    ingredients = ingredients_df[

        ingredients_df["recipe_id"] == recipe_id

    ]

    ingredient_text = ""

    for _,row in ingredients.iterrows():

        ingredient_text += (

            "• "
            + str(row["ingredient_name"])
            + " - "
            + str(row["quantity"])
            + "<br>"

        )


    # Get Instructions

    steps = steps_df[

        steps_df["recipe_id"] == recipe_id

    ].sort_values("step_number")


    instruction_text = ""

    for _,row in steps.iterrows():

        instruction_text += (

            str(row["step_number"])
            + ". "
            + str(row["instruction"])
            + "<br>"

        )


    return jsonify({

        "status":"success",

        "food":recipe["recipe_name"],

        "ingredients":ingredient_text,

        "instruction":instruction_text

    })
@app.route("/about")
def about():

    return render_template(
        "about.html"
    )
# ==========================================
# Translate Recipe
# ==========================================

# ==========================================
# Translate Recipe (Clean Version)
# ==========================================

@app.route("/translate_recipe", methods=["POST"])
def translate_recipe():

    data = request.get_json()

    recipe_id = str(data["recipe_id"]).strip()

    persons = int(data.get("persons", 2))

    language = data.get("language", "english")

    config = load_dataset(language)

    # ==========================================
    # Read Dataset
    # ==========================================

    ingredients = read_csv_safe(
        config["ingredients"],
        dtype=str
    )

    steps = read_csv_safe(
        config["steps"],
        dtype=str
    )

    ingredients.columns = ingredients.columns.str.strip()

    steps.columns = steps.columns.str.strip()

    ingredients[config["recipe_col"]] = (

        ingredients[config["recipe_col"]]

        .astype(str)

        .str.strip()

    )

    steps[config["recipe_col"]] = (

        steps[config["recipe_col"]]

        .astype(str)

        .str.strip()

    )

    # ==========================================
    # Filter Recipe
    # ==========================================

    ingredients = ingredients[
        ingredients[config["recipe_col"]] == recipe_id
    ].copy()

    steps = steps[
        steps[config["recipe_col"]] == recipe_id
    ].copy()

    # ==========================================
    # Scale Ingredient Quantity
    # ==========================================

    ingredients[config["quantity_col"]] = (

        ingredients[config["quantity_col"]]

        .apply(

            lambda x:

            scale_quantity(
                x,
                persons
            )

        )

    )

    # ==========================================
    # Scale Instructions
    # ==========================================

    steps[config["step_col"]] = (

        steps[config["step_col"]]

        .apply(

            lambda x:

            scale_instruction(
                x,
                persons
            )

        )

    )

    # ==========================================
    # Sort Steps
    # ==========================================

    if config["step_no"] in steps.columns:

        try:

            steps[config["step_no"]] = pd.to_numeric(

                steps[config["step_no"]]

            )

            steps = steps.sort_values(

                config["step_no"]

            )

        except:

            pass

    # ==========================================
    # Return JSON
    # ==========================================

    return jsonify({

        "ingredients":

        ingredients[

            [

                config["ingredient_col"],

                config["quantity_col"]

            ]

        ].rename(

            columns={

                config["ingredient_col"]:

                "ingredient_name",

                config["quantity_col"]:

                "quantity"

            }

        ).to_dict("records"),

        "steps":

        steps[

            [

                config["step_no"],

                config["step_col"]

            ]

        ].rename(

            columns={

                config["step_no"]:

                "step_number",

                config["step_col"]:

                "instruction"

            }

        ).to_dict("records")

    })

# ==========================================================
# React API endpoints
# These endpoints keep the original Flask functions/data logic,
# but return JSON so the React frontend can consume them.
# ==========================================================

@app.get("/api/session")
def api_session():
    return jsonify({"authenticated": "username" in session, "username": session.get("username", "")})

@app.post("/api/login")
def api_login():
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))
    user = User.query.filter_by(email=email).first()
    if user and bcrypt.check_password_hash(user.password, password):
        session["username"] = user.name
        return jsonify({"status": "success", "username": user.name})
    return jsonify({"status": "error", "message": "Invalid Email or Password"}), 401

@app.post("/api/register")
def api_register():
    data = request.get_json(silent=True) or {}
    name = str(data.get("name", "")).strip()
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))
    if not name or not email or not password:
        return jsonify({"status": "error", "message": "All fields are required."}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({"status": "error", "message": "Email already exists."}), 409
    errors = validate_password(password)
    if errors:
        return jsonify({"status": "error", "message": "Password does not meet the requirements.", "errors": errors}), 400
    hashed_password = bcrypt.generate_password_hash(password).decode("utf-8")
    db.session.add(User(name=name, email=email, password=hashed_password))
    db.session.commit()
    return jsonify({"status": "success", "message": "Account created successfully."})

@app.post("/api/forgot-password")
def api_forgot_password():
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip().lower()
    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({"status": "error", "message": "No account found with this email address."}), 404
    token = secrets.token_urlsafe(32)
    user.reset_token = token
    user.reset_token_expiry = datetime.utcnow() + timedelta(minutes=15)
    db.session.commit()
    reset_link = url_for("reset_password", token=token, _external=True)
    if not send_reset_email(user.email, reset_link):
        return jsonify({"status": "error", "message": "Unable to send email. Please try again later."}), 500
    return jsonify({"status": "success", "message": "Password reset link has been sent to your email."})

@app.post("/api/reset-password/<token>")
def api_reset_password(token):
    user = User.query.filter_by(reset_token=token).first()
    if not user:
        return jsonify({"status": "error", "message": "Invalid password reset link."}), 400
    if user.reset_token_expiry is None or datetime.utcnow() > user.reset_token_expiry:
        return jsonify({"status": "error", "message": "Password reset link expired."}), 400
    data = request.get_json(silent=True) or {}
    password = str(data.get("password", ""))
    confirm_password = str(data.get("confirm_password", ""))
    if password != confirm_password:
        return jsonify({"status": "error", "message": "Passwords do not match."}), 400
    errors = validate_password(password)
    if errors:
        return jsonify({"status": "error", "message": "Password does not meet the requirements.", "errors": errors}), 400
    user.password = bcrypt.generate_password_hash(password).decode("utf-8")
    user.reset_token = None
    user.reset_token_expiry = None
    db.session.commit()
    return jsonify({"status": "success", "message": "Password reset successfully."})

@app.post("/api/logout")
def api_logout():
    session.pop("username", None)
    return jsonify({"status": "success"})

@app.get("/api/balti-recipe/<food>")
def api_balti_recipe(food):
    english_df = read_csv_safe("data/recipes.csv", encoding="latin1", dtype=str)
    urdu_df = read_csv_safe("data/recipes_urdu.csv", encoding="utf-8-sig", dtype=str)
    english_df.columns = english_df.columns.str.strip()
    urdu_df.columns = urdu_df.columns.str.strip()
    english_df["Food Name"] = english_df["Food Name"].astype(str).str.strip()
    urdu_df["Food Name"] = urdu_df["Food Name"].astype(str).str.strip()
    english_df["S.No"] = english_df["S.No"].astype(str).str.strip()
    urdu_df["S.No"] = urdu_df["S.No"].astype(str).str.strip()
    recipe = english_df[english_df["Food Name"].str.lower() == food.strip().lower()]
    if recipe.empty:
        return jsonify({"error": "Recipe not found"}), 404
    row = recipe.iloc[0]
    recipe_no = str(row["S.No"])
    ur = urdu_df[urdu_df["S.No"] == recipe_no]
    return jsonify({"recipe_no": recipe_no, "food": str(row["Food Name"]), "urdu_food": str(ur.iloc[0]["Food Name"]) if not ur.empty else str(row["Food Name"])})

@app.get("/api/pakistani-recipe/<food>")
def api_pakistani_recipe(food):
    persons = max(1, min(10, int(request.args.get("persons", 2))))
    language = request.args.get("language", "english")
    if language not in ("english", "urdu"):
        language = "english"
    config = load_dataset(language)
    recipes_df = read_csv_safe(config["recipes"], dtype=str)
    ingredients_df = read_csv_safe(config["ingredients"], dtype=str)
    steps_df = read_csv_safe(config["steps"], dtype=str)
    for df in (recipes_df, ingredients_df, steps_df):
        df.columns = df.columns.astype(str).str.strip()
    recipes_df[config["recipe_col"]] = recipes_df[config["recipe_col"]].astype(str).str.strip()
    # recipe_name is English in the original dataset; for Urdu, search the recipe id through the English dataset.
    if language == "english":
        match = recipes_df[recipes_df[config["recipe_col"]].astype(str).str.strip() == ""]
    # Use the English dataset to resolve the food name to recipe_id, then load the requested language tables.
    en = read_csv_safe("data/Pakistani_Food/pakistani_recipes.csv", dtype=str)
    en.columns = en.columns.astype(str).str.strip()
    match = en[en["recipe_name"].astype(str).str.strip().str.lower() == food.strip().lower()]
    if match.empty:
        match = en[en["recipe_name"].astype(str).str.lower().str.contains(food.strip().lower(), regex=False)]
    if match.empty:
        return jsonify({"error": "Recipe not found"}), 404
    recipe_id = str(match.iloc[0]["recipe_id"]).strip()
    ingredients = ingredients_df[ingredients_df[config["recipe_col"]].astype(str).str.strip() == recipe_id].copy()
    steps = steps_df[steps_df[config["recipe_col"]].astype(str).str.strip() == recipe_id].copy()
    ingredients[config["quantity_col"]] = ingredients[config["quantity_col"]].apply(lambda x: scale_quantity(x, persons))
    steps[config["step_col"]] = steps[config["step_col"]].apply(lambda x: scale_instruction(x, persons))
    if config["step_no"] in steps.columns:
        try:
            steps[config["step_no"]] = pd.to_numeric(steps[config["step_no"]])
            steps = steps.sort_values(config["step_no"])
        except Exception:
            pass
    return jsonify({
        "food_name": str(match.iloc[0]["recipe_name"]),
        "recipe_id": recipe_id,
        "persons": persons,
        "language": language,
        "ingredients": ingredients[[config["ingredient_col"], config["quantity_col"]]].rename(columns={config["ingredient_col"]:"ingredient_name",config["quantity_col"]:"quantity"}).to_dict("records"),
        "steps": steps[[config["step_no"], config["step_col"]]].rename(columns={config["step_no"]:"step_number",config["step_col"]:"instruction"}).to_dict("records")
    })

# Run App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
