{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/bellamelati/FinalProjectBangkit/blob/main/Preprocessing.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **Preprocessing**"
      ],
      "metadata": {
        "id": "G2m5C1LbVnNi"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "##**Importing Libraries and Loading Data**\n",
        "\n"
      ],
      "metadata": {
        "id": "Hne7LJvNJaWg"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "pip install pandas"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "507jI1pIVk4u",
        "outputId": "e925dac6-4a51-4630-d69e-641205d0a644"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: pandas in /usr/local/lib/python3.10/dist-packages (2.0.3)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas) (2.8.2)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas) (2023.4)\n",
            "Requirement already satisfied: tzdata>=2022.1 in /usr/local/lib/python3.10/dist-packages (from pandas) (2024.1)\n",
            "Requirement already satisfied: numpy>=1.21.0 in /usr/local/lib/python3.10/dist-packages (from pandas) (1.25.2)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.2->pandas) (1.16.0)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pip install --upgrade pip"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "SL6yriJnH3-8",
        "outputId": "f4e4dc2b-04ac-489d-9708-cf6d64555390"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: pip in /usr/local/lib/python3.10/dist-packages (23.1.2)\n",
            "Collecting pip\n",
            "  Downloading pip-24.0-py3-none-any.whl (2.1 MB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m2.1/2.1 MB\u001b[0m \u001b[31m23.6 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hInstalling collected packages: pip\n",
            "  Attempting uninstall: pip\n",
            "    Found existing installation: pip 23.1.2\n",
            "    Uninstalling pip-23.1.2:\n",
            "      Successfully uninstalled pip-23.1.2\n",
            "Successfully installed pip-24.0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "J2WCjcPlVgxJ"
      },
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
        "from sklearn.decomposition import PCA\n",
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "# Load the data\n",
        "data = pd.read_csv('kaggle.csv')"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import zipfile\n",
        "import pandas as pd\n",
        "import requests\n",
        "\n",
        "# Download the ZIP file\n",
        "url = \"https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_csv_2024-04-18.zip\"\n",
        "zip_file = requests.get(url)\n",
        "with open(\"FoodData_Central_csv_2024-04-18.zip\", \"wb\") as f:\n",
        "    f.write(zip_file.content)\n",
        "\n",
        "# Extract the ZIP file\n",
        "with zipfile.ZipFile(\"FoodData_Central_csv_2024-04-18.zip\", \"r\") as zip_ref:\n",
        "    zip_ref.extractall()"
      ],
      "metadata": {
        "id": "E9aoXydOalxO"
      },
      "execution_count": 1,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "data_food_update_log_entry = pd.read_csv(\"FoodData_Central_csv_2024-04-18/food_update_log_entry.csv\")\n",
        "data_sr_legacy_food = pd.read_csv(\"FoodData_Central_csv_2024-04-18/sr_legacy_food.csv\")\n",
        "data_agricultural_samples = pd.read_csv(\"FoodData_Central_csv_2024-04-18/agricultural_samples.csv\")\n",
        "data_food_nutrient_source = pd.read_csv(\"FoodData_Central_csv_2024-04-18/food_nutrient_source.csv\")\n",
        "data_fndds_ingredient_nutrient_value = pd.read_csv(\"FoodData_Central_csv_2024-04-18/fndds_ingredient_nutrient_value.csv\")\n",
        "data_input_food = pd.read_csv(\"FoodData_Central_csv_2024-04-18/input_food.csv\")\n",
        "data_food_attribute_type = pd.read_csv(\"FoodData_Central_csv_2024-04-18/food_attribute_type.csv\")\n",
        "data_lab_method_code = pd.read_csv(\"FoodData_Central_csv_2024-04-18/lab_method_code.csv\")\n",
        "data_measure_unit = pd.read_csv(\"FoodData_Central_csv_2024-04-18/measure_unit.csv\")\n",
        "data_food_nutrient_conversion_factor = pd.read_csv(\"FoodData_Central_csv_2024-04-18/food_nutrient_conversion_factor.csv\")\n",
        "data_nutrient = pd.read_csv(\"FoodData_Central_csv_2024-04-18/nutrient.csv\")\n",
        "data_food_nutrient_derivation = pd.read_csv(\"FoodData_Central_csv_2024-04-18/food_nutrient_derivation.csv\")\n",
        "data_sub_sample_food = pd.read_csv(\"FoodData_Central_csv_2024-04-18/sub_sample_food.csv\")\n",
        "data_market_acquisition = pd.read_csv(\"FoodData_Central_csv_2024-04-18/market_acquisition.csv\")\n",
        "data_wweia_food_category = pd.read_csv(\"FoodData_Central_csv_2024-04-18/wweia_food_category.csv\")\n",
        "data_food_attribute = pd.read_csv(\"FoodData_Central_csv_2024-04-18/food_attribute.csv\")\n",
        "data_sample_food = pd.read_csv(\"FoodData_Central_csv_2024-04-18/sample_food.csv\")\n",
        "data_food = pd.read_csv(\"FoodData_Central_csv_2024-04-18/food.csv\")\n",
        "data_food_protein_conversion_factor = pd.read_csv(\"FoodData_Central_csv_2024-04-18/food_protein_conversion_factor.csv\")\n",
        "data_lab_method = pd.read_csv(\"FoodData_Central_csv_2024-04-18/lab_method.csv\")\n",
        "data_sub_sample_result = pd.read_csv(\"FoodData_Central_csv_2024-04-18/sub_sample_result.csv\")\n",
        "data_microbe = pd.read_csv(\"FoodData_Central_csv_2024-04-18/microbe.csv\")\n",
        "data_nutrient_incoming_name = pd.read_csv(\"FoodData_Central_csv_2024-04-18/nutrient_incoming_name.csv\")\n",
        "data_food_category = pd.read_csv(\"FoodData_Central_csv_2024-04-18/food_category.csv\")\n",
        "data_survey_fndds_food = pd.read_csv(\"FoodData_Central_csv_2024-04-18/survey_fndds_food.csv\")\n",
        "data_lab_method_nutrient = pd.read_csv(\"FoodData_Central_csv_2024-04-18/lab_method_nutrient.csv\")\n",
        "data_food_nutrient = pd.read_csv(\"FoodData_Central_csv_2024-04-18/food_nutrient.csv\")\n",
        "data_branded_food = pd.read_csv(\"FoodData_Central_csv_2024-04-18/branded_food.csv\")\n",
        "data_retention_factor = pd.read_csv(\"FoodData_Central_csv_2024-04-18/retention_factor.csv\")\n",
        "data_acquisition_samples = pd.read_csv(\"FoodData_Central_csv_2024-04-18/acquisition_samples.csv\")\n",
        "data_food_calorie_conversion_factor = pd.read_csv(\"FoodData_Central_csv_2024-04-18/food_calorie_conversion_factor.csv\")\n",
        "data_food_component = pd.read_csv(\"FoodData_Central_csv_2024-04-18/food_component.csv\")\n",
        "data_fndds_derivation = pd.read_csv(\"FoodData_Central_csv_2024-04-18/fndds_derivation.csv\")\n",
        "data_foundation_food = pd.read_csv(\"FoodData_Central_csv_2024-04-18/foundation_food.csv\")\n",
        "data_food_portion = pd.read_csv(\"FoodData_Central_csv_2024-04-18/food_portion.csv\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Xg2F4f8yatdL",
        "outputId": "36895354-a71a-48f6-e053-5581179e78ee"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "<ipython-input-2-1aa96d2c14ed>:16: DtypeWarning: Columns (5) have mixed types. Specify dtype option on import or set low_memory=False.\n",
            "  data_food_attribute = pd.read_csv(\"FoodData_Central_csv_2024-04-18/food_attribute.csv\")\n",
            "<ipython-input-2-1aa96d2c14ed>:27: DtypeWarning: Columns (10) have mixed types. Specify dtype option on import or set low_memory=False.\n",
            "  data_food_nutrient = pd.read_csv(\"FoodData_Central_csv_2024-04-18/food_nutrient.csv\")\n",
            "<ipython-input-2-1aa96d2c14ed>:28: DtypeWarning: Columns (2,3,4,6,9,12,16,17,18,19) have mixed types. Specify dtype option on import or set low_memory=False.\n",
            "  data_branded_food = pd.read_csv(\"FoodData_Central_csv_2024-04-18/branded_food.csv\")\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "### **Cleaning Data**"
      ],
      "metadata": {
        "id": "D38G7AA6W109"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Drop 'Unnamed: 0' if it's an index column with no use\n",
        "data.drop('Unnamed: 0', axis=1, inplace=True)\n",
        "\n",
        "# Convert columns with numeric data stored as strings, strip units like 'g' and 'mg'\n",
        "for col in data.columns:\n",
        "    if data[col].dtype == 'object':\n",
        "        data[col] = data[col].str.extract('(\\d+\\.?\\d*)').astype(float)\n",
        "\n",
        "# Fill missing values - fill with the mean of the column\n",
        "data.fillna(data.mean(), inplace=True)\n"
      ],
      "metadata": {
        "id": "JYEIakk1Wr4m"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "##**Feature Engineering**"
      ],
      "metadata": {
        "id": "lRV6L5cdWixf"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Ensure 'total_fat' is numeric. Convert if necessary.\n",
        "if data['total_fat'].dtype == object:\n",
        "    # Assuming 'total_fat' contains values like '5 g' or '5mg', we extract the numbers\n",
        "    data['total_fat'] = pd.to_numeric(data['total_fat'].str.extract('(\\d+\\.?\\d*)')[0])\n",
        "\n",
        "# Fill any missing values that might have been created\n",
        "data['total_fat'].fillna(data['total_fat'].mean(), inplace=True)\n",
        "\n",
        "# Now calculate the Calories to Fat ratio, safely adding 1 to avoid division by zero issues\n",
        "data['Calories_to_Fat_Ratio'] = data['calories'] / (data['total_fat'] + 1)\n"
      ],
      "metadata": {
        "id": "E8iEJvkAXOjC"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "##**Standardization**"
      ],
      "metadata": {
        "id": "t0j4bhO8W7CD"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.preprocessing import StandardScaler\n",
        "scaler = StandardScaler()\n",
        "\n",
        "# Select numerical columns (excluding those converted above which might need different treatment)\n",
        "numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns\n",
        "data[numeric_cols] = scaler.fit_transform(data[numeric_cols])"
      ],
      "metadata": {
        "id": "aGOWhXH0TdN1"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "##**One Hot Encoding**"
      ],
      "metadata": {
        "id": "wr4EiTJoW97m"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "data = pd.get_dummies(data, drop_first=True)"
      ],
      "metadata": {
        "id": "ObuFsIJaTe2Q"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "##**Dimensionality Reduction**"
      ],
      "metadata": {
        "id": "TdvUQvDZXDB5"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.decomposition import PCA\n",
        "pca = PCA(n_components=0.95)\n",
        "principal_components = pca.fit_transform(data[numeric_cols])"
      ],
      "metadata": {
        "id": "gnTy0lP3TjT6"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "##**Saving Process Data**"
      ],
      "metadata": {
        "id": "zd0UUWSMXH3J"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "data.to_csv('kaggle_data.csv', index=False)"
      ],
      "metadata": {
        "id": "yxGfQZo4TkxB"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}