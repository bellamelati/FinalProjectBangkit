{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "include_colab_link": true
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
        "id": "view-in-github",
        "colab_type": "text"
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
        "collapsed": true,
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
        "collapsed": true,
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