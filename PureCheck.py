import os
import base64
import json
from mimetypes import guess_type
import re
import matplotlib.pyplot as plt
import cv2
from dotenv import load_dotenv
from openai import AzureOpenAI

# ==============================
# Utility Functions
# ==============================

def local_image_to_data_url(image_path):
    """
    Encode a local image into a Base64 data URL.

    :param image_path: Path to the local image file.
    :return: A data URL containing the encoded image.
    """
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"  # Default MIME type

    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")

    return f"data:{mime_type};base64,{base64_encoded_data}"

# ==============================
# Azure OpenAI API Functions
# ==============================

class AzureOpenAIClient:
    """
    A client wrapper for Azure OpenAI API.
    """

    def __init__(self, api_endpoint, api_key, deployment_name, api_version):
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            base_url=api_endpoint,
        )
        self.deployment_name = deployment_name

    def analyze_image(
        self,
        image_data_url,
        system_message,
        user_message,
        clean_image_url_example,
        dirty_image_url_example,
        max_tokens=2000,
    ):
        """
        Analyze an image using Azure OpenAI API.
        """
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "This is a clean image example:",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": clean_image_url_example},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "This is a dirty image example:",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": dirty_image_url_example},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_message},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                },
            ],
            max_tokens=max_tokens,
        )
        return response

def extract_code(content: str) -> str:
    """
    Extract the first code block and ensure proper JSON formatting.
    """
    pattern = r"```(?:[A-Za-z0-9]*)\s*\n(.*?)```"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        # Get the content of the code block
        code = match.group(1).strip()
    else:
        code = content.strip()

    # Replace single quotes with double quotes
    code = code.replace("'", '"')
    return code

# ==============================
# Main Workflow
# ==============================

def main():
    # Load environment variables
    load_dotenv()
    # Configuration
    api_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment_name = "gpt-4o"
    api_version = os.getenv("AZURE_API_VERSION_GPT4")
    image_path = "dirty_images/img1.JPEG"

    # Initialize Azure OpenAI client
    client = AzureOpenAIClient(
        api_endpoint=api_endpoint,
        api_key=api_key,
        deployment_name=deployment_name,
        api_version=api_version,
    )

    # Encode the image to a Base64 data URL
    data_url = local_image_to_data_url(image_path)
    clean_image_url_example = local_image_to_data_url("images/img1.JPEG")
    dirty_image_url_example = local_image_to_data_url("images/img13.JPEG")
    # Define messages
    system_message = "You are an assistant specialized in analyzing cleanliness in industrial engineering images. Use the provided examples to determine whether a scene is clean or not. Provide reasons if the image is not clean."
    user_message = """
        Analyze the provided image for cleanliness in an industrial engineering setting. 
        Determine whether the scene is clean or not based on visible stains, dirt, debris, rust, or other contamination. 
        If the scene is not clean, provide reasons describing the observed issues in detail. 
        - Clean Image Example: The image is free of visible stains, dirt, debris, rust, or contamination. The surface appears uniform and smooth.
        - Dirty Image Example: The image contains visible stains, debris, rust, or contamination. These might appear as discoloration, spots, or accumulations.

        Return the results in the following JSON format:
        {
            'is_clean': true/false,
            'description': '<reason why the scene is not clean>'
        }
        Ensure to mark 'is_clean' as true only if the image is entirely free of dirt, stains, or contamination. 
        Provide clear reasons in 'issues_detected' when 'is_clean' is false, avoiding technical jargon.
    """

    # Call the API to analyze the image
    response = client.analyze_image(
        data_url,
        system_message,
        user_message,
        clean_image_url_example,
        dirty_image_url_example,
    )

    # Extract and visualize results
    if hasattr(response, "choices") and len(response.choices) > 0:
        try:
            content = extract_code(response.choices[0].message.content)
            analysis_result = json.loads(content)
            print(
                f"API Response: {image_path}",
                json.dumps(analysis_result, indent=4, ensure_ascii=False),
            )
        except json.JSONDecodeError as e:
            print("JSON parsing error:", str(e))
            print("Original content:", content)

if __name__ == "__main__":
    main()
