from googleapiclient.discovery import build
import os
import re

# Ensure API Key is set as an environment variable or managed securely
API_KEY = os.environ.get("GOOGLE_API_KEY") # Your Google API Key with Fact Check API enabled
FACT_CHECK_SERVICE = build("factchecktools", "v1alpha1", developerKey=API_KEY)

def search_fact_checks(query_text):
    try:
        request = FACT_CHECK_SERVICE.claims().search(query=query_text)
        response = request.execute()
        return response.get("claims", [])
    except Exception as e:
        print(f"Error querying Fact Check API: {e}")
        return []

# Example usage:
user_submitted_claim = "The moon is made of green cheese."
fact_checks = search_fact_checks(user_submitted_claim)

if fact_checks:
    print(f"Found {len(fact_checks)} fact-checks for '{user_submitted_claim}':")
    for claim_review in fact_checks:
        print(f"  Claim: {claim_review.get('text')}")
        if claim_review.get('claimReview'):
            publisher = claim_review['claimReview'][0].get('publisher', {}).get('name', 'N/A')
            rating = claim_review['claimReview'][0].get('textualRating', 'N/A')
            review_url = claim_review['claimReview'][0].get('url', '#')
            print(f"    Publisher: {publisher}")
            print(f"    Rating: {rating}")
            print(f"    Review URL: {review_url}")
else:
    print(f"No fact-checks found for '{user_submitted_claim}'.")