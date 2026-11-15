def explain_gemini_error(error, model_name):
    code = getattr(error, "code", None)
    detail = getattr(error, "message", None) or str(error)

    if code == 404:
        cause = (
            f"Model '{model_name}' is not available to the Google project used by "
            "this API key, or does not support this API method. Check model access "
            "with the in-app diagnostic and confirm the key's project in Google AI Studio."
        )
    elif code in (401, 403):
        cause = (
            "Google rejected this API key or project. Check that the key is active, "
            "the Gemini API is enabled, and any API-key restrictions allow it."
        )
    elif code == 429:
        cause = (
            "The Google project has reached a request, token, or spending quota. "
            "A new API key in the same project does not reset project quotas."
        )
    elif code == 400:
        cause = (
            "Google rejected the request. Check the model's supported inputs and "
            "the error details below."
        )
    elif code is not None and code >= 500:
        cause = "Google's service returned a server error. Try again later."
    else:
        cause = (
            "The request failed before a Gemini response was returned. Check the "
            "network connection and the error details below."
        )

    return f"{cause}\n\nGoogle API details ({code or 'network/client error'}): {detail}"
