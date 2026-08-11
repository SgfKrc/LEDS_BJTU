import pytest
from pydantic import ValidationError

from src.inference_service.protocol import ChatRequest
from src.multimodal import (
    build_openai_user_content,
    normalize_openai_message_content,
    validate_image_data_urls,
)


PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def test_valid_image_builds_openai_structured_content():
    assert validate_image_data_urls([PNG_DATA_URL]) == [PNG_DATA_URL]
    assert build_openai_user_content("看图", [PNG_DATA_URL]) == [
        {"type": "text", "text": "看图"},
        {"type": "image_url", "image_url": {"url": PNG_DATA_URL}},
    ]


@pytest.mark.parametrize(
    "value",
    [
        "https://example.test/image.png",
        "data:image/svg+xml;base64,PHN2Zz48L3N2Zz4=",
        "data:image/png;base64,dGV4dA==",
        "data:image/png;base64,not-base64",
    ],
)
def test_image_validation_rejects_remote_unsupported_or_spoofed_data(value):
    with pytest.raises(ValueError):
        validate_image_data_urls([value])


def test_chat_images_require_explicit_external_route():
    with pytest.raises(ValidationError):
        ChatRequest(message="看图", image_data_urls=[PNG_DATA_URL])
    with pytest.raises(ValidationError):
        ChatRequest(
            message="看图",
            image_data_urls=[PNG_DATA_URL],
            allow_external=True,
            prefer_external=True,
            routing_preference="local_only",
        )
    with pytest.raises(ValidationError):
        ChatRequest(
            message="看图",
            image_data_urls=[PNG_DATA_URL],
            allow_external=True,
            prefer_external=True,
            execution_mode="task_graph",
        )


def test_chat_images_accept_explicit_external_route():
    request = ChatRequest(
        message="看图",
        image_data_urls=[PNG_DATA_URL],
        allow_external=True,
        prefer_external=True,
    )
    assert request.image_data_urls == [PNG_DATA_URL]


def test_structured_content_rejects_unvalidated_remote_image():
    with pytest.raises(ValueError):
        normalize_openai_message_content([
            {
                "type": "image_url",
                "image_url": {"url": "https://example.test/private.png"},
            }
        ])
