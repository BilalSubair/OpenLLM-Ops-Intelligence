import unittest
import asyncio
from openai_api import list_models, process_chat_intent, stream_chat_completion, ChatMessage, ChatCompletionRequest


class TestOpenAIAdapterSuite(unittest.TestCase):
    def test_list_models_format(self):
        result = list_models()
        self.assertEqual(result["object"], "list")
        self.assertIn("data", result)
        self.assertGreater(len(result["data"]), 0)
        
        # Verify model object shape
        first_model = result["data"][0]
        self.assertIn("id", first_model)
        self.assertEqual(first_model["object"], "model")
        self.assertEqual(first_model["owned_by"], "openllm-ops-intelligence")

    def test_troubleshooting_intent_detection(self):
        messages = [
            ChatMessage(role="user", content="Why am I getting a CUDA out of memory error on GPU 0?")
        ]
        response = asyncio.run(process_chat_intent(
            model="qwen3.5:2b",
            messages=messages,
            max_tokens=256
        ))
        self.assertIn("Diagnostic Report", response)
        self.assertIn("Root Cause Analysis", response)
        self.assertIn("Remediation", response)

    def test_general_query_intent(self):
        messages = [
            ChatMessage(role="user", content="Hello, tell me what this platform does.")
        ]
        response = asyncio.run(process_chat_intent(
            model="qwen3.5:2b",
            messages=messages,
            max_tokens=128
        ))
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_streaming_sse_generator(self):
        async def run_stream():
            chunks = []
            async for chunk in stream_chat_completion(
                completion_id="chatcmpl-test1234",
                model="qwen3.5:2b",
                created_time=1700000000,
                content="CUDA memory optimized."
            ):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(run_stream())
        self.assertGreater(len(chunks), 1)
        self.assertTrue(chunks[-1].strip().endswith("[DONE]"))
        self.assertIn("chat.completion.chunk", chunks[0])


if __name__ == "__main__":
    unittest.main()
