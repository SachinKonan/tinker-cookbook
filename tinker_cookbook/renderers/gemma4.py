"""Renderer for the Gemma 4 canonical chat format."""

import tinker

from tinker_cookbook.renderers.base import (
    Message,
    ParseTermination,
    RenderContext,
    RenderedMessage,
    Renderer,
    ensure_text,
    parse_response_for_stop_token,
)


class Gemma4Renderer(Renderer):
    """Renderer for Gemma 4 instruct models (non-thinking mode).

    Format (matches google/gemma-4-*-it ``chat_template.jinja`` with
    ``enable_thinking=false``)::

        <bos><|turn>user
        What is 2+2?<turn|>
        <|turn>model
        <|channel>thought
        <channel|>4<turn|>

    ``<|turn>``, ``<turn|>``, ``<|channel>`` and ``<channel|>`` are single
    special tokens. The assistant header includes an *empty* thought channel
    (``<|channel>thought\\n<channel|>``) — that is exactly what the stock
    template appends for ``add_generation_prompt`` in non-thinking mode, so
    sampled completions start right where the model expects to answer.
    Unlike the stock template (which strips the channel block from history
    turns), we keep it in re-rendered history so that rendering is
    self-consistent between sampling and training.

    Tool calling is not supported.
    """

    @property
    def has_extension_property(self) -> bool:
        """Rendering a longer conversation extends the rendered prefix."""
        return True

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        """Render one chat message into Gemma 4 header and output chunks.

        Args:
            message (Message): The chat message to render.
            ctx (RenderContext): Positional context for the message.

        Returns:
            RenderedMessage: Header and output token chunks for the message.
        """
        role = "model" if message["role"] == "assistant" else message["role"]
        header_str = f"<|turn>{role}\n"
        if role == "model":
            # Empty thought channel: canonical non-thinking generation prompt.
            header_str += "<|channel>thought\n<channel|>"
        output_str = ensure_text(message["content"]) + "<turn|>\n"

        header = tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(header_str, add_special_tokens=False)
        )
        output: list[tinker.ModelInputChunk] = [
            tinker.types.EncodedTextChunk(
                tokens=self.tokenizer.encode(output_str, add_special_tokens=False)
            )
        ]
        return RenderedMessage(header=header, output=output)

    @property
    def _bos_tokens(self) -> list[int]:
        return self.tokenizer.encode("<bos>", add_special_tokens=False)

    @property
    def _end_message_token(self) -> int:
        (token,) = self.tokenizer.encode("<turn|>", add_special_tokens=False)
        return token

    def get_stop_sequences(self) -> list[int]:
        """Return stop sequences for Gemma 4 generation.

        Returns:
            list[int]: Single-element list containing the ``<turn|>`` token ID.
        """
        return [self._end_message_token]

    def parse_response(self, response: list[int]) -> tuple[Message, ParseTermination]:
        """Parse sampled token IDs back into an assistant Message.

        Strips the ``<turn|>`` stop token if present and decodes the remaining
        tokens into text content.

        Args:
            response (list[int]): Raw token IDs from the sampler.

        Returns:
            tuple[Message, ParseTermination]: ``STOP_SEQUENCE`` if the
                ``<turn|>`` stop token was found, ``MALFORMED`` otherwise.
        """
        return parse_response_for_stop_token(response, self.tokenizer, self._end_message_token)
