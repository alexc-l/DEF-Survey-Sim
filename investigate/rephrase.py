from sys import exit as sys_exit
from claude2_api.client import (
    ClaudeAPIClient,
    SendMessageResponse, SOCKSProxy,
)
from claude2_api.session import SessionData, get_session_data
from claude2_api.errors import ClaudeAPIError, MessageRateLimitError, OverloadError

# Wildcard import will also work the same as above
# from claude2_api import *

# # List of attachments filepaths, up to 5, max 10 MB each
# FILEPATH_LIST = [
#     "test1.txt",
#     "test2.txt",
# ]
socks_proxy = SOCKSProxy(
    "127.0.0.1",    # Proxy IP
    7890,                   # Proxy port
    version_num=4           # Either 4 or 5, defaults to 4
)
# This function will automatically retrieve a SessionData instance using selenium
# It will auto gather cookie session, user agent and organization ID.
# Omitting profile argument will use default Firefox profile
session: SessionData = get_session_data()

# Initialize a client instance using a session
# Optionally change the requests timeout parameter to best fit your needs...default to 240 seconds.
client = ClaudeAPIClient(session, timeout=240)

# Create a new chat and cache the chat_id
chat_id = "c8f75643-de3e-4e3f-a2e0-f26802d55a54"
if not chat_id:
    # This will not throw MessageRateLimitError
    # But it still means that account has no more messages left.
    print("\nMessage limit hit, cannot create chat...")
    sys_exit(1)

try:
    # Used for sending message with or without attachments
    # Returns a SendMessageResponse instance
    res: SendMessageResponse = client.send_message(
        chat_id, "Hello!",
    )
    # Inspect answer
    if res.answer:
        print(res.answer)
    else:
        # Inspect response status code and raw answer bytes
        print(f"\nError code {res.status_code}, raw_answer: {res.raw_answer}")
except ClaudeAPIError as e:
    # Identify the error
    if isinstance(e, MessageRateLimitError):
        # The exception will hold these informations about the rate limit:
        print(f"\nMessage limit hit, resets at {e.reset_date}")
        print(f"\n{e.sleep_sec} seconds left until -> {e.reset_timestamp}")
    elif isinstance(e, OverloadError):
        print(f"\nOverloaded error: {e}")
    else:
        print(f"\nGot unknown Claude error: {e}")
# finally:
#     # Perform chat deletion for cleanup
#     client.delete_chat(chat_id)

# # Get a list of all chats ids
# all_chat_ids = client.get_all_chat_ids()
# # Delete all chats
# for chat in all_chat_ids:
#     client.delete_chat(chat)
#
# # Or by using a shortcut utility
# client.delete_all_chats()
# sys_exit(0)