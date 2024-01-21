"""
chat_history.py

This module contains the chat history data structured as a list of dictionaries.
Each dictionary in the list represents a single message within the chat history,
detailing the role of the sender ('user' or 'assistant') and the content of the message.

Attributes:
    chat_message_history (List[Dict[str, str]]): A list where each element is a dictionary
                                                 containing the following key-value pairs:
                                                 'role' (str): The role of the message sender,
                                                               either 'user' or 'assistant'.
                                                 'content' (str): The actual text content of
                                                                  the message.

Example Usage:
    from chat_history import chat_message_history

    for message in chat_message_history:
        print(message['role'], message['content'])
"""


chat_message_history = [
    {"role": "assistant",
     "content": "Hey Ofer, it's Yuval from Libra (: Your insurance is ending at the end of the month. The price for the comprehensive insurance is 2,702 ILS and the mandatory insurance is 1,135 ILS. Renew?"},
    {"role": "user", "content": "not text message"}, {"role": "assistant",
                                                      "content": "Hey, I didn't receive your message because unfortunately, we do not receive photos, stickers, and audio files. Could you write it?"},
    {"role": "user",
     "content": "Hey Yuval, the offer is more expensive than what I paid.. I will check with other companies. Thank you very much \ud83d\ude4f"},
    {"role": "assistant",
     "content": "No problem, Ofer, don't finalize anywhere without talking to me. I will do everything to keep you with us."},
    {"role": "user", "content": "Beatrice, Ofer's wife... And I insure my cars and his through them..."},
    {"role": "assistant",
     "content": "What's happening, Beatrice? Any news with the insurance? How can we proceed and finalize this?"},
    {"role": "assistant",
     "content": "Hey, how are you? Did you get a chance to check other offers? I would be happy to know the price you received for comprehensive and mandatory insurance."},
    {"role": "assistant",
     "content": "What's happening, Ofer? Any news with the insurance? How can we proceed and finalize this?"},
    {"role": "user",
     "content": "Hey, I am currently with another company that offered me the price I paid you (Libra) for the year 2023. Mandatory: 995 ILS. Comprehensive: 1813 ILS. In addition, they are trying to reduce it further."},
    {"role": "assistant", "content": "I am checking what is the most we can do"}, {"role": "user",
                                                                                   "content": "The offer you gave me at the beginning of the WhatsApp conversation doesn't make sense. Year by year the car's value decreases, so there's no reason to pay almost double the insurance from the current year."},
    {"role": "user", "content": "Thank you very much"}, {"role": "assistant", "content": "Hey there!"},
    {"role": "assistant",
     "content": "I checked with the manager, I will be able to get to 2,000 ILS for comprehensive with the same coverages as this year - canceling the deductible at arrangement garages, coverage for hit and run accidents, a replacement car... I'm not familiar with the coverage you have but it's important to pay attention."},
    {"role": "assistant", "content": "The total will be 3,135 ILS. What do you think?"}, {"role": "user",
                                                                                          "content": "I paid 2,808 ILS including all the coverages you mentioned in the previous message. There is no reason I should pay an additional 300 plus ILS. Especially since I'll explain it again, a car decreases in value from year to year, so there's no reason to pay more for insurance! I am insured with another car with you, waiting for an offer from the additional company also for my car in addition to my husband's, and simply if you can't meet that, I will transfer the 2 cars to the other company. Thank you very much."},
    {"role": "assistant", "content": "Can I ask where you got the offer?"}, {"role": "user",
                                                                             "content": "At Yashir Bituach, before I switched to you, I was insured for many years with Yashir on both cars."},
    {"role": "assistant",
     "content": "There is no cancellation of the deductible and more... I assume their mandatory insurance is higher than ours? And the most I can do is 1,950 ILS for the comprehensive. Let me know if we can proceed?"},
    {"role": "user", "content": "Why can't you match the amount for me approximately to the sum of the year?"},
    {"role": "assistant", "content": "I really did my very very best - do you believe me?"},
    {"role": "user", "content": "Please try a bit more."}, {"role": "assistant",
                                                            "content": "Listen, I really tried more. This is a really, really good price for this coverage."},
    {"role": "user",
     "content": "What about the Librot (loyalty points) I've accumulated, won't they reduce the price further?"},
    {"role": "assistant", "content": "No, it's deep within the calculation already (:"},
    {"role": "assistant", "content": "I really did everything and we kept the same coverages. What do you say?"},
    {"role": "user", "content": "Please send me the comprehensive and the mandatory, how much is the total together?"},
    {"role": "assistant",
     "content": "Comprehensive 1,950 ILS and mandatory 1,135 ILS. Total 3,085 ILS for renewal. The exact same coverages as last year."},
    {"role": "assistant", "content": "What do you think?"},
    {"role": "user", "content": "I will ask my husband and update you. Thank you very much."},
    {"role": "assistant", "content": "Cool"}, {"role": "assistant",
                                               "content": "Good morning Beatrice, how are you? Did you get a chance to talk to your husband? I want to manage this before the end of the month."},
    {"role": "assistant",
     "content": "What's happening, Ofer? Any news with the insurance? How can we proceed and finalize this?"},
    {"role": "user", "content": "Good morning, can you please call me?"}, {"role": "assistant",
                                                                           "content": "Good morning, Ofer. Can I ask that they call you as soon as possible? In the meantime, I am available here."},
    {"role": "user", "content": "Yes, I would be happy if they could get back to me as soon as possible."},
    {"role": "user", "content": "Thank you very much"}
]
