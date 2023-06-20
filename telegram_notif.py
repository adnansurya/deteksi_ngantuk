import telebot
import auth

# Telegram bot API token
TOKEN = auth.tele_token

# Initialize Telegram bot
bot = telebot.TeleBot(TOKEN)

# Handler for the '/start' command
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Welcome to the Bot!")

# Handler for the '/send' command
@bot.message_handler(commands=['send'])
def send_image(message):
    # Load and send the image
    photo = open("image.jpg", 'rb')
    bot.send_photo(message.chat.id, photo)

    # Send a text message along with the image
    bot.send_message(message.chat.id, "Hello from the Bot!")

# Start the bot
bot.polling()
