#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

#include <tgbot/tgbot.h>
using namespace std;
using namespace TgBot;

#include <stdio.h>
#include <tgbot/tgbot.h>

#include <stdio.h>
#include <tgbot/tgbot.h>
#include <messenger.h>
#include <qthread.h>


//ToDo убрать баг с двумя back'ами



class BotKeyboard {
public:
    BotKeyboard() {
        WelcomeMenuIni();
        StopIni();
        IntervalIni();
    };

    InlineKeyboardMarkup::Ptr GetWelcomeMenuKeyboard(){
        return WelcomeMenu;
    }

    InlineKeyboardMarkup::Ptr GetStopKeyboard(){
        return Stop;
    }

    InlineKeyboardMarkup::Ptr GetIntervalKeyboard(){
        return Interval;
    }

    InlineKeyboardMarkup::Ptr GetSetupKeyboard(){
        return SetupIntrfaceMenu;
    }

    InlineKeyboardMarkup::Ptr GetLoginKeyboard(){
        return LoginMenu;
    }







private:
        // Методы инициализации клавиатуры
        void WelcomeMenuIni() {
            InlineKeyboardMarkup::Ptr keyboard(new InlineKeyboardMarkup);

            vector<InlineKeyboardButton::Ptr> menu;
            InlineKeyboardButton::Ptr LoginButton(new InlineKeyboardButton);
            LoginButton->text = "Login";
            LoginButton->callbackData = "Login";

            InlineKeyboardButton::Ptr SignUpButton(new InlineKeyboardButton);
            SignUpButton->text = "Sign Up";
            SignUpButton->callbackData = "Sign Up";

            menu.push_back(LoginButton);
            menu.push_back(SignUpButton);
            keyboard->inlineKeyboard.push_back(menu);
            WelcomeMenu = keyboard;
        }

        void StopIni() {
            InlineKeyboardMarkup::Ptr keyboard(new InlineKeyboardMarkup);
            vector<InlineKeyboardButton::Ptr> menu;
            InlineKeyboardButton::Ptr StopButton(new InlineKeyboardButton);
            StopButton->text = "Stop";
            StopButton->callbackData = "Stop";

            InlineKeyboardButton::Ptr BackButton(new InlineKeyboardButton);
            BackButton->text = "Back";
            BackButton->callbackData = "Back";

            InlineKeyboardButton::Ptr ExitButton(new InlineKeyboardButton);
            ExitButton->text = "Exit";
            ExitButton->callbackData = "Exit";

            menu.push_back(BackButton);
            menu.push_back(ExitButton);
            menu.push_back(StopButton);
            keyboard->inlineKeyboard.push_back(menu);
            Stop = keyboard;
        }

        void IntervalIni() {
            InlineKeyboardMarkup::Ptr keyboard(new InlineKeyboardMarkup);

            vector<InlineKeyboardButton::Ptr> menu;
            InlineKeyboardButton::Ptr BackButton(new InlineKeyboardButton);
            BackButton->text = "Back";
            BackButton->callbackData = "Back";

            InlineKeyboardButton::Ptr ExitButton(new InlineKeyboardButton);
            ExitButton->text = "Exit";
            ExitButton->callbackData = "Exit";

            menu.push_back(BackButton);
            menu.push_back(ExitButton);
            keyboard->inlineKeyboard.push_back(menu);
            Interval = keyboard;
        }

        void SetupIni() {
            InlineKeyboardMarkup::Ptr keyboard(new InlineKeyboardMarkup);

            vector<InlineKeyboardButton::Ptr> menu;
            InlineKeyboardButton::Ptr DefaultButton(new InlineKeyboardButton);
            DefaultButton->text = "Default";
            DefaultButton->callbackData = "Default";

            InlineKeyboardButton::Ptr CustomButton(new InlineKeyboardButton);
            CustomButton->text = "Custom";
            CustomButton->callbackData = "Custom";

            menu.push_back(DefaultButton);
            menu.push_back(CustomButton);
            keyboard->inlineKeyboard.push_back(menu);
            SetupIntrfaceMenu = keyboard;
        }

        void LoginMenuIni() {
            InlineKeyboardMarkup::Ptr keyboard(new InlineKeyboardMarkup);

            vector<InlineKeyboardButton::Ptr> menu;
            InlineKeyboardButton::Ptr SetupButton(new InlineKeyboardButton);
            SetupButton->text = "Interface";
            SetupButton->callbackData = "Interface";

            InlineKeyboardButton::Ptr SendMessageButton(new InlineKeyboardButton);
            SendMessageButton->text = "SendMessage";
            SendMessageButton->callbackData = "SendMessage";

            InlineKeyboardButton::Ptr DeleteUserButton(new InlineKeyboardButton);
            DeleteUserButton->text = "Delete user";
            DeleteUserButton->callbackData = "Delete user";

            InlineKeyboardButton::Ptr ExitButton(new InlineKeyboardButton);
            ExitButton->text = "Exit";
            ExitButton->callbackData = "Exit";


            menu.push_back(SetupButton);
            menu.push_back(SendMessageButton);
            menu.push_back(DeleteUserButton);
            menu.push_back(ExitButton);
            keyboard->inlineKeyboard.push_back(menu);
            LoginMenu = keyboard;
        }

        void SendMessageMenuIni() {
            // ToDo динамическое меню с загрузкой данных из БД
        }

        void CustomInterfaceMenuIni() {
            InlineKeyboardMarkup::Ptr keyboard(new InlineKeyboardMarkup);

            vector<InlineKeyboardButton::Ptr> menu;
            InlineKeyboardButton::Ptr AddButton(new InlineKeyboardButton);
            AddButton->text = "Add element";
            AddButton->callbackData = "Add element";

            InlineKeyboardButton::Ptr DeleteButton(new InlineKeyboardButton);
            DeleteButton->text = "Delete element";
            DeleteButton->callbackData = "Delete element";

            menu.push_back(AddButton);
            menu.push_back(DeleteButton);
            keyboard->inlineKeyboard.push_back(menu);
            CustomInterfaceMenu = keyboard;
        }

        void CustomElementsMenuIni() {
            // ToDo динамическое меню с загрузкой данных из БД
            // в цикле заполнение вектора кнопок
        }

        void MakeSolutionMenuIni() {
            InlineKeyboardMarkup::Ptr keyboard(new InlineKeyboardMarkup);

            vector<InlineKeyboardButton::Ptr> menu;
            InlineKeyboardButton::Ptr YesButton(new InlineKeyboardButton);
            YesButton->text = "Yes";
            YesButton->callbackData = "Yes";

            InlineKeyboardButton::Ptr BackButton(new InlineKeyboardButton);
            BackButton->text = "Back";
            BackButton->callbackData = "Back";

            menu.push_back(YesButton);
            menu.push_back(BackButton);
            keyboard->inlineKeyboard.push_back(menu);
            MakeSolutionMenu = keyboard;
        }

        // Клавиатура приветствия
        InlineKeyboardMarkup::Ptr WelcomeMenu;

        // Клавиатура, при загрузке фотографий
        InlineKeyboardMarkup::Ptr Stop;

        // Клавиатура для работы между стадий
        InlineKeyboardMarkup::Ptr Interval;

        // Клавиатура для выбора интерфейса
        InlineKeyboardMarkup::Ptr SetupIntrfaceMenu;

        // Клавиатура после входа
        InlineKeyboardMarkup::Ptr LoginMenu;

        // Клавиатура выбора пользователя  ДИНАМИЧЕСКАЯ
        InlineKeyboardMarkup::Ptr SendMessageMenu;

        // Клавиатура настройки кастомного интерфейса
        InlineKeyboardMarkup::Ptr CustomInterfaceMenu;

        // Клавиатура элементов зеркала    ДИНАМИЧЕСКАЯ
        InlineKeyboardMarkup::Ptr CustomElementsMenu;

        // Клавиатура удаления/выхода пользователя
        InlineKeyboardMarkup::Ptr MakeSolutionMenu;



};




int main() {
    // Инициализация Бота - Токен
    TgBot::Bot bot("705169992:AAF-gzGZGlAtUT4s3QFICRlgMoKDg0PC8ok");
    
    vector<InlineKeyboardMarkup::Ptr> keyboard;
    
    // Класс клавиатур
    BotKeyboard my;

    // Инициализаци вектора клавиатур
    keyboard.push_back(my.GetWelcomeMenuKeyboard());
    keyboard.push_back(my.GetIntervalKeyboard());
    keyboard.push_back(my.GetStopKeyboard());
    keyboard.push_back(my.GetSetupKeyboard());

    // Класс мессенджера    
    Messenger mes;
   
    // Команда Start - возвращает приветсвенное сообщение и меню
    bot.getEvents().onCommand("start",[&bot, &my, &mes](Message::Ptr message) {
        string response = mes.GetWelcomeMessage();
        bot.getApi().sendMessage(message->chat->id, response, false, 0, my.GetWelcomeMenuKeyboard(), "Markdown");
    });

    // Флаг стадии    
    int stageflag = 0;
    // Счетчик фотографий
    int photocount = 0;
    // Строка, хранящая текущую стадию
    std::string stage = "";


    bot.getEvents().onCallbackQuery([&bot,&mes, &my, &stageflag, &stage, &photocount](CallbackQuery::Ptr query) {
    // Сценарий Регистрации
        string response = "";
        // Если пришел запрос на регистрацию - начинаем регистрацию
        if (StringTools::startsWith(query->data, "Sign Up")) {
            stage = "Sign up";
            stageflag = 1;
            response = mes.GetSignUpNameMessage();
            bot.getApi().sendMessage(query->message->chat->id, response);
        } else if (StringTools::startsWith(query->data, "Back")) {
            response = mes.SignUpResp[stageflag];
            
            // Если стадия 5 регистрации - прием фотографий, то выводим для нее соотвествующее меню.
            if (stageflag == 5) {
                bot.getApi().sendMessage(query->message->chat->id, response, false, 0, my.GetStopKeyboard(), "Markdown");
            } else if (stageflag > 0) {
                // Если стадия 1 регистрации - запрос имени, то выводим соотвествующее сообщение.
                if (stageflag == 1) {
                    bot.getApi().sendMessage(query->message->chat->id, response);
                    stageflag++;
                } else {
                    bot.getApi().sendMessage(query->message->chat->id, response, false, 0, my.GetIntervalKeyboard(), "Markdown");
                }
                stageflag--;
                // Иначе выводим приветсвие
            } else {
                bot.getApi().sendMessage(query->message->chat->id, response, false, 0, my.GetWelcomeMenuKeyboard(), "Markdown");
            }

        } else if (StringTools::startsWith(query->data, "Exit")) {
            response = "Отмена регистрации. Наберите /start, чтобы продолжить работу.";
            stageflag = 0;
            photocount = 0;
            stage = "";
            bot.getApi().sendMessage(query->message->chat->id, response);

        } else if (StringTools::startsWith(query->data, "Stop")) {
            response = mes.GetPhotoCheckMessage();
            stageflag++;
            photocount = 0;
            bot.getApi().sendMessage(query->message->chat->id, response, false, 0, my.GetIntervalKeyboard(), "Markdown");
        } else if (StringTools::startsWith(query->data, "Default")) {
            stage = "";
            stageflag = 0;
            response = mes.GetDefautInterfaceMessage();
           
            // ToDo добавить отправку на бд дефолтные настройки БД
           
            bot.getApi().sendMessage(query->message->chat->id, response);
        } else if (StringTools::startsWith(query->data, "Login")) {
            // ToDo проверка chat-id, если его нет в бд, то запросить логин и пароль


            response = mes.GetLoginInputMessage();
            stage = "Login";
            stageflag = 1;
            bot.getApi().sendMessage(query->message->chat->id, response, false, 0, my.GetLoginKeyboard(), "Markdown");
        } 
    });


    bot.getEvents().onAnyMessage([&bot, &my, &stage, &mes, &stageflag, &photocount](TgBot::Message::Ptr message) {
        if (stage == "Sign up") {
            if (stageflag == 1) {

                // ToDo добавить отправку имени на БД
            
                string response = "Приятно познакомиться, " + message->text + ". " + mes.GetSignUpLoginMessage();
                bot.getApi().sendMessage(message->chat->id, response, false, 0, my.GetIntervalKeyboard(), "Markdown");
                stageflag++;
            } else if (stageflag == 2) {
            
                // ToDo добавить проверку и отправку логина на БД
            
                string response = mes.GetSignUpPasswordMessage();
                bot.getApi().sendMessage(message->chat->id, response, false, 0, my.GetIntervalKeyboard(), "Markdown");
                stageflag++;
            } else if (stageflag == 3) {
            
                // Todo добавить проверку и отправку на БД 
            
                string response = mes.GetSignUpCheckPasswordMessage();
                bot.getApi().sendMessage(message->chat->id, response, false, 0, my.GetIntervalKeyboard(), "Markdown");
                stageflag++;
            } else if (stageflag == 4) {
                string response = mes.GetSignUpEndMessage();
                bot.getApi().sendMessage(message->chat->id, response, false, 0, my.GetStopKeyboard(), "Markdown");
                stageflag++;
            } else if(stageflag == 5) {
                
                // ToDo: Json to Img и отправка файлов на бд


                //photocount++;
                //TgBot::PhotoSize::Ptr photo = message->photo[0];
                //std::string file = bot.getApi().getFile(photo->fileId);
                string response = "Фотка пришла";
                bot.getApi().sendMessage(message->chat->id, response, false, 0, my.GetStopKeyboard(), "Markdown");
                

            } else if(stageflag == 6) {

                // ToDo: отправка message-id на бд
                
                photocount = 0;
                string response = mes.GetPhotoCheckMessage();
                bot.getApi().sendMessage(message->chat->id, response, false, 0, my.GetSetupKeyboard(), "Markdown");   
            }
        }
    });

        signal(SIGINT, [](int s) {
            printf("SIGINT got\n");
            exit(0);
        });

        try {
            printf("Bot username: %s\n", bot.getApi().getMe()->username.c_str());
            bot.getApi().deleteWebhook();

            TgLongPoll longPoll(bot);
            while (true) {
                printf("Long poll started\n");
                longPoll.start();
            }
        } catch (exception& e) {
            printf("error: %s\n", e.what());
        }

        return 0;
    }



