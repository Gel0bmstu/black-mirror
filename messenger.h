#ifndef MESSENGER_H
#define MESSENGER_H

#include <string>

class Messenger {
public:
    Messenger(){
        SignUpResp.push_back(WelcomeMessage);
        SignUpResp.push_back(SignUpNameMessage);
        SignUpResp.push_back(SignUpLoginMessage);
        SignUpResp.push_back(SignUpPasswordMessage);
        SignUpResp.push_back(SignUpCheckPasswordMessage);
        SignUpResp.push_back(SignUpEndMessage);
        SignUpResp.push_back(PhotoCheckMessage);
        //SignUpResp.push_back(DefautSetupMessage);
    };
    std::string GetWelcomeMessage() {
        return WelcomeMessage;
    }
    std::string GetSignUpNameMessage() {
        return SignUpNameMessage;
    }
    std::string GetSignUpLoginMessage() {
        return SignUpLoginMessage ;
    }
    std::string GetSignUpPasswordMessage() {
        return SignUpPasswordMessage;
    }
    std::string GetSignUpCheckPasswordMessage() {
        return SignUpCheckPasswordMessage;
    }
    std::string GetSignUpEndMessage() {
        return SignUpEndMessage;
    }

    std::string GetPhotoCheckMessage() {
        return PhotoCheckMessage;
    }

    std::string GetLoginLoginInputMessage() {
        return LoginLoginInput;
    }

    std::string GetLoginPaswordInputMessage() {
        return LoginPaswordInput;
    }

    std::string GetLoginInputMessage() {
        return LoginInputMessage;
    }

    std::string GetSendMessage() {
        return SendMessage;
    }

    std::string GetSetInterfaceMessage() {
        return SetInterface;
    }

    std::string GetCustomInterfaceMessage() {
       return CustomInterface;
    }

    std::string GetDefautInterfaceMessage() {
       return DefautInterface;
    }

    std::string GetDestroyInterfaceMessage() {
       return DestroyInterface;
    }

    std::string GetInstallInterface() {
       return InstallInterface;
    }


    
    //std::string GetDefautSetupMessage() {
    //    return DefautSetupMessage;
    //}
    std::vector<string> SignUpResp;
    


//    std::string Get() {
//        return ;
//    }
//    std::string Get() {
//        return ;
//    }

    //std::

private:
    // Sign Up
    std::string WelcomeMessage = "Привет, я, бот-помощник, призван помочь Вам в работе с Вашим BlackMirror. Пожалуйста войдите или, если у вас еще нет аккаунат, зарегестрируйтесь :)";
    std::string SignUpNameMessage = "Отлично, очень рад, что Вы решили стать одним из нас! Как Вас зовут?";
    std::string SignUpLoginMessage = "Превосходно, напишите свой логин.";
    std::string SignUpPasswordMessage = "Классный ник! А теперь нам нужен твой пароль";
    std::string SignUpCheckPasswordMessage = "Ого, чет сложно! А можете повторить, а то я не успел.";
    std::string SignUpEndMessage = "Поздравляю, теперь Зеркало знает Вас лишь только на словах.Осталось всего ничего, отправьте мне фотографии с вашим лицом(желательно только вашим)";
    std::string PhotoCheckMessage = "А Вы ничего!Регистрация закончена. Хотите ли вы настроить интерфейс Зеркала под себя или оставим это дизайнерам(не зря же мы им платим) ?";
    //std::string DefautSetupMessage = "Пройдите к Зеркалу и наслаждайтесь!";
    
    // Login
    std::string LoginLoginInput = "К сожалению, Вас не узнать. Напишите ваш логин!";
    std::string LoginLoginError = "К сожалению, таких у нас нет. Наберите /start, чтобы пройти регистрацию!";

    std::string LoginPaswordInput = "Что-то такое припоминаю, наберите пароль!";
    std::string LoginPaswordError = "Неверно, попробуйте еще!";

    std::string LoginInputMessage = "С возвращением! Мы по вам скучали)";

    std::string SendMessage = "Кому отправим сообщение?";
    std::string WhatMessage = "Напишите сообщение";
    std::string SentMessage = "Пташка с вашим сообщением вылетела! Что-то еще?";

    std::string SetInterface = "Какой хотите интерфейс?";
    std::string DefautInterface = "Чтож, наши дизайнеры очень постарались(нет). Пройдите к Зеркалу и наслаждайтесь!";
    std::string CustomInterface = "Добавить или удалить?";
    std::string DestroyInterface = "Вы его больше не увидите! Удалить что-то еще?";
    std::string InstallInterface = "Уже купили и поставили! Добавить что-то еще?";

    std::string DeleteUser = "Вы точно хотите этого?";

    std::string AddPhoto = "Кидайте новые фотки по одной, а когда устанете - нажмите Stop.";

};

#endif // MESSENGER_H
