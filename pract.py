class Solution(object):
    def strongPasswordChecker(self, password):
        steps_to_make_password_strong = 0

        replace_password_string = self.check_password_string(password)

        change_password_length = self.check_password_length(password)
        
        repeating_chars = self.check_repeating_chars(password)      

        print(replace_password_string, change_password_length, repeating_chars)

        return repeating_chars + max(replace_password_string, change_password_length)

        

    def check_password_length(self, password):
        if len(password) < 6:
            return 6 - len(password)
        if len(password) > 20:
            return len(password) - 20
        return 0

    def check_lower_case(self, password):
        for letter in password:
            if letter.islower(): return 0
        return 1

    def check_upper_case(self, password):
        for letter in password:
            if letter.isupper(): return 0
        return 1

    def check_digit(self, password):
        for letter in password:
            if letter.isdigit(): return 0
        return 1
    
    def check_password_string(self, password):
        counter = 0
        counter += self.check_lower_case(password)
        counter += self.check_upper_case(password)
        counter += self.check_digit(password)
        return counter

    def check_repeating_chars(self, password):
        counter = 0
        for index in range(len(password)-2):
            if password[index] == password[index+1] and password[index] == password[index+2]:
                counter += 1
        return counter




a = Solution()
print(a.strongPasswordChecker('1111111111'))
        