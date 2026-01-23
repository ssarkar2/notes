// we want to model playing cards. Say 2 suits and 3 numbers 1,2,3. So a total of 6 cards


def CLUB;
def HEART;

class card <int card_num, string suit_str, string color, dag crd_repr> {
    int CardNum = card_num;
    string Suit_Str = suit_str;
    string Color = color;
    dag Card_Repr = crd_repr;
}


class club_card <int card_num>
  : card<card_num, "club", "black", (CLUB card_num)>;


class heart_card <int card_num>
  : card<card_num, "heart", "red", (HEART card_num)>;


def CLUB_1 : club_card<1>;
def CLUB_2 : club_card<2>;
def CLUB_3 : club_card<3>;
def HEART_1 : heart_card<1>;
def HEART_2 : heart_card<2>;
def HEART_3 : heart_card<3>;


/*
Sample record:
def HEART_3 {   // card heart_card
  int CardNum = 3;
  string Suit_Str = "heart";
  string Color = "red";
  dag Card_Repr = (HEART 3);
}
*/



// Now with multiclasses

multiclass cardnum <int card_num> {
    def _CLUB: card<card_num, "club", "black", (CLUB card_num)>;
    def _HEART: card<card_num, "heart", "red", (HEART card_num)>;
}


defm ONE : cardnum<1>;
defm TWO : cardnum<2>;
defm THREE : cardnum<3>;

/*
def THREE_HEART {       // card
  int CardNum = 3;
  string Suit_Str = "heart";
  string Color = "red";
  dag Card_Repr = (HEART 3);
}
*/