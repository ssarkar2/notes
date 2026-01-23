
def CLUB;
def HEART;

class card <int card_num, string suit_str, string color, dag crd_repr> {
    int CardNum = card_num;
    string Suit_Str = suit_str;
    string Color = color;
    dag Card_Repr = crd_repr;
}

multiclass cardnum <int card_num> {
    def _CLUB: card<card_num, "club", "black", (CLUB card_num)>;
    def _HEART: card<card_num, "heart", "red", (HEART card_num)>;
}


defset list<card> Deck = {
    defm ONE : cardnum<1>;
    defm TWO : cardnum<2>;
    defm THREE : cardnum<3>;
    def JOKER : card<?, ?, ?, ?>;
}

class laminated_card <card c> {
    card C = c;
}


foreach c = Deck in { def : laminated_card <c>;}
