package utilities;

public class Utilities{
  
  public enum Digits{
    ZERO("0"),
    ONE("1"),
    TWO("2"),
    THREE("3"),
    FOUR("4"),
    FIVE("5"),
    SIX("6"),
    SEVEN("7"),
    EIGHT("8"),
    NINE("9");
    
    Digits(String symbol){
      symbol_ = symbol;
    }
    
    private String symbol_;
  }
  
  public enum Operators{
    PLUS("+"),
    MINUS("-"),
    // TIMES("*"),
    DIV("/"), // Horizontal line in TeX format.
    // HAT("^"), // The ^ symbol appears above the symbol in TeX format.
    SQRT("\\sqrt{}"), // The known square root symbol in TeX format.
    EQUALS("=");
    
    Operators(String symbol){
      symbol_ = symbol;
    }
    
    private String symbol_;
  }
  
  public enum Separators{
    LEFT_PARENTHESIS("("),
    RIGHT_PARENTHESIS(")"),
    COMMA(",");
    
    Separators(String symbol){
      symbol_ = symbol;
    }
    
    private String symbol_;
  }
  
  public enum Variables{
    X("x"),
    Y("y");
    
    Variables(String symbol){
      symbol_ = symbol;
    }
    
    private String symbol_;
  }
  
  public enum UpperCaseLetters{
    A("A"), B("B"), C("C"), D("D"), E("E"),
    F("F"), G("G"), H("H"), I("I"), J("J"),
    K("K"), L("L"), M("M"), N("N"), O("O"),
    P("P"), Q("Q"), R("R"), S("S"), T("T"),
    U("U"), V("V"), W("W"), X("X"), Y("Y"),
    Z("Z");
    
    UpperCaseLetters(String symbol){
      symbol_ = symbol;
    }
    
    private String symbol_;
  }
  
  // functions: sin, cos, tan, exp, ln.
  public enum LowerCaseLetters{
    A("a"), /*B("b"),*/ C("c"), /*D("d"),*/ E("e"),
    /*F("f"),*/ /*G("g"),*/ /*H("h"),*/ I("i"), /*J("j"),*/
    /*K("k"),*/ L("l"), /*M("m"),*/ N("n"), O("o"),
    P("p"), /*Q("q"),*/ /*R("r"),*/ S("s"), T("t"),
    /*U("u"),*/ /*V("v"),*/ /*W("w"),*/ X("x"), /*Y("y"),*/
    /*Z("z")*/;
    
    LowerCaseLetters(String symbol){
      symbol_ = symbol;
    }
    
    private String symbol_;
  }
  
  public static final String[] ALL_SYMBOLS = {Digits.ZERO.name(),
                                              Digits.ONE.name(),
                                              Digits.TWO.name(),
                                              Digits.THREE.name(),
                                              Digits.FOUR.name(),
                                              Digits.FIVE.name(),
                                              Digits.SIX.name(),
                                              Digits.SEVEN.name(),
                                              Digits.EIGHT.name(),
                                              Digits.NINE.name(),
                                              Operators.PLUS.name(),
                                              Operators.MINUS.name(),
                                              Operators.EQUALS.name(),
                                              Variables.X.name(),
                                              Variables.Y.name()};
  
  // TODO
  // Notice that Variables.X.name() is equal to UpperCaseLetters.X.name().
  // This means that the one that appears second in the following if
  // else if statement will never be choosen.
  public static int nameToValue(String name){
    if(name == Digits.ZERO.name()){
      return 0;
    }
    else if(name == Digits.ONE.name()){
      return 1;
    }
    else if(name == Digits.TWO.name()){
      return 2;
    }
    else if(name == Digits.THREE.name()){
      return 3;
    }
    else if(name == Digits.FOUR.name()){
      return 4;
    }
    else if(name == Digits.FIVE.name()){
      return 5;
    }
    else if(name == Digits.SIX.name()){
      return 6;
    }
    else if(name == Digits.SEVEN.name()){
      return 7;
    }
    else if(name == Digits.EIGHT.name()){
      return 8;
    }
    else if(name == Digits.NINE.name()){
      return 9;
    }
    else if(name == Operators.PLUS.name()){
      return 10;
    }
    else if(name == Operators.MINUS.name()){
      return 11;
    }
    else if(name == Operators.DIV.name()){
      return 12;
    }
    else if(name == Operators.SQRT.name()){
      return 13;
    }
    else if(name == Operators.EQUALS.name()){
      return 14;
    }
    else if(name == Separators.LEFT_PARENTHESIS.name()){
      return 15;
    }
    else if(name == Separators.RIGHT_PARENTHESIS.name()){
      return 16;
    }
    else if(name == Separators.COMMA.name()){
      return 17;
    }
    else if(name == Variables.X.name()){
      return 18;
    }
    else if(name == Variables.Y.name()){
      return 19;
    }
    else if(name == UpperCaseLetters.A.name()){
      return 20;
    }
    else if(name == UpperCaseLetters.B.name()){
      return 21;
    }
    else if(name == UpperCaseLetters.C.name()){
      return 22;
    }
    else if(name == UpperCaseLetters.D.name()){
      return 23;
    }
    else if(name == UpperCaseLetters.E.name()){
      return 24;
    }
    else if(name == UpperCaseLetters.F.name()){
      return 25;
    }
    else if(name == UpperCaseLetters.G.name()){
      return 26;
    }
    else if(name == UpperCaseLetters.H.name()){
      return 27;
    }
    else if(name == UpperCaseLetters.I.name()){
      return 28;
    }
    else if(name == UpperCaseLetters.J.name()){
      return 29;
    }
    else if(name == UpperCaseLetters.K.name()){
      return 30;
    }
    else if(name == UpperCaseLetters.L.name()){
      return 31;
    }
    else if(name == UpperCaseLetters.M.name()){
      return 32;
    }
    else if(name == UpperCaseLetters.N.name()){
      return 33;
    }
    else if(name == UpperCaseLetters.O.name()){
      return 34;
    }
    else if(name == UpperCaseLetters.P.name()){
      return 35;
    }
    else if(name == UpperCaseLetters.Q.name()){
      return 36;
    }
    else if(name == UpperCaseLetters.R.name()){
      return 37;
    }
    else if(name == UpperCaseLetters.S.name()){
      return 38;
    }
    else if(name == UpperCaseLetters.T.name()){
      return 39;
    }
    else if(name == UpperCaseLetters.U.name()){
      return 40;
    }
    else if(name == UpperCaseLetters.V.name()){
      return 41;
    }
    else if(name == UpperCaseLetters.W.name()){
      return 42;
    }
    else if(name == UpperCaseLetters.X.name()){
      return 43;
    }
    else if(name == UpperCaseLetters.Y.name()){
      return 44;
    }
    else if(name == UpperCaseLetters.Z.name()){
      return 45;
    }
    else{
      return -1;
    }
  }
}