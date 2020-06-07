grammar Assertion;


assertion
      :   implication (OR implication)* EOF
      ;

implication
      :   LP FA VAR (CM VAR)* DOT disjunction IMP disjunction RP
      ;

disjunction
      :   conjunction (OR conjunction)*
      ;

conjunction
      :   term (AND term)*
      ;

term
      :   TRUE
      |   func op num
      |   func op func
//      |   VAR op num                        // bounds for whole input
//      |   VAR op array                      // bounds for input segments
      |   VAR LB INT RB op num              // bounds for each element
      |   VAR LB INT RB op VAR LB INT RB    // fairness
      ;

func
      :   D0 LP VAR CM VAR RP           // 0-norm distance function
      |   D2 LP VAR CM VAR RP           // 2-norm distance function
      |   DI LP VAR CM VAR RP           // inf-norm distance function
      |   ARG_MAX LP VAR RP             // arg max index for output
      |   ARG_MIN LP VAR RP             // arg min index for output
      |   LIN_INP LP VAR CM array RP    // linear application for input
      |   LIN_OUT LP VAR CM array RP    // linear application for output
      ;

op
      :   GE
      |   GT
      |   LE
      |   LT
      |   EQ
      |   NE
      ;

array
      :   LB num (CM num)* RB
      ;

num
      :   INT
      |   FLT
      ;


TRUE  :   'True' ;

D0    :   'd0' ;
D2    :   'd2' ;
DI    :   'di' ;
ARG_MAX : 'arg_max' ;
ARG_MIN : 'arg_min' ;
LIN_INP : 'lin_inp' ;
LIN_OUT : 'lin_out' ;

VAR   :   [a-zA-Z_][a-zA-Z0-9_]* ;
INT   :   '0' | '-'? [1-9][0-9]*;
FLT   :   ('0' | '-'? [1-9][0-9]*) '.' [0-9]*;

LP    :   '(' ;
RP    :   ')' ;
LB    :   '[' ;
RB    :   ']' ;
CM    :   ',' ;
FA    :   '\\A' ;
DOT   :   '.' ;

GE    :   '>=' ;
GT    :   '>' ;
LE    :   '<=' ;
LT    :   '<' ;
EQ    :   '=' ;
NE    :   '!=' ;

OR    :   '\\/' ;
AND   :   '/\\' ;
IMP   :   '=>' ;

WS    :   [ \t\r\n]+ -> skip ;