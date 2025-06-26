% Somma dei pesi dei termini per una determinata classe.
% Caso base: lista vuota → score = 0
sum_bias([], _, 0).

% Caso 1: il termine ha un peso per la classe → aggiungilo alla somma ricorsiva
sum_bias([Word|Rest], Class, Score) :-
    term_weight(Word, Class, W),
    sum_bias(Rest, Class, R),
    Score is W + R.

% Caso 2: il termine NON ha un peso per la classe → ignoralo e prosegui
sum_bias([Word|Rest], Class, Score) :-
    \+ term_weight(Word, Class, _),
    sum_bias(Rest, Class, Score).

% Classifica un testo trovando la classe con somma di pesi più alta
classify_text(Words, Class) :-
    sum_bias(Words, left, L),
    sum_bias(Words, center, C),
    sum_bias(Words, right, R),
    max_member((_, Class), [(L, left), (C, center), (R, right)]).
