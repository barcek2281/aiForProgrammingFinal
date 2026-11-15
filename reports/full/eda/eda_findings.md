# EDA findings (full data)

1. Class imbalance: 35 sections; the largest ('Қаржы') has 64,854 acts (30.04%), the smallest ('Тәуелсіз Мемлекеттер Достастығы') 110. The top three sections cover 56.9% of all acts, so accuracy would be misleading and F1-macro is used.
2. Time: the yearly number of acts grows from 18 in 1991 to a peak of 15,092 in 2023. Acts adopted from 2023 make up 20.1% of the shown period, which gives a test set of a reasonable size for a time-based split.
3. Title length: median 19 words (IQR 12-25), maximum 525. 6,526 titles (3.0%) are longer than the IQR limit of 44 words and 4 consist of a single word. Texts are short, so sparse TF-IDF features are a sensible first choice.
4. Vocabulary: 34,361 distinct word forms in 3,965,957 words. The 20 most frequent words ('туралы', 'жылғы', 'енгізу', ...) give 35.4% of all words and are mostly function and formula words, so stop-word removal and IDF weighting are needed. 11,438 word forms occur only once - a sign of rich Kazakh morphology that stemming or character n-grams should reduce.
5. Class drift: the share of 'Қаржы' changes from 16% in 1995-1999 to 40% in the latest period. The class mix is not stable in time, so a random split would be too optimistic; the split is done by date.
6. Train/test shift: 'Қаржы' grows from 27.7% to 39.1% of acts, 'Халықаралық қатынастар' falls from 4.1% to 1.1%. 3 sections have fewer than 20 test acts, so their per-class F1 will be unstable.
7. Areas of regulation: an act has 1.99 areas on average (up to 12); 70.8% of acts have more than one. The most frequent area covers 49.87% of acts and describes the procedure, not the topic. That is why the single-valued 'database_section' is the target and the multi-valued field would need a multi-label model.
