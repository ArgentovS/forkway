import pandas as pd
import nltk
import pymorphy2
import re
import missingno as msno

def text_converter(elem):

    # удаляем html-теги и неалфавитные символы
    text = re.sub('[^a-zа-я\+]', ' ', re.sub('<.+?>', '', elem).lower())

    # токенизируем текст по словам
    text = nltk.word_tokenize(text)

    # лемматирзируем английские слова
    lemmatizer = nltk.stem.WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text]

    # лемматирзируем русские слова
    lemmatizer = pymorphy2.MorphAnalyzer()
    text = [lemmatizer.parse(word)[0].normal_form for word in text]

    # убираем стоп-слова и возвращаем список полученных отнормированных слов
    stop_words = nltk.corpus.stopwords.words('english')
    stop_words.extend(nltk.corpus.stopwords.words('russian'))

    return ' '.join([word for word in text if not(word in stop_words)])


def labeled_skills(df):
    # Сформируем список столбцов исходного датасета
    # с размеченными в объявлении навыками
    # первым элементом списка оставим 'id' для сохранения размерности датафрейма
    cols = ['id']
    for col in df.columns:
        if 'key' in col:
            cols.append(col)
    # Сформируем и отобразим датафрейм размеченных навыков
    df_skills = df[cols]
    msno.matrix(df_skills);
    index_none = df_skills.loc[df_skills[cols[1:]].isna().all(axis=1)].index

    return df_skills, index_none


def skill_vectors(df, df_skills, max_skills):
    # Сформируем список уникальных навыков в исходных данных
    unique_skills = []
    for col in df_skills.columns[1:]: # Начинаем со второго столбца, т.к. в первом храниться 'id'
        unique_skills.extend(df_skills[col].unique())
    unique_skills = [skill for skill in list(set(unique_skills))
                     if pd.notnull(skill)]

    # Сформируем датафрейм с векторами таргетов (признаками навков)
    df_y = pd.concat([df['id'],
                     pd.DataFrame(columns=unique_skills)], axis=1).fillna(0)
    for i, row in df[df_skills.columns[1:]].iterrows():
        for key, skill in row.iteritems():
            if pd.notnull(skill):
                df_y.loc[i, skill] = 1
    df_skills_max = df_y.drop(['id'], axis=1)

    # Отсортируем компетенции по востребованности
    df_sort_skills = pd.DataFrame(columns=df_skills_max.columns)
    for col in df_skills_max.columns:
        df_sort_skills.loc[0, col] = df_skills_max[col].sum()

    # Сформируем векторы с максимальной востребованными компетенциями
    sort = dict(df_sort_skills)
    sort_1 = {elem[0]: elem[1][0] for elem in sort.items()}
    d = {x: sort_1[x] for x in sorted(sort_1, key=sort_1.get, reverse=True)}
    list_max_col = [elem for elem in d]
    df = pd.concat([df, df_skills_max], axis=1)

    return df_skills_max[list_max_col[:max_skills]], d, list_max_col[:max_skills], df

