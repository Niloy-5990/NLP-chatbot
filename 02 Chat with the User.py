import pandas

dataframe = pandas.read_csv("qna_chitchat_professional.csv")

print(dataframe)



dataframe.dropna(inplace=True)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()




import numpy

vectorizer.fit(numpy.concatenate((dataframe.Question,
                                  dataframe.Answer)))

vectorized_questions = vectorizer.transform(dataframe.Question)

print(vectorized_questions)
print(vectorizer.vocabulary_)



from sklearn.metrics.pairwise import cosine_similarity

while True:

    user_input = input()

    print("Your input is :", user_input)

    if user_input == "quit":
        print("Program exited successfully.")

        break

    vectorized_user_input = vectorizer.transform([user_input])

    similarities = cosine_similarity(vectorized_user_input,
                                     vectorized_questions)

    closest_question = numpy.argmax(similarities,
                                    axis=1)

    # print("Similarities: ", similarities)
    #
    print("Closest question: ", closest_question)

    answer = dataframe.Answer.iloc[closest_question].values[0]

    print("Answer: ", answer)
