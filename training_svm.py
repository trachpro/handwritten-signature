from sklearn import svm

X, y = prepare_dependent_process_data('/content/Dataset/dataset1/real')
# print(X.shape)
clf = svm.SVC(gamma='scale')
clf.fit(X, y)

feature = get_feature('/content/Dataset/dataset1/real/00400004.png')
# feature = feature.reshape(2048)

result = clf.predict(feature)
print(result)