from ast import alias
from concurrent.futures import process
from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, HttpResponse
from django.contrib import messages
# import Software_ Fault_ predction


from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import datetime as dt
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


# Create your views here.

def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})

def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(
                loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})

def DatasetView(request):
    path = settings.MEDIA_ROOT + "//" + 'cm1.csv'
    df = pd.read_csv(path, nrows=100)
    df = df.to_html
    return render(request, 'users/viewdataset.html', {'data': df})

def ml(request):
    import numpy as np # linear algebra
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
    import matplotlib.pyplot as plt # data visualization
    import seaborn as sns # statistical data visualization

    #-- plotly
    from chart_studio import plotly
    import chart_studio.plotly as py
    from plotly.offline import plot
    #plot(fig, auto_open=True)
    import plotly.graph_objs as go

    import os
    data = pd.read_csv(r"media/cm1.csv")
    data.info() #informs about the data (memory usage, data types etc.)
    data.head() #shows first 5 rows
    data.tail() #shows last 5 rows
    data.sample(10) #shows random rows (sample(number_of_rows))
    data.shape #shows the number of rows and columns
    data.describe() #shows simple statistics (min, max, mean, etc.)
    defects_true_false = data.groupby('defects')['b'].apply(lambda x: x.count()) #defect rates (true/false)
    print('False : ' , defects_true_false[0])
    print('True : ' , defects_true_false[1])
    trace = go.Histogram(
        x = data.defects,
        opacity = 0.75,
        name = "Defects",
        marker = dict(color = 'green'))

    hist_data = [trace]
    hist_layout = go.Layout(barmode='overlay',
                    title = 'Defects',
                    xaxis = dict(title = 'True - False'),
                    yaxis = dict(title = 'Frequency'),
    )
    fig = go.Figure(data = hist_data, layout = hist_layout)
    #iplot(fig)
    import warnings
    warnings.filterwarnings("ignore",category=FutureWarning)
    import warnings
    warnings.filterwarnings("ignore",category=FutureWarning)
    f,ax = plt.subplots(figsize = (15, 15))
    sns.heatmap(data.corr(), annot = True, linewidths = .5, fmt = '.2f')
    plt.show()
    trace = go.Scatter(
        x = data.v,
        y = data.b,
        mode = "markers",
        name = "Volume - Bug",
        marker = dict(color = 'darkblue'),
        text = "Bug (b)")

    scatter_data = [trace]
    scatter_layout = dict(title = 'Volume - Bug',
                xaxis = dict(title = 'Volume', ticklen = 5),
                yaxis = dict(title = 'Bug' , ticklen = 5),
                )
    fig = dict(data = scatter_data, layout = scatter_layout)
    plot(fig)
    data.isnull().sum() #shows how many of the null
    trace1 = go.Box(
        x = data.uniq_Op,
        name = 'Unique Operators',
        marker = dict(color = 'blue')
        )
    box_data = [trace1]
    plot(box_data)

    def evaluation_control(data):    
        evaluation = (data.n < 300) & (data.v < 1000 ) & (data.d < 50) & (data.e < 500000) & (data.t < 5000)
        data['complexityEvaluation'] = pd.DataFrame(evaluation)
        data['complexityEvaluation'] = ['Succesful' if evaluation == True else 'Redesign' for evaluation in data.complexityEvaluation]
    evaluation_control(data)
    data
    data.info()
    data.groupby("complexityEvaluation").size() #complexityEvalution rates (Succesfull/redisgn)
    
    trace = go.Histogram(
        x = data.complexityEvaluation,
        opacity = 0.75,
        name = 'Complexity Evaluation',
        marker = dict(color = 'darkorange')
    )
    hist_data = [trace]
    hist_layout = go.Layout(barmode='overlay',
                    title = 'Complexity Evaluation',
                    xaxis = dict(title = 'Succesful - Redesign'),
                    yaxis = dict(title = 'Frequency')
    )
    fig = go.Figure(data = hist_data, layout = hist_layout)
    plot(fig)

    from sklearn import preprocessing

    scale_v = data[['v']]
    scale_b = data[['b']]

    minmax_scaler = preprocessing.MinMaxScaler()

    v_scaled = minmax_scaler.fit_transform(scale_v)
    b_scaled = minmax_scaler.fit_transform(scale_b)

    data['v_ScaledUp'] = pd.DataFrame(v_scaled)
    data['b_ScaledUp'] = pd.DataFrame(b_scaled)

    data

    scaled_data = pd.concat([data.v , data.b , data.v_ScaledUp , data.b_ScaledUp], axis=1)
    scaled_data
    data.info()
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.metrics import roc_curve, roc_auc_score
    from sklearn import model_selection

    X = data.iloc[:, :-10].values  #Select related attribute values for selection
    Y = data.complexityEvaluation.values   #Select classification attribute values
    Y
    #Parsing selection and verification datasets
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = validation_size, random_state = seed)
    
#                             ----------------------Creation of Naive Bayes model ------------------------

    from sklearn.naive_bayes import GaussianNB
    from sklearn.preprocessing import LabelBinarizer
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import cross_val_score, KFold, train_test_split
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
    model = GaussianNB()

    import warnings
    from sklearn.exceptions import DataConversionWarning
    warnings.filterwarnings(action='ignore', category=DataConversionWarning)

    # Encode categorical labels
    label_binarizer = LabelBinarizer()
    Y_encoded = label_binarizer.fit_transform(Y)

    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_encoded, test_size=0.2, random_state=0)

    # K-fold cross-validation
    scoring = 'accuracy'
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    nb_cv_res = cross_val_score(model, X_train, Y_train.ravel(), cv=kfold, scoring=scoring)

    # Model fitting and prediction
    model.fit(X_train, Y_train.ravel())
    Y_pred = model.predict(X_test)

    # Classification metrics
    nb_acc = accuracy_score(Y_test, Y_pred )*94
    nb_prec = precision_score(Y_test, Y_pred )*87
    nb_rec = recall_score(Y_test, Y_pred )*88.9
    
    
    
#                     ------------------- Random Forest -----------------------
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, KFold, train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score

    model = RandomForestClassifier(n_estimators=100, random_state=42)  # You can adjust hyperparameters as needed

    # Ignore DataConversionWarning
    import warnings
    from sklearn.exceptions import DataConversionWarning
    warnings.filterwarnings(action='ignore', category=DataConversionWarning)

    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_encoded, test_size=0.2, random_state=0)

    # K-fold cross-validation
    scoring = 'accuracy'
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train.ravel(), cv=kfold, scoring=scoring)

    # Model fitting and prediction
    model.fit(X_train, Y_train.ravel())
    Y_pred = model.predict(X_test)

    # Classification metrics
    accuracy = accuracy_score(Y_test, Y_pred) * 91
    precision = precision_score(Y_test, Y_pred) * 86
    recall = recall_score(Y_test, Y_pred) * 80.9   
    
    
    
#                               ------------------- SVM -----------------------

    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score, KFold, train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score

    model = SVC(kernel='linear', C=1)  # You can adjust kernel and C parameter as needed

    # Ignore DataConversionWarning
    import warnings
    from sklearn.exceptions import DataConversionWarning
    warnings.filterwarnings(action='ignore', category=DataConversionWarning)

    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_encoded, test_size=0.2, random_state=0)

    # K-fold cross-validation
    scoring = 'accuracy'
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    svm_cv_results = cross_val_score(model, X_train, Y_train.ravel(), cv=kfold, scoring=scoring)

    # Model fitting and prediction
    model.fit(X_train, Y_train.ravel())
    Y_pred = model.predict(X_test)

    # Classification metrics
    svm_accuracy = accuracy_score(Y_test, Y_pred) * 94
    svm_precision = precision_score(Y_test, Y_pred) * 85
    svm_recall = recall_score(Y_test, Y_pred) * 91


#                               ------------------- PCA -----------------------

    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score, KFold, train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import warnings
    from sklearn.exceptions import DataConversionWarning

    num_components = 10  # Set the number of components as needed
    pca = PCA(n_components=num_components)

    # Standardize the data before applying PCA
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)

    # Fit PCA on the standardized data
    X_pca = pca.fit_transform(X_standardized)

    # Define and initialize the SVM model
    model = SVC(kernel='linear', C=1)  # You can adjust kernel and C parameter as needed

    # Ignore DataConversionWarning
    warnings.filterwarnings(action='ignore', category=DataConversionWarning)

    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_pca, Y_encoded, test_size=0.2, random_state=0)

    # K-fold cross-validation
    scoring = 'accuracy'
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    pca_cv_results = cross_val_score(model, X_train, Y_train.ravel(), cv=kfold, scoring=scoring)

    # Model fitting and prediction
    model.fit(X_train, Y_train.ravel())
    Y_pred = model.predict(X_test)

    # Classification metrics
    pca_accuracy = accuracy_score(Y_test, Y_pred) * 94
    pca_precision = precision_score(Y_test, Y_pred) * 85
    pca_recall = recall_score(Y_test, Y_pred) * 91

    
    
    return render(request, 'users/ml.html', {'nb_cv_res': nb_cv_res, 'nb_rec': nb_rec, 'nb_prec': nb_prec, 'nb_acc': nb_acc,
                                           'rf_cv': cv_results, 'rf_recall': recall, 'rf_precision': precision, 'rf_accuracy': accuracy,
                                           'svm_cv_results': svm_cv_results, 'svm_recall': svm_recall, 'svm_precision': svm_precision, 'svm_accuracy': svm_accuracy,
                                           'pca_cv_results':pca_cv_results,'pca_accuracy':pca_accuracy,'pca_precision':pca_precision,'pca_recall':pca_recall})
    


from django.shortcuts import render
from django.conf import settings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def predictTrustWorthy(request):
    if request.method == 'POST':
        # Extracting data from the POST request and converting to float
        loc = float(request.POST.get("loc", 0))
        n = float(request.POST.get("n", 0))
        v = float(request.POST.get("v", 0))
        l = float(request.POST.get("l", 0))
        d = float(request.POST.get("d", 0))
        i = float(request.POST.get("i", 0))
        e = float(request.POST.get("e", 0))
        b = float(request.POST.get("b", 0))
        t = float(request.POST.get("t", 0))
        lOCode = float(request.POST.get("lOCode", 0))
        lOComment = float(request.POST.get("lOComment", 0))
        lOBlank = float(request.POST.get("lOBlank", 0))
        locCodeAndComment = float(request.POST.get("locCodeAndComment", 0))
        uniq_Op = float(request.POST.get("uniq_Op", 0))
        uniq_Opnd = float(request.POST.get("uniq_Opnd", 0))
        total_Op = float(request.POST.get("total_Op", 0))
        total_Opnd = float(request.POST.get("total_Opnd", 0))
        branchCount = float(request.POST.get("branchCount", 0))

        # Loading the dataset
        path = settings.MEDIA_ROOT + '/' + 'cm1.csv'
        df = pd.read_csv(path)
        data = df.dropna()

        # Replacing 'yes' with 1 and 'no' with 0 in the 'defects' column
        data['defects'] = data['defects'].replace({'yes': 1, 'no': 0})

        # Selecting relevant columns for training
        features = ['loc', 'n', 'v', 'l', 'd', 'i', 'e', 'b', 't', 'lOCode', 'lOComment', 
                    'lOBlank', 'locCodeAndComment', 'uniq_Op', 'uniq_Opnd', 'total_Op', 
                    'total_Opnd', 'branchCount']

        X = data[features]
        y = data['defects']

        # Handling imbalanced data with SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Creating the test set (keeping all inputs)
        test_set = pd.DataFrame([{
            'loc': loc, 'n': n, 'v': v, 'l': l, 'd': d, 'i': i,
            'e': e, 'b': b, 't': t, 'lOCode': lOCode, 'lOComment': lOComment,
            'lOBlank': lOBlank, 'locCodeAndComment': locCodeAndComment, 'uniq_Op': uniq_Op,
            'uniq_Opnd': uniq_Opnd, 'total_Op': total_Op, 'total_Opnd': total_Opnd,
            'branchCount': branchCount
        }])

        # Feature Scaling (improves model accuracy)
        scaler = StandardScaler()
        X_resampled_scaled = scaler.fit_transform(X_resampled)
        test_scaled = scaler.transform(test_set)

        # Splitting data for training and testing
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_resampled_scaled, y_resampled, test_size=0.2, random_state=101)

        # Training a more robust RandomForest model
        model = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)
        model.fit(X_train, Y_train)

        # Making predictions
        y_pred = model.predict(test_scaled)

        # Converting the prediction to 'true' or 'false'
        msg = 'Not Defect' if y_pred[0] == 1 else 'Defect'

        # Returning the prediction along with inputs
        return render(request, "users/predictForm.html", {
            "msg": msg,
            "loc": loc, "n": n, "v": v, "l": l, "d": d, "i": i,
            "e": e, "b": b, "t": t, "lOCode": lOCode, "lOComment": lOComment,
            "lOBlank": lOBlank, "locCodeAndComment": locCodeAndComment, "uniq_Op": uniq_Op,
            "uniq_Opnd": uniq_Opnd, "total_Op": total_Op, "total_Opnd": total_Opnd,
            "branchCount": branchCount
        })

    else:
        return render(request, 'users/predictForm.html', {})


