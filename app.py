import pandas as pd
import streamlit as st
import ast
# public data types
possible_type = ['Categorical', 'Ordinal', 'Numerical']
possible_rescaling_method = ['MinMaxScaler', 'Logarithmic', 'None']
possible_ordinal_mapping = ['auto', 'user_defined']
possible_to_dummies = ['True', 'False']
possible_regression_model = ['Regression Tree', 'Linear Regression']
possible_classification_model = ['Logistic Regression', 'Random Forest', 'SV classifier', 'Kernel Method']


def handle_attri_perference(selected_attribute, attri_options, attr):
    if selected_attribute == attri_options[0]:
        st.write('anything')
        return
    st.sidebar.text("Attribute Info "+selected_attribute)
    attribute_type = st.sidebar.selectbox("Attribute Type", tuple(possible_type))
    attr[selected_attribute]['type'] = attribute_type
    if attr[selected_attribute]['type'] == 'Numerical':
        rescaling_method = st.sidebar.selectbox("Normalization/Rescaling",tuple(possible_rescaling_method))
        attr[selected_attribute]['NormalizationMethod'] = rescaling_method
    if attr[selected_attribute]['type'] == 'Ordinal':
        ordinal_mapping = st.sidebar.selectbox("Ordinal Mapping", tuple(possible_ordinal_mapping))
        if ordinal_mapping == 'user_defined':
            user_defined_ordinal_map = st.sidebar.text_area('ordinal map')
            attr[selected_attribute]['Ordinal_mapping'] = ast.literal_eval(user_defined_ordinal_map)
    if attr[selected_attribute]['type'] == 'Categorical':
        to_dummies = st.sidebar.selectbox("Use one-hot", tuple(possible_to_dummies))
        attr[selected_attribute]['toDummies'] = (to_dummies == 'True')
    return attr

# preprocess data, should be cached
@st.cache
def preprocess_data(df, attr):
    """
    Preprocess data based on user perference
    :param df:
    :param attr:
    :return:
    """
    pass

def render_infrastructure():
    # declare title and sidebar
    st.title("Interactive AI")
    st.sidebar.title("Settings")
    show_data = st.sidebar.checkbox('show data overview')
    # adk the user to upload the a dataset and offer a default dataset, which is the mushrooms dataset
    uploaded_file = st.file_uploader("Upload data here", type='csv')
    data = pd.read_csv('./data/mushrooms.csv')
    if uploaded_file is not None:
        data = uploaded_file
    if show_data:
        st.subheader('Data set overview')
        st.dataframe(data=data)
    # ask the user to select the prediction target
    columns = list(data.columns)

    prediction_target = st.sidebar.selectbox("Select prediction target", tuple(columns))
    prediction_type = st.sidebar.selectbox("Select prediction task", ('Regression', 'Classification'))
    columns.remove(prediction_target)
    # ask the user to adjust the attribute type
    # first generate default attribute type and preprocess approach
    attr = {}
    for attribute in columns:

        # type : type of the attribute, possible value:[Categorical, Ordinal, Numerical], default:[Categorical]
        # NormalizationMethod: way of normalizing the value, only viable for numerical values.
        #     MinMaxScaler: linearly rescale the attribute to the range of 0 to 1
        #     Logarithmic: first apply logarithmic transformation then use MinMaxScaler. Should be applied to attributes
        #         with extremely large range
        #     None: does not apply rescaling techniques
        #     possible value:[MinMaxScaler, Logarithmic, None],
        #     default:[MinMaxScaler]
        # Ordinal_mapping: the way of mapping ordinal value. Only viable for Ordinal data
        #     Possible option:[auto, user_defined]. default:[auto]
        #     auto: use alphabetic order or numerical order to decide the order of ordinal values.
        #     user_defined: let user to define a dictionary to map ordinal values to integer that does not contain same
        #         output values. if the user defined dictionary yields failure then go back to auto.
        # toDummies: if to change categorical data to one-hot representation. Only viable for categorical data
        #     Possible option:[True, False], default:[True]
        #
        attr[attribute] = {'type':'Categorical','NormalizationMethod':'MinMaxScaler', 'Ordinal_mapping':'auto',
                           'toDummies':True}

    # render selection box
    attri_options = ['Select Attribute']+ columns

    selected_attribute = st.sidebar.selectbox('Select attribute for processing setting', attri_options)
    attr = handle_attri_perference(selected_attribute, attri_options, attr)

    # if prediction_type
    return (data, prediction_target, prediction_type, attr)

def main():
    render_infrastructure()


if __name__ == '__main__':
    main()
