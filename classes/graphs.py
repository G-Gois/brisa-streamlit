import streamlit as st
import altair as alt
from .sample_data import SampleData
class Graphs:
    def __init__(self) -> None:
        self.sample_data = SampleData().get_sales_data()
    def predicted_vs_actual(self,model,X_test):
        pred = model.predict(X_test)
        pred_ci = pred.conf_int()
        ax = X_test['2022':].plot(label='Observado')
        pred.predicted_mean.plot(ax=ax, label='Previsão', alpha=.7, figsize=(14, 7))
        ax.fill_between(pred_ci.index,
                        pred_ci.iloc[:, 0],
                        pred_ci.iloc[:, 1], color='k', alpha=.2)
        ax.set_xlabel('Data')
        ax.set_ylabel('Número de pedidos')
        ax.set_title('Previsão vs. Observado')
        st.line_chart(ax)

    def line(self,predictions):
        st.line_chart(predictions,x="Actual",y="Predicted")

    def multiline(self):
        chart = alt.Chart(self.sample_data).mark_line(width=3000).encode(
            x='Date',
            y='Sales',
            tooltip=['Date', 'Sales'],
            color='Predicted',
            text='Predicted'
        )
        st.altair_chart(chart)