import streamlit as st
st.write('Hello, world!')
def bubble_sort(things):

    needs_pass = True

    while needs_pass:

        needs_pass = False

        for idx in range(1, len(things)):

            if things[idx - 1] > things[idx]:

                things[idx - 1], things[idx] = things[idx], things[idx - 1]

                needs_pass = True
