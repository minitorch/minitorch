import inspect
import streamlit as st

img_id_counter = 0


def get_image_id():
    global img_id_counter
    img_id_counter += 1
    return img_id_counter


def get_img_tag(src, width=None):
    img_id = get_image_id()
    if width is not None:
        style = """
<style>.img-{} {{
    float: left;
    width: {}px;
}}
</style>
        """.format(
            img_id, width
        )
    else:
        style = ""
    return """
        <img src="{}" class="img-{}" alt="img-{}" />
        {}
    """.format(
        src, img_id, img_id, style
    )


def render_function(fn):
    st.markdown(
        """
```python
%s

```"""
        % inspect.getsource(fn)
    )
