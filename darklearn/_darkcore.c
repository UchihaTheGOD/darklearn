/* darklearn/_darkcore.c — DarkCore's optional C core.
 *
 * Same kernels as darkcore.py, in C: tiled matmul (float, row-major,
 * block 32, i-k-j order) and fused linear+ReLU. No external libraries,
 * only the C stdlib and the CPython C API.
 *
 * Inputs/outputs cross the boundary as raw float32 bytes: the Python
 * side passes array('f') objects (buffer protocol) and gets `bytes`
 * back, which it wraps with array('f').frombytes(). That keeps this
 * file free of any knowledge of darklearn's Python classes.
 *
 * The C core is strictly optional: if it is not built, darkcore.py
 * falls back to the pure-Python kernels with no error. Build with:
 *     python3 scripts/build_c.py
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>

/* --- helpers ---------------------------------------------------------- */

static int
get_float_buffer(PyObject *obj, Py_buffer *buf, const char *name)
{
    if (PyObject_GetBuffer(obj, buf, PyBUF_SIMPLE) < 0) {
        PyErr_Format(PyExc_TypeError, "%s must support the buffer protocol", name);
        return -1;
    }
    if (buf->len % 4 != 0) {
        PyBuffer_Release(buf);
        PyErr_Format(PyExc_ValueError, "%s length is not a multiple of 4 bytes", name);
        return -1;
    }
    return 0;
}

static int
check_len(Py_buffer *buf, Py_ssize_t need, const char *name)
{
    if (buf->len < need * (Py_ssize_t)sizeof(float)) {
        PyErr_Format(PyExc_ValueError,
                     "%s too short: need %zd float32 elements, got %zd bytes",
                     name, need, buf->len);
        return -1;
    }
    return 0;
}

/* --- tiled matmul ----------------------------------------------------- */
/* matmul(a, b, m, k, n, block=32) -> bytes of shape (m, n), float32.
 * a: (m, k), b: (k, n), both row-major. i-k-j order, tiles of `block`.
 */
static PyObject *
darkcore_matmul(PyObject *self, PyObject *args)
{
    PyObject *a_obj, *b_obj;
    Py_ssize_t m, k, n;
    long block = 32;
    if (!PyArg_ParseTuple(args, "OOnnn|l", &a_obj, &b_obj, &m, &k, &n, &block))
        return NULL;
    if (m < 0 || k < 0 || n < 0 || block <= 0) {
        PyErr_SetString(PyExc_ValueError, "m, k, n must be >= 0 and block > 0");
        return NULL;
    }

    Py_buffer a_buf, b_buf;
    if (get_float_buffer(a_obj, &a_buf, "a") < 0) return NULL;
    if (get_float_buffer(b_obj, &b_buf, "b") < 0) {
        PyBuffer_Release(&a_buf);
        return NULL;
    }
    if (check_len(&a_buf, m * k, "a") < 0 ||
        check_len(&b_buf, k * n, "b") < 0) {
        PyBuffer_Release(&a_buf);
        PyBuffer_Release(&b_buf);
        return NULL;
    }

    PyObject *out = PyBytes_FromStringAndSize(NULL, m * n * (Py_ssize_t)sizeof(float));
    if (out == NULL) {
        PyBuffer_Release(&a_buf);
        PyBuffer_Release(&b_buf);
        return NULL;
    }
    float *a = (float *)a_buf.buf;
    float *b = (float *)b_buf.buf;
    float *o = (float *)PyBytes_AS_STRING(out);
    memset(o, 0, (size_t)(m * n * sizeof(float)));

    for (Py_ssize_t ii = 0; ii < m; ii += block) {
        Py_ssize_t i_end = ii + block < m ? ii + block : m;
        for (Py_ssize_t kk = 0; kk < k; kk += block) {
            Py_ssize_t k_end = kk + block < k ? kk + block : k;
            for (Py_ssize_t jj = 0; jj < n; jj += block) {
                Py_ssize_t j_end = jj + block < n ? jj + block : n;
                for (Py_ssize_t i = ii; i < i_end; i++) {
                    float *o_row = o + i * n;
                    const float *a_row = a + i * k;
                    for (Py_ssize_t k2 = kk; k2 < k_end; k2++) {
                        float aik = a_row[k2];
                        const float *b_row = b + k2 * n;
                        for (Py_ssize_t j = jj; j < j_end; j++)
                            o_row[j] += aik * b_row[j];
                    }
                }
            }
        }
    }

    PyBuffer_Release(&a_buf);
    PyBuffer_Release(&b_buf);
    return out;
}

/* --- fused linear + relu ---------------------------------------------- */
/* fused_linear_relu(x, w, b, batch, in_f, out_f) -> (out_bytes, mask_bytes)
 * x: (batch, in_f), w: (out_f, in_f), b: (out_f,).
 * out: (batch, out_f) float32 bytes; mask: (batch*out_f) bytes, 1 where
 * the pre-ReLU value was > 0. One pass over x/W, no intermediates.
 */
static PyObject *
darkcore_fused_linear_relu(PyObject *self, PyObject *args)
{
    PyObject *x_obj, *w_obj, *b_obj;
    Py_ssize_t batch, in_f, out_f;
    if (!PyArg_ParseTuple(args, "OOOnnn", &x_obj, &w_obj, &b_obj,
                          &batch, &in_f, &out_f))
        return NULL;
    if (batch < 0 || in_f < 0 || out_f < 0) {
        PyErr_SetString(PyExc_ValueError, "batch, in_f, out_f must be >= 0");
        return NULL;
    }

    Py_buffer x_buf, w_buf, b_buf;
    if (get_float_buffer(x_obj, &x_buf, "x") < 0) return NULL;
    if (get_float_buffer(w_obj, &w_buf, "w") < 0) {
        PyBuffer_Release(&x_buf);
        return NULL;
    }
    if (get_float_buffer(b_obj, &b_buf, "b") < 0) {
        PyBuffer_Release(&x_buf);
        PyBuffer_Release(&w_buf);
        return NULL;
    }
    if (check_len(&x_buf, batch * in_f, "x") < 0 ||
        check_len(&w_buf, out_f * in_f, "w") < 0 ||
        check_len(&b_buf, out_f, "b") < 0) {
        PyBuffer_Release(&x_buf);
        PyBuffer_Release(&w_buf);
        PyBuffer_Release(&b_buf);
        return NULL;
    }

    PyObject *out = PyBytes_FromStringAndSize(
        NULL, batch * out_f * (Py_ssize_t)sizeof(float));
    PyObject *mask = PyBytes_FromStringAndSize(NULL, batch * out_f);
    if (out == NULL || mask == NULL) {
        Py_XDECREF(out);
        Py_XDECREF(mask);
        PyBuffer_Release(&x_buf);
        PyBuffer_Release(&w_buf);
        PyBuffer_Release(&b_buf);
        return NULL;
    }
    float *x = (float *)x_buf.buf;
    float *w = (float *)w_buf.buf;
    float *b = (float *)b_buf.buf;
    float *o = (float *)PyBytes_AS_STRING(out);
    unsigned char *mk = (unsigned char *)PyBytes_AS_STRING(mask);
    memset(o, 0, (size_t)(batch * out_f * sizeof(float)));

    for (Py_ssize_t i = 0; i < batch; i++) {
        float *o_row = o + i * out_f;
        const float *x_row = x + i * in_f;
        for (Py_ssize_t k2 = 0; k2 < in_f; k2++) {
            float xv = x_row[k2];
            for (Py_ssize_t j = 0; j < out_f; j++)
                o_row[j] += xv * w[j * in_f + k2];
        }
    }
    for (Py_ssize_t i = 0; i < batch * out_f; i++) {
        float v = o[i] + b[i % out_f];
        if (v > 0.0f) {
            o[i] = v;
            mk[i] = 1;
        } else {
            o[i] = 0.0f;
            mk[i] = 0;
        }
    }

    PyBuffer_Release(&x_buf);
    PyBuffer_Release(&w_buf);
    PyBuffer_Release(&b_buf);
    PyObject *res = PyTuple_Pack(2, out, mask);
    Py_DECREF(out);
    Py_DECREF(mask);
    return res;
}

/* --- module ----------------------------------------------------------- */

static PyMethodDef DarkcoreMethods[] = {
    {"matmul", darkcore_matmul, METH_VARARGS,
     "matmul(a, b, m, k, n, block=32) -> bytes: tiled float32 matmul."},
    {"fused_linear_relu", darkcore_fused_linear_relu, METH_VARARGS,
     "fused_linear_relu(x, w, b, batch, in_f, out_f) -> (bytes, bytes)."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef darkcore_module = {
    PyModuleDef_HEAD_INIT, "_darkcore",
    "DarkCore C core: tiled matmul + fused linear+ReLU (optional).",
    -1, DarkcoreMethods,
};

PyMODINIT_FUNC
PyInit__darkcore(void)
{
    return PyModule_Create(&darkcore_module);
}
